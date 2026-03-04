// RUN: enzymexlamlir-opt --inline-mcmc-regions --sicm %s | FileCheck %s

// ============================================================================
// Test: DotAbsorbScatter Case 3 — channel-scatter chain matching
// ============================================================================
//
// This tests the pattern that arises in NUFFT after FFT absorption:
//   scatter(zeros[2,2,3], idx[K=2,2], updates[2,K=2])
//   → transpose([0,2,1]) → [2,3,2]
//   → slice(channel 0/1) → reshape → complex(re[3,2], im[3,2])
//   → reshape[6] → dot_general(B[6,5], v[6], contracting=[0]x[0])
//
// The scatter has a channel dim (size 2 for re/im), with:
//   update_window_dims = [0], scatter_dims_to_operand = [1, 2]
//
// SICM absorbs the scatter chain: computes linear indices from the scatter
// indices (accounting for the transpose permutation), gathers from B, and
// forms complex data from the scatter updates.
//
// Result: dot contracts on K=2 instead of P=6.

// CHECK-LABEL: func.func @test_dot_absorb_scatter_channel
// CHECK-SAME: %[[RNG:.*]]: tensor<2xui64>, %[[PRIOR:.*]]: tensor<f64>, %[[B:.*]]: tensor<6x5xcomplex<f64>>, %[[IDX:.*]]: tensor<2x2xi64>, %[[TRACE:.*]]: tensor<1x1xf64>
// gather(B, linear_idx) should be hoisted before mcmc_region
// CHECK: stablehlo.gather
// CHECK-NOT: stablehlo.scatter
// CHECK: enzyme.mcmc_region
// Inside: no scatter/transpose/fft, dot uses gathered matrix with K < P
// CHECK: enzyme.sample_region
// CHECK-NOT: stablehlo.scatter
// CHECK: stablehlo.dot_general
// CHECK: enzyme.yield

module {
  func.func private @normal(%rng : tensor<2xui64>, %mean : tensor<f64>, %std : tensor<f64>)
      -> (tensor<2xui64>, tensor<f64>) {
    %sample = arith.addf %mean, %std : tensor<f64>
    return %rng, %sample : tensor<2xui64>, tensor<f64>
  }

  func.func private @normal_logpdf(%x : tensor<f64>, %mean : tensor<f64>, %std : tensor<f64>)
      -> tensor<f64> {
    %neg = arith.negf %x : tensor<f64>
    return %neg : tensor<f64>
  }

  func.func @test_dot_absorb_scatter_channel(%rng : tensor<2xui64>, %prior : tensor<f64>,
                                              %B : tensor<6x5xcomplex<f64>>,
                                              %idx : tensor<2x2xi64>,
                                              %trace : tensor<1x1xf64>)
      -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>) {
    %step_size = arith.constant dense<0.1> : tensor<f64>

    %result:3 = enzyme.mcmc @model_scatter_channel(%rng, %prior, %B, %idx) given %trace
        step_size = %step_size {
      selection = [[#enzyme.symbol<1>]],
      all_addresses = [[#enzyme.symbol<1>]],
      hmc_config = #enzyme.hmc_config<trajectory_length = 1.0>,
      num_samples = 1 : i64,
      thinning = 1 : i64,
      num_warmup = 0 : i64
    } : (tensor<2xui64>, tensor<f64>, tensor<6x5xcomplex<f64>>, tensor<2x2xi64>, tensor<1x1xf64>, tensor<f64>)
        -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>)

    return %result#0, %result#1, %result#2 : tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>
  }

  func.func @model_scatter_channel(%rng : tensor<2xui64>, %prior : tensor<f64>,
                                    %B : tensor<6x5xcomplex<f64>>,
                                    %idx : tensor<2x2xi64>)
      -> (tensor<2xui64>, tensor<f64>) {
    %std = arith.constant dense<1.0> : tensor<f64>

    %x:2 = enzyme.sample @normal(%rng, %prior, %std) {
      logpdf = @normal_logpdf,
      symbol = #enzyme.symbol<1>
    } : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)

    // Build sample-dependent updates: [channel=2, K=2]
    %x_bcast = stablehlo.broadcast_in_dim %x#1, dims = [] : (tensor<f64>) -> tensor<2x2xf64>

    // Scatter into zeros[2, 2, 3] with channel dim:
    //   update_window_dims = [0] (channel dim of updates)
    //   scatter_dims_to_operand = [1, 2] (indices map to row, col)
    //   inserted_window_dims = [1, 2]
    %zeros = stablehlo.constant dense<0.0> : tensor<2x2x3xf64>
    %scattered = "stablehlo.scatter"(%zeros, %idx, %x_bcast) <{
      scatter_dimension_numbers = #stablehlo.scatter<
        update_window_dims = [0],
        inserted_window_dims = [1, 2],
        scatter_dims_to_operand_dims = [1, 2],
        index_vector_dim = 1
      >,
      indices_are_sorted = false,
      unique_indices = true
    }> ({
    ^bb0(%arg0: tensor<f64>, %arg1: tensor<f64>):
      stablehlo.return %arg1 : tensor<f64>
    }) : (tensor<2x2x3xf64>, tensor<2x2xi64>, tensor<2x2xf64>) -> tensor<2x2x3xf64>

    // Transpose: [2,2,3] -> [2,3,2] (perm [0,2,1])
    %transposed = stablehlo.transpose %scattered, dims = [0, 2, 1]
        : (tensor<2x2x3xf64>) -> tensor<2x3x2xf64>

    // Slice channel 0 (real part)
    %re_slice = "stablehlo.slice"(%transposed) {
      start_indices = array<i64: 0, 0, 0>,
      limit_indices = array<i64: 1, 3, 2>,
      strides = array<i64: 1, 1, 1>
    } : (tensor<2x3x2xf64>) -> tensor<1x3x2xf64>
    %re = stablehlo.reshape %re_slice : (tensor<1x3x2xf64>) -> tensor<3x2xf64>

    // Slice channel 1 (imaginary part)
    %im_slice = "stablehlo.slice"(%transposed) {
      start_indices = array<i64: 1, 0, 0>,
      limit_indices = array<i64: 2, 3, 2>,
      strides = array<i64: 1, 1, 1>
    } : (tensor<2x3x2xf64>) -> tensor<1x3x2xf64>
    %im = stablehlo.reshape %im_slice : (tensor<1x3x2xf64>) -> tensor<3x2xf64>

    // Form complex grid
    %complex_grid = stablehlo.complex %re, %im : tensor<3x2xcomplex<f64>>

    // Reshape to [6] for dot
    %reshaped = stablehlo.reshape %complex_grid : (tensor<3x2xcomplex<f64>>) -> tensor<6xcomplex<f64>>

    // dot_general with contracting dim 0 on LHS
    %y = stablehlo.dot_general %B, %reshaped, contracting_dims = [0] x [0]
        : (tensor<6x5xcomplex<f64>>, tensor<6xcomplex<f64>>) -> tensor<5xcomplex<f64>>

    // Extract real part of first element as scalar
    %real_y = stablehlo.real %y : (tensor<5xcomplex<f64>>) -> tensor<5xf64>
    %slice = "stablehlo.slice"(%real_y) {start_indices = array<i64: 0>, limit_indices = array<i64: 1>, strides = array<i64: 1>} : (tensor<5xf64>) -> tensor<1xf64>
    %result = stablehlo.reshape %slice : (tensor<1xf64>) -> tensor<f64>

    return %x#0, %result : tensor<2xui64>, tensor<f64>
  }
}
