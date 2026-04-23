// RUN: enzymexlamlir-opt --inline-mcmc-regions --sicm %s | FileCheck %s

// ============================================================================
// Test: Full Comrade NUFFT chain — two-iteration SICM fixpoint
// ============================================================================
//
// This test mirrors the exact pattern from the Comrade EHT benchmark:
//   scatter(zeros[2,R,C], idx[K,2], updates[2,K])
//   → transpose([0,2,1]) → [2,C,R]
//   → slice(channel 0/1) → reshape → complex(re[C,R], im[C,R])
//   → fft([C,R]) → reshape[P] → dot_general(B[P,M], v[P], contracting=[0]x[0])
//
// SICM absorbs this in two fixpoint iterations:
//   Iteration 1: DotAbsorbFFTHLO absorbs FFT into B
//     B[P,M] → reshape[C,R,M] → transpose[M,C,R] → fft → transpose[C,R,M] → reshape[P,M]
//     This is hoisted before mcmc_region.
//   Iteration 2: DotAbsorbScatterHLO (Case 3) absorbs scatter chain
//     gather(fft(B), linear_idx) → gathered_B[K,M]
//     Hoisted before mcmc_region.
//
// Result: dot contracts on K=2 instead of P=12, no FFT/scatter inside loop.
//
// Using scaled-down Comrade dims: R=3, C=4, K=2, M=5, P=R*C=12

// CHECK-LABEL: func.func @test_dot_absorb_comrade_chain
// CHECK-SAME: %[[RNG:.*]]: tensor<2xui64>, %[[PRIOR:.*]]: tensor<f64>, %[[B:.*]]: tensor<12x5xcomplex<f64>>, %[[IDX:.*]]: tensor<2x2xi64>, %[[TRACE:.*]]: tensor<1x1xf64>
// FFT absorbed into B and hoisted (iteration 1), then gather from result (iteration 2)
// CHECK: stablehlo.reshape %[[B]]
// CHECK: stablehlo.transpose
// CHECK: stablehlo.fft
// CHECK: stablehlo.transpose
// CHECK: stablehlo.reshape
// After FFT absorption, scatter is absorbed via gather
// CHECK: stablehlo.gather
// CHECK-NOT: stablehlo.scatter
// CHECK-NOT: stablehlo.fft
// CHECK: enzyme.mcmc_region
// Inside: no scatter, no fft, no transpose chain, just dot on gathered matrix
// CHECK: enzyme.sample_region
// CHECK-NOT: stablehlo.scatter
// CHECK-NOT: stablehlo.fft
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

  func.func @test_dot_absorb_comrade_chain(%rng : tensor<2xui64>, %prior : tensor<f64>,
                                            %B : tensor<12x5xcomplex<f64>>,
                                            %idx : tensor<2x2xi64>,
                                            %trace : tensor<1x1xf64>)
      -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>) {
    %step_size = arith.constant dense<0.1> : tensor<f64>

    %result:8 = enzyme.mcmc @model_comrade_chain(%rng, %prior, %B, %idx) given %trace
        step_size = %step_size {
      selection = [[#enzyme.symbol<1>]],
      all_addresses = [[#enzyme.symbol<1>]],
      hmc_config = #enzyme.hmc_config<trajectory_length = 1.0>,
      num_samples = 1 : i64,
      thinning = 1 : i64,
      num_warmup = 0 : i64
    } : (tensor<2xui64>, tensor<f64>, tensor<12x5xcomplex<f64>>, tensor<2x2xi64>, tensor<1x1xf64>, tensor<f64>)
        -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>, tensor<1x1xf64>, tensor<1x1xf64>, tensor<f64>, tensor<f64>, tensor<1x1xf64>)

    return %result#0, %result#1, %result#2 : tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>
  }

  func.func @model_comrade_chain(%rng : tensor<2xui64>, %prior : tensor<f64>,
                                  %B : tensor<12x5xcomplex<f64>>,
                                  %idx : tensor<2x2xi64>)
      -> (tensor<2xui64>, tensor<f64>) {
    %std = arith.constant dense<1.0> : tensor<f64>

    %x:2 = enzyme.sample @normal(%rng, %prior, %std) {
      logpdf = @normal_logpdf,
      symbol = #enzyme.symbol<1>
    } : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)

    // Build sample-dependent updates: [channel=2, K=2]
    %x_bcast = stablehlo.broadcast_in_dim %x#1, dims = [] : (tensor<f64>) -> tensor<2x2xf64>

    // Scatter into zeros[2, 3, 4] with channel dim (matches Comrade: zeros[2, R, C]):
    //   update_window_dims = [0] (channel dim of updates)
    //   scatter_dims_to_operand = [1, 2] (indices map to row, col)
    //   inserted_window_dims = [1, 2]
    %zeros = stablehlo.constant dense<0.0> : tensor<2x3x4xf64>
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
    }) : (tensor<2x3x4xf64>, tensor<2x2xi64>, tensor<2x2xf64>) -> tensor<2x3x4xf64>

    // Transpose: [2,3,4] -> [2,4,3] (perm [0,2,1]) — matches Comrade exactly
    %transposed = stablehlo.transpose %scattered, dims = [0, 2, 1]
        : (tensor<2x3x4xf64>) -> tensor<2x4x3xf64>

    // Slice channel 0 (real part): [0:1, 0:4, 0:3]
    %re_slice = "stablehlo.slice"(%transposed) {
      start_indices = array<i64: 0, 0, 0>,
      limit_indices = array<i64: 1, 4, 3>,
      strides = array<i64: 1, 1, 1>
    } : (tensor<2x4x3xf64>) -> tensor<1x4x3xf64>
    %re = stablehlo.reshape %re_slice : (tensor<1x4x3xf64>) -> tensor<4x3xf64>

    // Slice channel 1 (imaginary part): [1:2, 0:4, 0:3]
    %im_slice = "stablehlo.slice"(%transposed) {
      start_indices = array<i64: 1, 0, 0>,
      limit_indices = array<i64: 2, 4, 3>,
      strides = array<i64: 1, 1, 1>
    } : (tensor<2x4x3xf64>) -> tensor<1x4x3xf64>
    %im = stablehlo.reshape %im_slice : (tensor<1x4x3xf64>) -> tensor<4x3xf64>

    // Form complex grid: [4, 3] complex
    %complex_grid = stablehlo.complex %re, %im : tensor<4x3xcomplex<f64>>

    // FFT on the complex grid: [4, 3] -> [4, 3]
    %fft_result = stablehlo.fft %complex_grid, type = FFT, length = [4, 3] : (tensor<4x3xcomplex<f64>>) -> tensor<4x3xcomplex<f64>>

    // Reshape to [12] for dot (P = 4*3 = 12)
    %reshaped = stablehlo.reshape %fft_result : (tensor<4x3xcomplex<f64>>) -> tensor<12xcomplex<f64>>

    // dot_general with contracting dim 0 on LHS (matches Comrade: [P,M] x [P] -> [M])
    %y = stablehlo.dot_general %B, %reshaped, contracting_dims = [0] x [0]
        : (tensor<12x5xcomplex<f64>>, tensor<12xcomplex<f64>>) -> tensor<5xcomplex<f64>>

    // Extract real part of first element as scalar
    %real_y = stablehlo.real %y : (tensor<5xcomplex<f64>>) -> tensor<5xf64>
    %slice = "stablehlo.slice"(%real_y) {start_indices = array<i64: 0>, limit_indices = array<i64: 1>, strides = array<i64: 1>} : (tensor<5xf64>) -> tensor<1xf64>
    %result = stablehlo.reshape %slice : (tensor<1xf64>) -> tensor<f64>

    return %x#0, %result : tensor<2xui64>, tensor<f64>
  }
}
