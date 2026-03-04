// RUN: enzymexlamlir-opt --inline-mcmc-regions --sicm %s | FileCheck %s

// ============================================================================
// Test: DotAbsorbScatter with contracting dim 0
// ============================================================================
//
// Model: dot_general(A:[4,3], reshape(scatter(zeros:[2,2], idx, data:[2])) -> [4])
//        where A and idx are sample-invariant, contracting_dims = [0] x [0].
// SICM gathers from A along dim 0: gathered_A = gather(A, linear_idx) : [2, 3]
// Then hoists gathered_A outside mcmc_region. Dot shrinks from [4,3] to [2,3].

// CHECK-LABEL: func.func @test_dot_absorb_scatter_dim0
// CHECK-SAME: %[[RNG:.*]]: tensor<2xui64>, %[[PRIOR:.*]]: tensor<f64>, %[[A:.*]]: tensor<4x3xf64>, %[[IDX:.*]]: tensor<2x2xi32>, %[[TRACE:.*]]: tensor<1x1xf64>
// gather(A, linear_idx) should be hoisted before mcmc_region
// CHECK: stablehlo.gather
// CHECK-NOT: stablehlo.scatter
// CHECK: enzyme.mcmc_region
// Inside: no scatter, dot uses gathered matrix
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

  func.func @test_dot_absorb_scatter_dim0(%rng : tensor<2xui64>, %prior : tensor<f64>,
                                           %A : tensor<4x3xf64>, %idx : tensor<2x2xi32>,
                                           %trace : tensor<1x1xf64>)
      -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>) {
    %step_size = arith.constant dense<0.1> : tensor<f64>

    %result:3 = enzyme.mcmc @model_scatter_dim0(%rng, %prior, %A, %idx) given %trace
        step_size = %step_size {
      selection = [[#enzyme.symbol<1>]],
      all_addresses = [[#enzyme.symbol<1>]],
      hmc_config = #enzyme.hmc_config<trajectory_length = 1.0>,
      num_samples = 1 : i64,
      thinning = 1 : i64,
      num_warmup = 0 : i64
    } : (tensor<2xui64>, tensor<f64>, tensor<4x3xf64>, tensor<2x2xi32>, tensor<1x1xf64>, tensor<f64>)
        -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>)

    return %result#0, %result#1, %result#2 : tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>
  }

  func.func @model_scatter_dim0(%rng : tensor<2xui64>, %prior : tensor<f64>,
                                  %A : tensor<4x3xf64>, %idx : tensor<2x2xi32>)
      -> (tensor<2xui64>, tensor<f64>) {
    %std = arith.constant dense<1.0> : tensor<f64>

    %x:2 = enzyme.sample @normal(%rng, %prior, %std) {
      logpdf = @normal_logpdf,
      symbol = #enzyme.symbol<1>
    } : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)

    // Build sample-dependent data vector
    %data = stablehlo.broadcast_in_dim %x#1, dims = [] : (tensor<f64>) -> tensor<2xf64>

    // Scatter data into a 2x2 zeros grid using invariant indices
    %zeros = stablehlo.constant dense<0.0> : tensor<2x2xf64>
    %scattered = "stablehlo.scatter"(%zeros, %idx, %data) <{
      scatter_dimension_numbers = #stablehlo.scatter<
        inserted_window_dims = [0, 1],
        scatter_dims_to_operand_dims = [0, 1],
        index_vector_dim = 1
      >,
      indices_are_sorted = false,
      unique_indices = true
    }> ({
    ^bb0(%arg0: tensor<f64>, %arg1: tensor<f64>):
      stablehlo.return %arg1 : tensor<f64>
    }) : (tensor<2x2xf64>, tensor<2x2xi32>, tensor<2xf64>) -> tensor<2x2xf64>

    // Reshape to [4] for dot
    %reshaped = stablehlo.reshape %scattered : (tensor<2x2xf64>) -> tensor<4xf64>

    // dot_general with contracting dim 0 on LHS
    %y = stablehlo.dot_general %A, %reshaped, contracting_dims = [0] x [0]
        : (tensor<4x3xf64>, tensor<4xf64>) -> tensor<3xf64>

    %slice = "stablehlo.slice"(%y) {start_indices = array<i64: 0>, limit_indices = array<i64: 1>, strides = array<i64: 1>} : (tensor<3xf64>) -> tensor<1xf64>
    %result = stablehlo.reshape %slice : (tensor<1xf64>) -> tensor<f64>

    return %x#0, %result : tensor<2xui64>, tensor<f64>
  }
}
