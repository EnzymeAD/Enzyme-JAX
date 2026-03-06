// RUN: enzymexlamlir-opt --inline-mcmc-regions --sicm %s | FileCheck %s

// ============================================================================
// Test: DotAbsorbTranspose — absorb transpose from RHS into LHS matrix
// ============================================================================
//
// Model: dot_general(A:[2,4], reshape(transpose(X:[2,2], [1,0])) -> [4,1])
//        where A is sample-invariant.
// SICM absorbs: A' = reshape(transpose(reshape(A,[2,2,2]), [0,2,1]), [2,4])
// Then hoists the pre-transposed matrix A' outside mcmc_region.

// CHECK-LABEL: func.func @test_dot_absorb_transpose
// CHECK-SAME: %[[RNG:.*]]: tensor<2xui64>, %[[PRIOR:.*]]: tensor<f64>, %[[A:.*]]: tensor<2x4xf64>, %[[TRACE:.*]]: tensor<1x1xf64>
// The transposed A should be hoisted before mcmc_region
// CHECK: stablehlo.reshape %[[A]] : (tensor<2x4xf64>) -> tensor<2x2x2xf64>
// CHECK-NEXT: stablehlo.transpose
// CHECK-NEXT: stablehlo.reshape
// CHECK-NOT: stablehlo.transpose
// CHECK: enzyme.mcmc_region
// Inside: no transpose, just dot_general with pre-transposed matrix
// CHECK: enzyme.sample_region
// CHECK-NOT: stablehlo.transpose
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

  func.func @test_dot_absorb_transpose(%rng : tensor<2xui64>, %prior : tensor<f64>,
                                        %A : tensor<2x4xf64>,
                                        %trace : tensor<1x1xf64>)
      -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>) {
    %step_size = arith.constant dense<0.1> : tensor<f64>

    %result:8 = enzyme.mcmc @model_transpose(%rng, %prior, %A) given %trace
        step_size = %step_size {
      selection = [[#enzyme.symbol<1>]],
      all_addresses = [[#enzyme.symbol<1>]],
      hmc_config = #enzyme.hmc_config<trajectory_length = 1.0>,
      num_samples = 1 : i64,
      thinning = 1 : i64,
      num_warmup = 0 : i64
    } : (tensor<2xui64>, tensor<f64>, tensor<2x4xf64>, tensor<1x1xf64>, tensor<f64>)
        -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>, tensor<1x1xf64>, tensor<1x1xf64>, tensor<f64>, tensor<f64>, tensor<1x1xf64>)

    return %result#0, %result#1, %result#2 : tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>
  }

  func.func @model_transpose(%rng : tensor<2xui64>, %prior : tensor<f64>,
                              %A : tensor<2x4xf64>)
      -> (tensor<2xui64>, tensor<f64>) {
    %std = arith.constant dense<1.0> : tensor<f64>

    %x:2 = enzyme.sample @normal(%rng, %prior, %std) {
      logpdf = @normal_logpdf,
      symbol = #enzyme.symbol<1>
    } : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)

    // Build a 2x2 matrix from the sample
    %x_mat = stablehlo.broadcast_in_dim %x#1, dims = [] : (tensor<f64>) -> tensor<2x2xf64>

    // Transpose [1,0]
    %transposed = stablehlo.transpose %x_mat, dims = [1, 0] : (tensor<2x2xf64>) -> tensor<2x2xf64>

    // Reshape to [4, 1]
    %reshaped = stablehlo.reshape %transposed : (tensor<2x2xf64>) -> tensor<4x1xf64>

    // dot_general(A, reshaped)
    %y = stablehlo.dot_general %A, %reshaped, contracting_dims = [1] x [0]
        : (tensor<2x4xf64>, tensor<4x1xf64>) -> tensor<2x1xf64>

    %slice = "stablehlo.slice"(%y) {start_indices = array<i64: 0, 0>, limit_indices = array<i64: 1, 1>, strides = array<i64: 1, 1>} : (tensor<2x1xf64>) -> tensor<1x1xf64>
    %result = stablehlo.reshape %slice : (tensor<1x1xf64>) -> tensor<f64>

    return %x#0, %result : tensor<2xui64>, tensor<f64>
  }
}
