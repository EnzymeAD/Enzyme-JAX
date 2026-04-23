// RUN: enzymexlamlir-opt --inline-mcmc-regions --sicm %s | FileCheck %s

// ============================================================================
// Test: DotAbsorbTranspose with contracting dim 0
// ============================================================================
//
// Model: dot_general(A:[6,5], reshape(transpose(X:[2,3], [1,0])) -> [6])
//        where A is sample-invariant, contracting_dims = [0] x [0].
// SICM absorbs the transpose into A:
//   A[6,5] → reshape[3,2,5] → transpose([1,0,2])[2,3,5] → reshape[6,5]

// CHECK-LABEL: func.func @test_dot_absorb_transpose_dim0
// CHECK-SAME: %[[RNG:.*]]: tensor<2xui64>, %[[PRIOR:.*]]: tensor<f64>, %[[A:.*]]: tensor<6x5xf64>, %[[TRACE:.*]]: tensor<1x1xf64>
// Inverse transpose of A hoisted before mcmc_region
// CHECK: stablehlo.reshape %[[A]] : (tensor<6x5xf64>) -> tensor<3x2x5xf64>
// CHECK-NEXT: stablehlo.transpose
// CHECK-NEXT: stablehlo.reshape
// CHECK-NOT: stablehlo.transpose
// CHECK: enzyme.mcmc_region
// Inside: no transpose, dot_general uses pre-transposed matrix
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

  func.func @test_dot_absorb_transpose_dim0(%rng : tensor<2xui64>, %prior : tensor<f64>,
                                             %A : tensor<6x5xf64>,
                                             %trace : tensor<1x1xf64>)
      -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>) {
    %step_size = arith.constant dense<0.1> : tensor<f64>

    %result:8 = enzyme.mcmc @model_transpose_dim0(%rng, %prior, %A) given %trace
        step_size = %step_size {
      selection = [[#enzyme.symbol<1>]],
      all_addresses = [[#enzyme.symbol<1>]],
      hmc_config = #enzyme.hmc_config<trajectory_length = 1.0>,
      num_samples = 1 : i64,
      thinning = 1 : i64,
      num_warmup = 0 : i64
    } : (tensor<2xui64>, tensor<f64>, tensor<6x5xf64>, tensor<1x1xf64>, tensor<f64>)
        -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>, tensor<1x1xf64>, tensor<1x1xf64>, tensor<f64>, tensor<f64>, tensor<1x1xf64>)

    return %result#0, %result#1, %result#2 : tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>
  }

  func.func @model_transpose_dim0(%rng : tensor<2xui64>, %prior : tensor<f64>,
                                    %A : tensor<6x5xf64>)
      -> (tensor<2xui64>, tensor<f64>) {
    %std = arith.constant dense<1.0> : tensor<f64>

    %x:2 = enzyme.sample @normal(%rng, %prior, %std) {
      logpdf = @normal_logpdf,
      symbol = #enzyme.symbol<1>
    } : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)

    // Build a 2x3 matrix from the sample
    %x_mat = stablehlo.broadcast_in_dim %x#1, dims = [] : (tensor<f64>) -> tensor<2x3xf64>

    // Transpose [1, 0]: [2,3] -> [3,2]
    %transposed = stablehlo.transpose %x_mat, dims = [1, 0] : (tensor<2x3xf64>) -> tensor<3x2xf64>

    // Reshape to [6] for dot
    %reshaped = stablehlo.reshape %transposed : (tensor<3x2xf64>) -> tensor<6xf64>

    // dot_general with contracting dim 0 on LHS
    %y = stablehlo.dot_general %A, %reshaped, contracting_dims = [0] x [0]
        : (tensor<6x5xf64>, tensor<6xf64>) -> tensor<5xf64>

    %slice = "stablehlo.slice"(%y) {start_indices = array<i64: 0>, limit_indices = array<i64: 1>, strides = array<i64: 1>} : (tensor<5xf64>) -> tensor<1xf64>
    %result = stablehlo.reshape %slice : (tensor<1xf64>) -> tensor<f64>

    return %x#0, %result : tensor<2xui64>, tensor<f64>
  }
}
