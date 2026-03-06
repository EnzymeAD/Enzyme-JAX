// RUN: enzymexlamlir-opt --inline-mcmc-regions --sicm %s | FileCheck %s

// ============================================================================
// Test: DivideScaleFactorization
// Case 1: divide(broadcast(s) * A, B) -> broadcast(s) * divide(A, B)
//   when A, B are sample-invariant
// Case 2: divide(X, broadcast(s) * A) -> broadcast(1/s) * divide(X, A)
//   when A is sample-invariant
// ============================================================================

// CHECK-LABEL: func.func @test_divide_num
// CHECK-SAME: %[[RNG:.*]]: tensor<2xui64>, %[[PRIOR:.*]]: tensor<f64>, %[[A:.*]]: tensor<3xf64>, %[[B:.*]]: tensor<3xf64>, %[[TRACE:.*]]: tensor<1x1xf64>
// divide(A, B) should be hoisted before mcmc_region
// CHECK: stablehlo.divide %[[A]], %[[B]] : tensor<3xf64>
// CHECK-NOT: stablehlo.divide %[[A]], %[[B]]
// CHECK: enzyme.mcmc_region
// Inside: broadcast(scale) * hoisted_divide
// CHECK: enzyme.sample_region
// CHECK: stablehlo.broadcast_in_dim
// CHECK-NEXT: stablehlo.multiply
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

  // Test Case 1: divide(broadcast(s) * A, B) with A, B invariant
  func.func @test_divide_num(%rng : tensor<2xui64>, %prior : tensor<f64>,
                              %A : tensor<3xf64>, %B : tensor<3xf64>,
                              %trace : tensor<1x1xf64>)
      -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>) {
    %step_size = arith.constant dense<0.1> : tensor<f64>

    %result:8 = enzyme.mcmc @model_divide_num(%rng, %prior, %A, %B) given %trace
        step_size = %step_size {
      selection = [[#enzyme.symbol<1>]],
      all_addresses = [[#enzyme.symbol<1>]],
      hmc_config = #enzyme.hmc_config<trajectory_length = 1.0>,
      num_samples = 1 : i64,
      thinning = 1 : i64,
      num_warmup = 0 : i64
    } : (tensor<2xui64>, tensor<f64>, tensor<3xf64>, tensor<3xf64>, tensor<1x1xf64>, tensor<f64>)
        -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>, tensor<1x1xf64>, tensor<1x1xf64>, tensor<f64>, tensor<f64>, tensor<1x1xf64>)

    return %result#0, %result#1, %result#2 : tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>
  }

  func.func @model_divide_num(%rng : tensor<2xui64>, %prior : tensor<f64>,
                               %A : tensor<3xf64>, %B : tensor<3xf64>)
      -> (tensor<2xui64>, tensor<f64>) {
    %std = arith.constant dense<1.0> : tensor<f64>

    %scale:2 = enzyme.sample @normal(%rng, %prior, %std) {
      logpdf = @normal_logpdf,
      symbol = #enzyme.symbol<1>
    } : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)

    %scale_vec = stablehlo.broadcast_in_dim %scale#1, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %scaled_A = stablehlo.multiply %scale_vec, %A : tensor<3xf64>
    %div_result = stablehlo.divide %scaled_A, %B : tensor<3xf64>

    %slice = "stablehlo.slice"(%div_result) {start_indices = array<i64: 0>, limit_indices = array<i64: 1>, strides = array<i64: 1>} : (tensor<3xf64>) -> tensor<1xf64>
    %result = stablehlo.reshape %slice : (tensor<1xf64>) -> tensor<f64>

    return %scale#0, %result : tensor<2xui64>, tensor<f64>
  }
}
