// RUN: enzymexlamlir-opt --inline-mcmc-regions --sicm %s | FileCheck %s

// ============================================================================
// Test: ExpAddFactorization — exp(broadcast(s) + A) -> broadcast(exp(s)) * exp(A)
// ============================================================================

// CHECK-LABEL: func.func @test_exp_add
// CHECK-SAME: %[[RNG:.*]]: tensor<2xui64>, %[[PRIOR:.*]]: tensor<f64>, %[[A:.*]]: tensor<3xf64>, %[[TRACE:.*]]: tensor<1x1xf64>
// exp(A) should be hoisted before mcmc_region
// CHECK: stablehlo.exponential %[[A]] : tensor<3xf64>
// CHECK: enzyme.mcmc_region
// Inside: exp(scale) broadcast and multiply with hoisted exp(A)
// CHECK: enzyme.sample_region
// CHECK: stablehlo.exponential %{{.*}} : tensor<f64>
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

  func.func @test_exp_add(%rng : tensor<2xui64>, %prior : tensor<f64>,
                           %A : tensor<3xf64>, %trace : tensor<1x1xf64>)
      -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>) {
    %step_size = arith.constant dense<0.1> : tensor<f64>

    %result:8 = enzyme.mcmc @model_exp_add(%rng, %prior, %A) given %trace
        step_size = %step_size {
      selection = [[#enzyme.symbol<1>]],
      all_addresses = [[#enzyme.symbol<1>]],
      hmc_config = #enzyme.hmc_config<trajectory_length = 1.0>,
      num_samples = 1 : i64,
      thinning = 1 : i64,
      num_warmup = 0 : i64
    } : (tensor<2xui64>, tensor<f64>, tensor<3xf64>, tensor<1x1xf64>, tensor<f64>)
        -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>, tensor<1x1xf64>, tensor<1x1xf64>, tensor<f64>, tensor<f64>, tensor<1x1xf64>)

    return %result#0, %result#1, %result#2 : tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>
  }

  func.func @model_exp_add(%rng : tensor<2xui64>, %prior : tensor<f64>,
                            %A : tensor<3xf64>)
      -> (tensor<2xui64>, tensor<f64>) {
    %std = arith.constant dense<1.0> : tensor<f64>

    %scale:2 = enzyme.sample @normal(%rng, %prior, %std) {
      logpdf = @normal_logpdf,
      symbol = #enzyme.symbol<1>
    } : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)

    // broadcast(scale) + A (additive, not multiplicative)
    %scale_vec = stablehlo.broadcast_in_dim %scale#1, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %shifted = stablehlo.add %scale_vec, %A : tensor<3xf64>
    %exp_result = stablehlo.exponential %shifted : tensor<3xf64>

    %slice = "stablehlo.slice"(%exp_result) {start_indices = array<i64: 0>, limit_indices = array<i64: 1>, strides = array<i64: 1>} : (tensor<3xf64>) -> tensor<1xf64>
    %result = stablehlo.reshape %slice : (tensor<1xf64>) -> tensor<f64>

    return %scale#0, %result : tensor<2xui64>, tensor<f64>
  }
}
