// RUN: enzymexlamlir-opt --inline-mcmc-regions --sicm %s | FileCheck %s

// ============================================================================
// Test: DotAbsorbDiagMul — absorb per-column scale into LHS matrix
// ============================================================================
//
// Model: dot_general(A, multiply(X, broadcast(c))) where A, c are invariant
//        and X is a non-scalar sample-dependent tensor (add of broadcast + iota).
// SICM rewrites: dot_general(multiply(A, broadcast(c)), X)
// Then hoists multiply(A, broadcast(c)) outside mcmc_region.

// CHECK-LABEL: func.func @test_dot_absorb_diag_mul
// CHECK-SAME: %[[RNG:.*]]: tensor<2xui64>, %[[PRIOR:.*]]: tensor<f64>, %[[A:.*]]: tensor<3x2xf64>, %[[C:.*]]: tensor<2xf64>, %[[IOTA:.*]]: tensor<2x1xf64>, %[[TRACE:.*]]: tensor<1x1xf64>
// The pre-scaled matrix should be hoisted before mcmc_region
// CHECK: stablehlo.broadcast_in_dim %[[C]]
// CHECK-NEXT: stablehlo.multiply %[[A]]
// CHECK: enzyme.mcmc_region
// Inside: sample, build X, dot_general with pre-scaled A — no multiply on RHS
// CHECK: enzyme.sample_region
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

  func.func @test_dot_absorb_diag_mul(%rng : tensor<2xui64>, %prior : tensor<f64>,
                                       %A : tensor<3x2xf64>, %c : tensor<2xf64>,
                                       %iota_data : tensor<2x1xf64>,
                                       %trace : tensor<1x1xf64>)
      -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>) {
    %step_size = arith.constant dense<0.1> : tensor<f64>

    %result:8 = enzyme.mcmc @model_diag_mul(%rng, %prior, %A, %c, %iota_data) given %trace
        step_size = %step_size {
      selection = [[#enzyme.symbol<1>]],
      all_addresses = [[#enzyme.symbol<1>]],
      hmc_config = #enzyme.hmc_config<trajectory_length = 1.0>,
      num_samples = 1 : i64,
      thinning = 1 : i64,
      num_warmup = 0 : i64
    } : (tensor<2xui64>, tensor<f64>, tensor<3x2xf64>, tensor<2xf64>, tensor<2x1xf64>, tensor<1x1xf64>, tensor<f64>)
        -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>, tensor<1x1xf64>, tensor<1x1xf64>, tensor<f64>, tensor<f64>, tensor<1x1xf64>)

    return %result#0, %result#1, %result#2 : tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>
  }

  func.func @model_diag_mul(%rng : tensor<2xui64>, %prior : tensor<f64>,
                             %A : tensor<3x2xf64>, %c : tensor<2xf64>,
                             %iota_data : tensor<2x1xf64>)
      -> (tensor<2xui64>, tensor<f64>) {
    %std = arith.constant dense<1.0> : tensor<f64>

    %x:2 = enzyme.sample @normal(%rng, %prior, %std) {
      logpdf = @normal_logpdf,
      symbol = #enzyme.symbol<1>
    } : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)

    // X = broadcast(x) + iota_data — non-scalar sample-dependent tensor
    %x_bcast = stablehlo.broadcast_in_dim %x#1, dims = [] : (tensor<f64>) -> tensor<2x1xf64>
    %X = stablehlo.add %x_bcast, %iota_data : tensor<2x1xf64>

    // Multiply by invariant per-column scale c
    %c_bcast = stablehlo.broadcast_in_dim %c, dims = [0] : (tensor<2xf64>) -> tensor<2x1xf64>
    %scaled_x = stablehlo.multiply %X, %c_bcast : tensor<2x1xf64>

    // dot_general(A, scaled_x)
    %y = stablehlo.dot_general %A, %scaled_x, contracting_dims = [1] x [0]
        : (tensor<3x2xf64>, tensor<2x1xf64>) -> tensor<3x1xf64>

    %slice = "stablehlo.slice"(%y) {start_indices = array<i64: 0, 0>, limit_indices = array<i64: 1, 1>, strides = array<i64: 1, 1>} : (tensor<3x1xf64>) -> tensor<1x1xf64>
    %result = stablehlo.reshape %slice : (tensor<1x1xf64>) -> tensor<f64>

    return %x#0, %result : tensor<2xui64>, tensor<f64>
  }
}
