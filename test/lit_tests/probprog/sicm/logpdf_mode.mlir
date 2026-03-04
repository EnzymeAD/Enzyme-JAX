// RUN: enzymexlamlir-opt --inline-mcmc-regions --sicm %s | FileCheck %s

// ============================================================================
// Test: SICM in logpdf mode (custom logpdf_fn, no enzyme.sample)
// ============================================================================
//
// The logpdf function computes: logpdf(pos) = log(broadcast(x) * A)[0]
// where x = pos[0,0] is the scalar position and A:[3] is invariant.
//
// LogMultiplyDistributionHLO should split:
//   log(broadcast(x) * A) -> log(x) + log(A)
// Then log(A) is fully invariant and gets hoisted.

// CHECK-LABEL: func.func @test_logpdf_sicm
// CHECK-SAME: %[[RNG:.*]]: tensor<2xui64>, %[[A:.*]]: tensor<3xf64>, %[[POS0:.*]]: tensor<1x1xf64>

// log(A) should be hoisted before mcmc_region
// CHECK: stablehlo.log %[[A]] : tensor<3xf64>
// CHECK: enzyme.mcmc_region

// Inside: log(x) + hoisted_log_A, no log of a tensor<3xf64>
// CHECK-NOT: stablehlo.log %{{.*}} : tensor<3xf64>
// CHECK: stablehlo.log %{{.*}} : tensor<f64>
// CHECK: stablehlo.broadcast_in_dim
// CHECK-NEXT: stablehlo.add
// CHECK: enzyme.yield

module {
  func.func private @logpdf(%pos : tensor<1x1xf64>, %A : tensor<3xf64>)
      -> tensor<f64> {
    // Extract scalar from position
    %x = stablehlo.reshape %pos : (tensor<1x1xf64>) -> tensor<f64>
    // broadcast(x) * A
    %x_vec = stablehlo.broadcast_in_dim %x, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %scaled = stablehlo.multiply %x_vec, %A : tensor<3xf64>
    // log(broadcast(x) * A)
    %log_result = stablehlo.log %scaled : tensor<3xf64>
    // Extract scalar
    %slice = "stablehlo.slice"(%log_result) {start_indices = array<i64: 0>, limit_indices = array<i64: 1>, strides = array<i64: 1>} : (tensor<3xf64>) -> tensor<1xf64>
    %r = stablehlo.reshape %slice : (tensor<1xf64>) -> tensor<f64>
    return %r : tensor<f64>
  }

  func.func @test_logpdf_sicm(%rng : tensor<2xui64>, %A : tensor<3xf64>,
                                %pos0 : tensor<1x1xf64>)
      -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>) {
    %step_size = arith.constant dense<0.1> : tensor<f64>

    %result:3 = enzyme.mcmc (%rng, %A)
        step_size = %step_size logpdf_fn = @logpdf initial_position = %pos0 {
      selection = [[]],
      all_addresses = [[]],
      hmc_config = #enzyme.hmc_config<trajectory_length = 1.0>,
      num_samples = 1 : i64,
      thinning = 1 : i64,
      num_warmup = 0 : i64
    } : (tensor<2xui64>, tensor<3xf64>, tensor<f64>, tensor<1x1xf64>)
        -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>)

    return %result#0, %result#1, %result#2 : tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>
  }
}
