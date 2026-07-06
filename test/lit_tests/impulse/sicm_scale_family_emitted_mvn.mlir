// RUN: enzymexlamlir-opt %s --pass-pipeline='builtin.module(inline-infer-regions,canonicalize,cse,lower-impulse-to-stablehlo{linalg-only=true},arith-raise{stablehlo=true},canonicalize,cse,sicm)' | FileCheck %s

// (lower-impulse-to-stablehlo{linalg-only} for the LA ops, arith-raise for the
// the density region, exactly as it does on a traced StableHLO MVN body.
// The fn-absent emitted MVN site is written in generic op form because the
// pretty `impulse.sample(...)` form does not round-trip through the parser.

module {
  func.func private @halfnormal_sampler(%rng : tensor<2xui64>, %std : tensor<f64>) -> (tensor<2xui64>, tensor<f64>) {
    %sample = stablehlo.abs %std : tensor<f64>
    return %rng, %sample : tensor<2xui64>, tensor<f64>
  }
  func.func private @halfnormal_logpdf(%x : tensor<f64>, %std : tensor<f64>) -> tensor<f64> {
    %neg = stablehlo.negate %x : tensor<f64>
    return %neg : tensor<f64>
  }
  func.func @model(%rng : tensor<2xui64>, %R_data : tensor<3x3xf64>) -> (tensor<2xui64>, tensor<3xf64>) {
    %std = stablehlo.constant dense<1.0> : tensor<f64>
    %tau:2 = impulse.sample @halfnormal_sampler(%rng, %std) {
      logpdf = @halfnormal_logpdf, symbol = #impulse.symbol<1>, support = #impulse.support<POSITIVE>
    } : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    %tau_sq = stablehlo.multiply %tau#1, %tau#1 : tensor<f64>
    %tau_sq_mat = stablehlo.broadcast_in_dim %tau_sq, dims = [] : (tensor<f64>) -> tensor<3x3xf64>
    %K = stablehlo.multiply %tau_sq_mat, %R_data : tensor<3x3xf64>
    %mu = stablehlo.constant dense<0.0> : tensor<3xf64>
    %y:2 = "impulse.sample"(%tau#0, %mu, %K) {
      dist = #impulse.dist<MVN>, symbol = #impulse.symbol<2>, support = #impulse.support<REAL>
    } : (tensor<2xui64>, tensor<3xf64>, tensor<3x3xf64>) -> (tensor<2xui64>, tensor<3xf64>)
    return %y#0, %y#1 : tensor<2xui64>, tensor<3xf64>
  }

  // CHECK-LABEL: func.func @test
  // invariant factor cholesky(%arg1) sits above the impulse.infer_region.
  // CHECK: %[[CHOL:.+]] = stablehlo.cholesky %arg1 : tensor<3x3xf64>
  // CHECK: impulse.infer_region @model
  // Inside the joint logpdf, the scale is decomposed: sqrt(tau^2) (proven
  // nonnegative) multiplies the hoisted cholesky(R).
  // CHECK: %[[TAUSQ:.+]] = stablehlo.multiply %[[TAUARG:.+]], %[[TAUARG]] {enzymexla.non_negative = {{.*}}} : tensor<f64>
  // CHECK-NEXT: %[[SQRT:.+]] = stablehlo.sqrt %[[TAUSQ]] : tensor<f64>
  // CHECK-NEXT: %[[SQRTB:.+]] = stablehlo.broadcast_in_dim %[[SQRT]], dims = [] : (tensor<f64>) -> tensor<3x3xf64>
  // CHECK-NEXT: %[[U:.+]] = stablehlo.multiply %[[SQRTB]], %[[CHOL]] : tensor<3x3xf64>
  func.func @test(%rng : tensor<2xui64>, %R_data : tensor<3x3xf64>, %trace : tensor<1x4xf64>) -> (tensor<1x4xf64>, tensor<1xi1>, tensor<2xui64>) {
    %step_size = stablehlo.constant dense<0.1> : tensor<f64>
    %result:3 = impulse.infer @model(%rng, %R_data) given %trace init_step_size = %step_size {
      selection = [[#impulse.symbol<1>]],
      all_addresses = [[#impulse.symbol<1>], [#impulse.symbol<2>]],
      hmc_config = #impulse.hmc_config<trajectory_length = 1.0>,
      num_samples = 1 : i64, thinning = 1 : i64, num_warmup = 0 : i64
    } : (tensor<2xui64>, tensor<3x3xf64>, tensor<1x4xf64>, tensor<f64>) -> (tensor<1x4xf64>, tensor<1xi1>, tensor<2xui64>)
    return %result#0, %result#1, %result#2 : tensor<1x4xf64>, tensor<1xi1>, tensor<2xui64>
  }
}
