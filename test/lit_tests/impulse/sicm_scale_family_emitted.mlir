// RUN: enzymexlamlir-opt %s --inline-infer-regions --sicm | FileCheck %s

// sicm_scale_family.mlir with tau drawn through an emitted distribution.

module {
  func.func private @normal_sampler(%r: tensor<2xui64>, %mu: tensor<f64>, %sig: tensor<f64>) -> (tensor<2xui64>, tensor<f64>) {
    return %r, %mu : tensor<2xui64>, tensor<f64>
  }
  func.func private @mvn_sampler(%rng : tensor<2xui64>, %cov : tensor<3x3xf64>) -> (tensor<2xui64>, tensor<3xf64>) {
    %zero = stablehlo.constant dense<0.0> : tensor<3xf64>
    return %rng, %zero : tensor<2xui64>, tensor<3xf64>
  }
  func.func private @mvn_logpdf(%x : tensor<3xf64>, %cov : tensor<3x3xf64>) -> tensor<f64> {
    %L = stablehlo.cholesky %cov, lower = true : tensor<3x3xf64>
    %log_L = stablehlo.log %L : tensor<3x3xf64>
    %sliced = stablehlo.slice %log_L [0:1, 0:1] : (tensor<3x3xf64>) -> tensor<1x1xf64>
    %result = stablehlo.reshape %sliced : (tensor<1x1xf64>) -> tensor<f64>
    return %result : tensor<f64>
  }
  func.func @model(%rng : tensor<2xui64>, %R_data : tensor<3x3xf64>) -> (tensor<2xui64>, tensor<3xf64>) {
    %mu = stablehlo.constant dense<0.0> : tensor<f64>
    %sig = stablehlo.constant dense<1.0> : tensor<f64>
    %tau:2 = impulse.sample @normal_sampler(%rng, %mu, %sig) <{
      symbol = #impulse.symbol<1>,
      dist = #impulse.dist<STD_NORMAL, [#impulse.xform<AFFINE>]>
    }> : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    %tau_sq = stablehlo.multiply %tau#1, %tau#1 : tensor<f64>
    %tau_sq_mat = stablehlo.broadcast_in_dim %tau_sq, dims = [] : (tensor<f64>) -> tensor<3x3xf64>
    %K = stablehlo.multiply %tau_sq_mat, %R_data : tensor<3x3xf64>
    %y:2 = impulse.sample @mvn_sampler(%tau#0, %K) <{
      logpdf = @mvn_logpdf, symbol = #impulse.symbol<2>
    }> : (tensor<2xui64>, tensor<3x3xf64>) -> (tensor<2xui64>, tensor<3xf64>)
    return %y#0, %y#1 : tensor<2xui64>, tensor<3xf64>
  }

  // CHECK-LABEL: func.func @test
  // CHECK: %[[CHOL:.+]] = stablehlo.cholesky %arg1, lower = true : tensor<3x3xf64>
  // CHECK-NEXT: %[[LOGCHOL:.+]] = stablehlo.log %[[CHOL]] : tensor<3x3xf64>
  // CHECK: impulse.infer_region @model
  // CHECK: impulse.sample_region
  // CHECK: impulse.random %{{.+}} <rng_distribution = NORMAL>
  // CHECK: support = <REAL>
  // CHECK-SAME: dist = <STD_NORMAL, [#impulse.xform<AFFINE>]>
  // CHECK: %[[SCALE:.+]] = stablehlo.sqrt %{{.+}} {enzymexla.non_negative = {{.*}}} : tensor<f64>
  // CHECK-NEXT: %[[LOGSCALE:.+]] = stablehlo.log %[[SCALE]] : tensor<f64>
  // CHECK-NEXT: %[[LOGSCALEB:.+]] = stablehlo.broadcast_in_dim %[[LOGSCALE]], dims = [] : (tensor<f64>) -> tensor<3x3xf64>
  // CHECK-NEXT: %[[LOGL:.+]] = stablehlo.add %[[LOGSCALEB]], %[[LOGCHOL]] : tensor<3x3xf64>
  func.func @test(%rng : tensor<2xui64>, %R_data : tensor<3x3xf64>, %trace : tensor<1x4xf64>) -> (tensor<1x4xf64>, tensor<1x2xi1>, tensor<1xf64>, tensor<2xui64>) {
    %step_size = stablehlo.constant dense<0.1> : tensor<f64>
    %result:4 = impulse.infer @model(%rng, %R_data) given %trace init_step_size = %step_size <{
      selection = [[#impulse.symbol<1>]],
      all_addresses = [[#impulse.symbol<1>], [#impulse.symbol<2>]],
      hmc_config = #impulse.hmc_config<trajectory_length = 1.0>,
      num_samples = 1 : i64, thinning = 1 : i64, num_warmup = 0 : i64
    }> : (tensor<2xui64>, tensor<3x3xf64>, tensor<1x4xf64>, tensor<f64>) -> (tensor<1x4xf64>, tensor<1x2xi1>, tensor<1xf64>, tensor<2xui64>)
    return %result#0, %result#1, %result#2, %result#3 : tensor<1x4xf64>, tensor<1x2xi1>, tensor<1xf64>, tensor<2xui64>
  }
}
