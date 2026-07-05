// RUN: enzymexlamlir-opt %s --inline-infer-regions --enzyme-hlo-generate-td="patterns=impulse.sicm_sqrt_scale" --transform-interpreter --enzyme-hlo-remove-transform | FileCheck %s

// The POSITIVE supports are derived from the dist attributes, not declared.

module {
  func.func private @normal_sampler(%r: tensor<2xui64>, %mu: tensor<f64>, %sig: tensor<f64>) -> (tensor<2xui64>, tensor<f64>) {
    return %r, %mu : tensor<2xui64>, tensor<f64>
  }
  func.func @model(%rng : tensor<2xui64>, %R : tensor<3x3xf64>) -> (tensor<2xui64>, tensor<3x3xf64>) {
    %mu = stablehlo.constant dense<0.0> : tensor<f64>
    %sig = stablehlo.constant dense<1.0> : tensor<f64>
    %tau1:2 = impulse.sample @normal_sampler(%rng, %mu, %sig) {
      symbol = #impulse.symbol<1>, dist = #impulse.dist<STD_NORMAL, [#impulse.xform<AFFINE>, #impulse.xform<EXP>]>
    } : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    %tau2:2 = impulse.sample @normal_sampler(%tau1#0, %mu, %sig) {
      symbol = #impulse.symbol<2>, dist = #impulse.dist<STD_NORMAL, [#impulse.xform<AFFINE>, #impulse.xform<EXP>]>
    } : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    %s = stablehlo.add %tau1#1, %tau2#1 : tensor<f64>
    %bcast = stablehlo.broadcast_in_dim %s, dims = [] : (tensor<f64>) -> tensor<3x3xf64>
    %K = stablehlo.multiply %bcast, %R : tensor<3x3xf64>
    %root = stablehlo.sqrt %K : tensor<3x3xf64>
    return %tau2#0, %root : tensor<2xui64>, tensor<3x3xf64>
  }

  // CHECK-LABEL: func.func @test
  // CHECK: impulse.sample_region
  // CHECK: dist = #impulse.dist<STD_NORMAL, [#impulse.xform<AFFINE>, #impulse.xform<EXP>]>
  // CHECK-SAME: support = #impulse.support<POSITIVE>
  // CHECK: impulse.sample_region
  // CHECK: dist = #impulse.dist<STD_NORMAL, [#impulse.xform<AFFINE>, #impulse.xform<EXP>]>
  // CHECK-SAME: support = #impulse.support<POSITIVE>
  // CHECK: %[[S:.+]] = stablehlo.add %{{.+}}#1, %{{.+}}#1 {enzymexla.non_negative = {{.*}}} : tensor<f64>
  // CHECK: %[[SQRTR:.+]] = stablehlo.sqrt %{{.+}} : tensor<3x3xf64>
  // CHECK-NEXT: %[[SQRTS:.+]] = stablehlo.sqrt %[[S]] : tensor<f64>
  // CHECK-NEXT: %[[SQRTSB:.+]] = stablehlo.broadcast_in_dim %[[SQRTS]], dims = [] : (tensor<f64>) -> tensor<3x3xf64>
  // CHECK-NEXT: %[[OUT:.+]] = stablehlo.multiply %[[SQRTSB]], %[[SQRTR]] : tensor<3x3xf64>
  func.func @test(%rng : tensor<2xui64>, %R : tensor<3x3xf64>, %trace : tensor<1x2xf64>) -> (tensor<1x2xf64>, tensor<1x2xi1>, tensor<1xf64>, tensor<2xui64>) {
    %step_size = stablehlo.constant dense<0.1> : tensor<f64>
    %res:4 = impulse.infer @model(%rng, %R) given %trace init_step_size = %step_size {
      selection = [[#impulse.symbol<1>], [#impulse.symbol<2>]],
      all_addresses = [[#impulse.symbol<1>], [#impulse.symbol<2>]],
      hmc_config = #impulse.hmc_config<trajectory_length = 1.0>,
      num_samples = 1 : i64, thinning = 1 : i64, num_warmup = 0 : i64
    } : (tensor<2xui64>, tensor<3x3xf64>, tensor<1x2xf64>, tensor<f64>) -> (tensor<1x2xf64>, tensor<1x2xi1>, tensor<1xf64>, tensor<2xui64>)
    return %res#0, %res#1, %res#2, %res#3 : tensor<1x2xf64>, tensor<1x2xi1>, tensor<1xf64>, tensor<2xui64>
  }
}
