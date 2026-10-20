// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=impulse.sicm_divide_scale" --transform-interpreter --enzyme-hlo-remove-transform | FileCheck %s

module {
  func.func private @normal_sampler(%rng : tensor<2xui64>, %std : tensor<f64>) -> (tensor<2xui64>, tensor<f64>) {
    %sample = stablehlo.abs %std : tensor<f64>
    return %rng, %sample : tensor<2xui64>, tensor<f64>
  }
  func.func private @normal_logpdf(%x : tensor<f64>, %std : tensor<f64>) -> tensor<f64> {
    %neg = stablehlo.negate %x : tensor<f64>
    return %neg : tensor<f64>
  }

  // CHECK-LABEL: func.func @scaled_numerator
  func.func @scaled_numerator(%rng : tensor<2xui64>, %A : tensor<3x3xf64>, %B : tensor<3x3xf64>, %trace : tensor<1x1xf64>) -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>) {
    %step_size = stablehlo.constant dense<0.1> : tensor<f64>
    %std = stablehlo.constant dense<1.0> : tensor<f64>
    // CHECK: ^bb0(%{{.+}}: tensor<2xui64>, %[[A:.+]]: tensor<3x3xf64>, %[[B:.+]]: tensor<3x3xf64>):
    %res:8 = impulse.infer_region @model(%rng, %A, %B) given %trace init_step_size = %step_size sampler {
    ^bb0(%arg_rng: tensor<2xui64>, %arg_A: tensor<3x3xf64>, %arg_B: tensor<3x3xf64>):
      %tau:2 = impulse.sample_region(%arg_rng, %std) sampler {
      ^bb0(%r: tensor<2xui64>, %s: tensor<f64>):
        %v = stablehlo.abs %s : tensor<f64>
        impulse.yield %r, %v : tensor<2xui64>, tensor<f64>
      } logpdf {
      ^bb0(%x: tensor<f64>, %s: tensor<f64>):
        %lp = stablehlo.negate %x : tensor<f64>
        impulse.yield %lp : tensor<f64>
      } <{address = [#impulse.symbol<1>], fn = @normal_sampler, logpdf_fn = @normal_logpdf, support = #impulse.support<REAL>}> : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
      // CHECK: %[[TAU:.+]]:2 = impulse.sample_region
      %bcast = stablehlo.broadcast_in_dim %tau#1, dims = [] : (tensor<f64>) -> tensor<3x3xf64>
      %K = stablehlo.multiply %bcast, %arg_A : tensor<3x3xf64>
      // CHECK: %[[DIV:.+]] = stablehlo.divide %[[A]], %[[B]] : tensor<3x3xf64>
      // CHECK-NEXT: %[[SB:.+]] = stablehlo.broadcast_in_dim %[[TAU]]#1, dims = [] : (tensor<f64>) -> tensor<3x3xf64>
      // CHECK-NEXT: %[[OUT:.+]] = stablehlo.multiply %[[SB]], %[[DIV]] : tensor<3x3xf64>
      %D = stablehlo.divide %K, %arg_B : tensor<3x3xf64>
      // CHECK: impulse.yield %[[TAU]]#0, %[[OUT]] : tensor<2xui64>, tensor<3x3xf64>
      impulse.yield %tau#0, %D : tensor<2xui64>, tensor<3x3xf64>
    } logpdf {
    } <{all_addresses = [[#impulse.symbol<1>]], hmc_config = #impulse.hmc_config<trajectory_length = 1.0>, selection = [[#impulse.symbol<1>]]}> : (tensor<2xui64>, tensor<3x3xf64>, tensor<3x3xf64>, tensor<1x1xf64>, tensor<f64>) -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>, tensor<1x1xf64>, tensor<1x1xf64>, tensor<f64>, tensor<f64>, tensor<1x1xf64>)
    return %res#0, %res#1, %res#2 : tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>
  }

  // CHECK-LABEL: func.func @scaled_denominator
  func.func @scaled_denominator(%rng : tensor<2xui64>, %A : tensor<3x3xf64>, %trace : tensor<1x1xf64>) -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>) {
    %step_size = stablehlo.constant dense<0.1> : tensor<f64>
    %std = stablehlo.constant dense<1.0> : tensor<f64>
    // CHECK: %[[ONE:.+]] = stablehlo.constant dense<1.000000e+00> : tensor<f64>
    // CHECK: ^bb0(%{{.+}}: tensor<2xui64>, %[[DA:.+]]: tensor<3x3xf64>):
    %res:8 = impulse.infer_region @model(%rng, %A) given %trace init_step_size = %step_size sampler {
    ^bb0(%arg_rng: tensor<2xui64>, %arg_A: tensor<3x3xf64>):
      %tau:2 = impulse.sample_region(%arg_rng, %std) sampler {
      ^bb0(%r: tensor<2xui64>, %s: tensor<f64>):
        %v = stablehlo.abs %s : tensor<f64>
        impulse.yield %r, %v : tensor<2xui64>, tensor<f64>
      } logpdf {
      ^bb0(%x: tensor<f64>, %s: tensor<f64>):
        %lp = stablehlo.negate %x : tensor<f64>
        impulse.yield %lp : tensor<f64>
      } <{address = [#impulse.symbol<1>], fn = @normal_sampler, logpdf_fn = @normal_logpdf, support = #impulse.support<REAL>}> : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
      // CHECK: %[[DTAU:.+]]:2 = impulse.sample_region
      // CHECK: %[[X:.+]] = stablehlo.broadcast_in_dim %[[DTAU]]#1, dims = [] : (tensor<f64>) -> tensor<3x3xf64>
      %x = stablehlo.broadcast_in_dim %tau#1, dims = [] : (tensor<f64>) -> tensor<3x3xf64>
      %bcast = stablehlo.broadcast_in_dim %tau#1, dims = [] : (tensor<f64>) -> tensor<3x3xf64>
      %K = stablehlo.multiply %bcast, %arg_A : tensor<3x3xf64>
      // CHECK: %[[DDIV:.+]] = stablehlo.divide %[[X]], %[[DA]] : tensor<3x3xf64>
      // CHECK-NEXT: %[[RECIP:.+]] = stablehlo.divide %[[ONE]], %[[DTAU]]#1 : tensor<f64>
      // CHECK-NEXT: %[[RECIPB:.+]] = stablehlo.broadcast_in_dim %[[RECIP]], dims = [] : (tensor<f64>) -> tensor<3x3xf64>
      // CHECK-NEXT: %[[DOUT:.+]] = stablehlo.multiply %[[RECIPB]], %[[DDIV]] : tensor<3x3xf64>
      %D = stablehlo.divide %x, %K : tensor<3x3xf64>
      // CHECK: impulse.yield %[[DTAU]]#0, %[[DOUT]] : tensor<2xui64>, tensor<3x3xf64>
      impulse.yield %tau#0, %D : tensor<2xui64>, tensor<3x3xf64>
    } logpdf {
    } <{all_addresses = [[#impulse.symbol<1>]], hmc_config = #impulse.hmc_config<trajectory_length = 1.0>, selection = [[#impulse.symbol<1>]]}> : (tensor<2xui64>, tensor<3x3xf64>, tensor<1x1xf64>, tensor<f64>) -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>, tensor<1x1xf64>, tensor<1x1xf64>, tensor<f64>, tensor<f64>, tensor<1x1xf64>)
    return %res#0, %res#1, %res#2 : tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>
  }
}
