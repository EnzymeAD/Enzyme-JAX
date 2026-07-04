// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=impulse.sicm_exp_add" --transform-interpreter --enzyme-hlo-remove-transform | FileCheck %s

module {
  func.func private @normal_sampler(%rng : tensor<2xui64>, %std : tensor<f64>) -> (tensor<2xui64>, tensor<f64>) {
    %sample = stablehlo.abs %std : tensor<f64>
    return %rng, %sample : tensor<2xui64>, tensor<f64>
  }
  func.func private @normal_logpdf(%x : tensor<f64>, %std : tensor<f64>) -> tensor<f64> {
    %neg = stablehlo.negate %x : tensor<f64>
    return %neg : tensor<f64>
  }

  // CHECK-LABEL: func.func @exp_add_shift
  func.func @exp_add_shift(%rng : tensor<2xui64>, %R : tensor<3x3xf64>, %trace : tensor<1x1xf64>) -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>) {
    %step_size = stablehlo.constant dense<0.1> : tensor<f64>
    %std = stablehlo.constant dense<1.0> : tensor<f64>
    // CHECK: ^bb0(%{{.+}}: tensor<2xui64>, %[[R:.+]]: tensor<3x3xf64>):
    %res:8 = impulse.infer_region @model(%rng, %R) given %trace step_size = %step_size sampler {
    ^bb0(%arg_rng: tensor<2xui64>, %arg_R: tensor<3x3xf64>):
      %tau:2 = impulse.sample_region(%arg_rng, %std) sampler {
      ^bb0(%r: tensor<2xui64>, %s: tensor<f64>):
        %v = stablehlo.abs %s : tensor<f64>
        impulse.yield %r, %v : tensor<2xui64>, tensor<f64>
      } logpdf {
      ^bb0(%x: tensor<f64>, %s: tensor<f64>):
        %lp = stablehlo.negate %x : tensor<f64>
        impulse.yield %lp : tensor<f64>
      } {address = [#impulse.symbol<1>], fn = @normal_sampler, logpdf_fn = @normal_logpdf, support = #impulse.support<REAL>} : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
      // CHECK: %[[TAU:.+]]:2 = impulse.sample_region
      %bcast = stablehlo.broadcast_in_dim %tau#1, dims = [] : (tensor<f64>) -> tensor<3x3xf64>
      %K = stablehlo.add %bcast, %arg_R : tensor<3x3xf64>
      // CHECK: %[[EXPA:.+]] = stablehlo.exponential %[[R]] : tensor<3x3xf64>
      // CHECK-NEXT: %[[EXPS:.+]] = stablehlo.exponential %[[TAU]]#1 : tensor<f64>
      // CHECK-NEXT: %[[EXPSB:.+]] = stablehlo.broadcast_in_dim %[[EXPS]], dims = [] : (tensor<f64>) -> tensor<3x3xf64>
      // CHECK-NEXT: %[[OUT:.+]] = stablehlo.multiply %[[EXPSB]], %[[EXPA]] : tensor<3x3xf64>
      %E = stablehlo.exponential %K : tensor<3x3xf64>
      // CHECK: impulse.yield %[[TAU]]#0, %[[OUT]] : tensor<2xui64>, tensor<3x3xf64>
      impulse.yield %tau#0, %E : tensor<2xui64>, tensor<3x3xf64>
    } logpdf {
    } {all_addresses = [[#impulse.symbol<1>]], hmc_config = #impulse.hmc_config<trajectory_length = 1.0>, selection = [[#impulse.symbol<1>]]} : (tensor<2xui64>, tensor<3x3xf64>, tensor<1x1xf64>, tensor<f64>) -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>, tensor<1x1xf64>, tensor<1x1xf64>, tensor<f64>, tensor<f64>, tensor<1x1xf64>)
    return %res#0, %res#1, %res#2 : tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>
  }
}
