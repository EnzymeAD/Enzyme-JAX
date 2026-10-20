// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=impulse.sicm_dot_general_scale" --transform-interpreter --enzyme-hlo-remove-transform | FileCheck %s

module {
  func.func private @halfnormal_sampler(%rng : tensor<2xui64>, %std : tensor<f64>) -> (tensor<2xui64>, tensor<f64>) {
    %sample = stablehlo.abs %std : tensor<f64>
    return %rng, %sample : tensor<2xui64>, tensor<f64>
  }
  func.func private @halfnormal_logpdf(%x : tensor<f64>, %std : tensor<f64>) -> tensor<f64> {
    %neg = stablehlo.negate %x : tensor<f64>
    return %neg : tensor<f64>
  }

  // CHECK-LABEL: func.func @scaled_lhs_dot
  func.func @scaled_lhs_dot(%rng : tensor<2xui64>, %A : tensor<3x3xf64>, %x : tensor<3xf64>, %trace : tensor<1x1xf64>) -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>) {
    %step_size = stablehlo.constant dense<0.1> : tensor<f64>
    %std = stablehlo.constant dense<1.0> : tensor<f64>
    // CHECK: ^bb0(%{{.+}}: tensor<2xui64>, %[[A:.+]]: tensor<3x3xf64>, %[[X:.+]]: tensor<3xf64>):
    %res:8 = impulse.infer_region @model(%rng, %A, %x) given %trace init_step_size = %step_size sampler {
    ^bb0(%arg_rng: tensor<2xui64>, %arg_A: tensor<3x3xf64>, %arg_x: tensor<3xf64>):
      %tau:2 = impulse.sample_region(%arg_rng, %std) sampler {
      ^bb0(%r: tensor<2xui64>, %s: tensor<f64>):
        %v = stablehlo.abs %s : tensor<f64>
        impulse.yield %r, %v : tensor<2xui64>, tensor<f64>
      } logpdf {
      ^bb0(%lx: tensor<f64>, %s: tensor<f64>):
        %lp = stablehlo.negate %lx : tensor<f64>
        impulse.yield %lp : tensor<f64>
      } <{address = [#impulse.symbol<1>], fn = @halfnormal_sampler, logpdf_fn = @halfnormal_logpdf, support = #impulse.support<POSITIVE>}> : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
      // CHECK: %[[TAU:.+]]:2 = impulse.sample_region
      // CHECK: %[[TAUSQ:.+]] = stablehlo.multiply %[[TAU]]#1, %[[TAU]]#1 : tensor<f64>
      %tausq = stablehlo.multiply %tau#1, %tau#1 : tensor<f64>
      %bcast = stablehlo.broadcast_in_dim %tausq, dims = [] : (tensor<f64>) -> tensor<3x3xf64>
      %scaledA = stablehlo.multiply %bcast, %arg_A : tensor<3x3xf64>
      // CHECK: %[[DOT:.+]] = stablehlo.dot_general %[[A]], %[[X]], contracting_dims = [1] x [0] : (tensor<3x3xf64>, tensor<3xf64>) -> tensor<3xf64>
      // CHECK-NEXT: %[[SB:.+]] = stablehlo.broadcast_in_dim %[[TAUSQ]], dims = [] : (tensor<f64>) -> tensor<3xf64>
      // CHECK-NEXT: %[[RES:.+]] = stablehlo.multiply %[[SB]], %[[DOT]] : tensor<3xf64>
      %y = stablehlo.dot_general %scaledA, %arg_x, contracting_dims = [1] x [0] : (tensor<3x3xf64>, tensor<3xf64>) -> tensor<3xf64>
      // CHECK: impulse.yield %[[TAU]]#0, %[[RES]] : tensor<2xui64>, tensor<3xf64>
      impulse.yield %tau#0, %y : tensor<2xui64>, tensor<3xf64>
    } logpdf {
    } <{all_addresses = [[#impulse.symbol<1>]], hmc_config = #impulse.hmc_config<trajectory_length = 1.0>, selection = [[#impulse.symbol<1>]]}> : (tensor<2xui64>, tensor<3x3xf64>, tensor<3xf64>, tensor<1x1xf64>, tensor<f64>) -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>, tensor<1x1xf64>, tensor<1x1xf64>, tensor<f64>, tensor<f64>, tensor<1x1xf64>)
    return %res#0, %res#1, %res#2 : tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>
  }
}
