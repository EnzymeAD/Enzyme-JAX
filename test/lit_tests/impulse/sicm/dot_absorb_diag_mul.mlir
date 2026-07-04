// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=impulse.sicm_dot_absorb_diag_mul" --transform-interpreter --enzyme-hlo-remove-transform | FileCheck %s

module {
  func.func private @normal_sampler(%rng : tensor<2xui64>, %std : tensor<4xf64>) -> (tensor<2xui64>, tensor<4xf64>) {
    %sample = stablehlo.abs %std : tensor<4xf64>
    return %rng, %sample : tensor<2xui64>, tensor<4xf64>
  }
  func.func private @normal_logpdf(%x : tensor<4xf64>, %std : tensor<4xf64>) -> tensor<4xf64> {
    %neg = stablehlo.negate %x : tensor<4xf64>
    return %neg : tensor<4xf64>
  }

  // CHECK-LABEL: func.func @absorb_diag_mul
  func.func @absorb_diag_mul(%rng : tensor<2xui64>, %A : tensor<3x4xf64>, %c : tensor<4xf64>, %trace : tensor<1x1xf64>) -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>) {
    %step_size = stablehlo.constant dense<0.1> : tensor<f64>
    %std = stablehlo.constant dense<1.0> : tensor<4xf64>
    // CHECK: ^bb0(%{{.+}}: tensor<2xui64>, %[[A:.+]]: tensor<3x4xf64>, %[[C:.+]]: tensor<4xf64>):
    %res:8 = impulse.infer_region @model(%rng, %A, %c) given %trace step_size = %step_size sampler {
    ^bb0(%arg_rng: tensor<2xui64>, %arg_A: tensor<3x4xf64>, %arg_c: tensor<4xf64>):
      // CHECK: %[[X:.+]]:2 = impulse.sample_region
      %x:2 = impulse.sample_region(%arg_rng, %std) sampler {
      ^bb0(%r: tensor<2xui64>, %s: tensor<4xf64>):
        %v = stablehlo.abs %s : tensor<4xf64>
        impulse.yield %r, %v : tensor<2xui64>, tensor<4xf64>
      } logpdf {
      ^bb0(%px: tensor<4xf64>, %s: tensor<4xf64>):
        %lp = stablehlo.negate %px : tensor<4xf64>
        impulse.yield %lp : tensor<4xf64>
      } {address = [#impulse.symbol<1>], fn = @normal_sampler, logpdf_fn = @normal_logpdf, support = #impulse.support<REAL>} : (tensor<2xui64>, tensor<4xf64>) -> (tensor<2xui64>, tensor<4xf64>)
      // CHECK: %[[CB:.+]] = stablehlo.broadcast_in_dim %[[C]], dims = [1] : (tensor<4xf64>) -> tensor<3x4xf64>
      // CHECK-NEXT: %[[NLHS:.+]] = stablehlo.multiply %[[A]], %[[CB]] : tensor<3x4xf64>
      // CHECK-NEXT: %[[DOT:.+]] = stablehlo.dot_general %[[NLHS]], %[[X]]#1, contracting_dims = [1] x [0] : (tensor<3x4xf64>, tensor<4xf64>) -> tensor<3xf64>
      %cb = stablehlo.broadcast_in_dim %arg_c, dims = [0] : (tensor<4xf64>) -> tensor<4xf64>
      %sx = stablehlo.multiply %cb, %x#1 : tensor<4xf64>
      %y = stablehlo.dot_general %arg_A, %sx, contracting_dims = [1] x [0] : (tensor<3x4xf64>, tensor<4xf64>) -> tensor<3xf64>
      // CHECK: impulse.yield %[[X]]#0, %[[DOT]] : tensor<2xui64>, tensor<3xf64>
      impulse.yield %x#0, %y : tensor<2xui64>, tensor<3xf64>
    } logpdf {
    } {all_addresses = [[#impulse.symbol<1>]], hmc_config = #impulse.hmc_config<trajectory_length = 1.0>, selection = [[#impulse.symbol<1>]]} : (tensor<2xui64>, tensor<3x4xf64>, tensor<4xf64>, tensor<1x1xf64>, tensor<f64>) -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>, tensor<1x1xf64>, tensor<1x1xf64>, tensor<f64>, tensor<f64>, tensor<1x1xf64>)
    return %res#0, %res#1, %res#2 : tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>
  }
}
