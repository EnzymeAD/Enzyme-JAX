// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=impulse.sicm_cholesky_outer_product_scale" --transform-interpreter --enzyme-hlo-remove-transform | FileCheck %s

module {
  func.func private @halfnormal_sampler(%rng : tensor<2xui64>, %std : tensor<3xf64>) -> (tensor<2xui64>, tensor<3xf64>) {
    %sample = stablehlo.abs %std : tensor<3xf64>
    return %rng, %sample : tensor<2xui64>, tensor<3xf64>
  }
  func.func private @halfnormal_logpdf(%x : tensor<3xf64>, %std : tensor<3xf64>) -> tensor<f64> {
    %c = stablehlo.constant dense<0.0> : tensor<f64>
    return %c : tensor<f64>
  }

  // CHECK-LABEL: func.func @positive_support_sigma
  func.func @positive_support_sigma(%rng : tensor<2xui64>, %Omega : tensor<3x3xf64>, %trace : tensor<1x1xf64>) -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>) {
    %step_size = stablehlo.constant dense<0.1> : tensor<f64>
    %std = stablehlo.constant dense<1.0> : tensor<3xf64>
    // CHECK: ^bb0(%{{.+}}: tensor<2xui64>, %[[OMEGA:.+]]: tensor<3x3xf64>):
    %res:8 = impulse.infer_region @model(%rng, %Omega) given %trace init_step_size = %step_size sampler {
    ^bb0(%arg_rng: tensor<2xui64>, %arg_Omega: tensor<3x3xf64>):
      %sigma:2 = impulse.sample_region(%arg_rng, %std) sampler {
      ^bb0(%r: tensor<2xui64>, %s: tensor<3xf64>):
        %v = stablehlo.abs %s : tensor<3xf64>
        impulse.yield %r, %v : tensor<2xui64>, tensor<3xf64>
      } logpdf {
      ^bb0(%x: tensor<3xf64>, %s: tensor<3xf64>):
        %lp = stablehlo.constant dense<0.0> : tensor<f64>
        impulse.yield %lp : tensor<f64>
      } {address = [#impulse.symbol<1>], fn = @halfnormal_sampler, logpdf_fn = @halfnormal_logpdf, support = #impulse.support<POSITIVE>} : (tensor<2xui64>, tensor<3xf64>) -> (tensor<2xui64>, tensor<3xf64>)
      // CHECK: %[[SIGMA:.+]]:2 = impulse.sample_region
      // CHECK: %[[ROW:.+]] = stablehlo.broadcast_in_dim %[[SIGMA]]#1, dims = [0] : (tensor<3xf64>) -> tensor<3x3xf64>
      %row = stablehlo.broadcast_in_dim %sigma#1, dims = [0] : (tensor<3xf64>) -> tensor<3x3xf64>
      %col = stablehlo.broadcast_in_dim %sigma#1, dims = [1] : (tensor<3xf64>) -> tensor<3x3xf64>
      %outer = stablehlo.multiply %row, %col : tensor<3x3xf64>
      %K = stablehlo.multiply %outer, %arg_Omega : tensor<3x3xf64>
      // CHECK: %[[CHOL:.+]] = stablehlo.cholesky %[[OMEGA]], lower = true : tensor<3x3xf64>
      // CHECK-NEXT: %[[L:.+]] = stablehlo.multiply %[[ROW]], %[[CHOL]] : tensor<3x3xf64>
      %L = stablehlo.cholesky %K, lower = true : tensor<3x3xf64>
      // CHECK: impulse.yield %[[SIGMA]]#0, %[[L]] : tensor<2xui64>, tensor<3x3xf64>
      impulse.yield %sigma#0, %L : tensor<2xui64>, tensor<3x3xf64>
    } logpdf {
    } {all_addresses = [[#impulse.symbol<1>]], hmc_config = #impulse.hmc_config<trajectory_length = 1.0>, selection = [[#impulse.symbol<1>]]} : (tensor<2xui64>, tensor<3x3xf64>, tensor<1x1xf64>, tensor<f64>) -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>, tensor<1x1xf64>, tensor<1x1xf64>, tensor<f64>, tensor<f64>, tensor<1x1xf64>)
    return %res#0, %res#1, %res#2 : tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>
  }

  // CHECK-LABEL: func.func @negative_unproven_sign
  func.func @negative_unproven_sign(%rng : tensor<2xui64>, %Omega : tensor<3x3xf64>, %trace : tensor<1x1xf64>) -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>) {
    %step_size = stablehlo.constant dense<0.1> : tensor<f64>
    %std = stablehlo.constant dense<1.0> : tensor<3xf64>
    // CHECK: ^bb0(%{{.+}}: tensor<2xui64>, %[[NOMEGA:.+]]: tensor<3x3xf64>):
    %res:8 = impulse.infer_region @model(%rng, %Omega) given %trace init_step_size = %step_size sampler {
    ^bb0(%arg_rng: tensor<2xui64>, %arg_Omega: tensor<3x3xf64>):
      %sigma:2 = impulse.sample_region(%arg_rng, %std) sampler {
      ^bb0(%r: tensor<2xui64>, %s: tensor<3xf64>):
        %v = stablehlo.abs %s : tensor<3xf64>
        impulse.yield %r, %v : tensor<2xui64>, tensor<3xf64>
      } logpdf {
      ^bb0(%x: tensor<3xf64>, %s: tensor<3xf64>):
        %lp = stablehlo.constant dense<0.0> : tensor<f64>
        impulse.yield %lp : tensor<f64>
      } {address = [#impulse.symbol<1>], fn = @halfnormal_sampler, logpdf_fn = @halfnormal_logpdf, support = #impulse.support<REAL>} : (tensor<2xui64>, tensor<3xf64>) -> (tensor<2xui64>, tensor<3xf64>)
      // CHECK: %[[NSIGMA:.+]]:2 = impulse.sample_region
      // CHECK: %[[NROW:.+]] = stablehlo.broadcast_in_dim %[[NSIGMA]]#1, dims = [0] : (tensor<3xf64>) -> tensor<3x3xf64>
      // CHECK-NEXT: %[[NCOL:.+]] = stablehlo.broadcast_in_dim %[[NSIGMA]]#1, dims = [1] : (tensor<3xf64>) -> tensor<3x3xf64>
      // CHECK-NEXT: %[[NOUTER:.+]] = stablehlo.multiply %[[NROW]], %[[NCOL]] : tensor<3x3xf64>
      // CHECK-NEXT: %[[NK:.+]] = stablehlo.multiply %[[NOUTER]], %[[NOMEGA]] : tensor<3x3xf64>
      // CHECK-NEXT: %[[NL:.+]] = stablehlo.cholesky %[[NK]], lower = true : tensor<3x3xf64>
      %row = stablehlo.broadcast_in_dim %sigma#1, dims = [0] : (tensor<3xf64>) -> tensor<3x3xf64>
      %col = stablehlo.broadcast_in_dim %sigma#1, dims = [1] : (tensor<3xf64>) -> tensor<3x3xf64>
      %outer = stablehlo.multiply %row, %col : tensor<3x3xf64>
      %K = stablehlo.multiply %outer, %arg_Omega : tensor<3x3xf64>
      %L = stablehlo.cholesky %K, lower = true : tensor<3x3xf64>
      impulse.yield %sigma#0, %L : tensor<2xui64>, tensor<3x3xf64>
    } logpdf {
    } {all_addresses = [[#impulse.symbol<1>]], hmc_config = #impulse.hmc_config<trajectory_length = 1.0>, selection = [[#impulse.symbol<1>]]} : (tensor<2xui64>, tensor<3x3xf64>, tensor<1x1xf64>, tensor<f64>) -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>, tensor<1x1xf64>, tensor<1x1xf64>, tensor<f64>, tensor<f64>, tensor<1x1xf64>)
    return %res#0, %res#1, %res#2 : tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>
  }
}
