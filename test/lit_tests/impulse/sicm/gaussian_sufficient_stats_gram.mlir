// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=impulse.sicm_gaussian_sufficient_stats_gram" --transform-interpreter --enzyme-hlo-remove-transform | FileCheck %s

module {
  func.func private @normal_sampler(%rng : tensor<2xui64>, %std : tensor<f64>) -> (tensor<2xui64>, tensor<f64>) {
    return %rng, %std : tensor<2xui64>, tensor<f64>
  }
  func.func private @normal_logpdf(%x : tensor<f64>, %std : tensor<f64>) -> tensor<f64> {
    %neg = stablehlo.negate %x : tensor<f64>
    return %neg : tensor<f64>
  }

  // CHECK-LABEL: func.func @linreg_gram
  // CHECK: ^bb0(%{{.+}}: tensor<2xui64>, %[[X:.+]]: tensor<20x3xf64>, %[[Y:.+]]: tensor<20xf64>):
  // CHECK: %[[TAU:.+]]:2 = impulse.sample_region
  // CHECK: %[[BETA:.+]] = stablehlo.broadcast_in_dim %[[TAU]]#1, dims = [] : (tensor<f64>) -> tensor<3xf64>
  // CHECK-NEXT: %[[G:.+]] = stablehlo.dot_general %[[X]], %[[X]], contracting_dims = [0] x [0] : (tensor<20x3xf64>, tensor<20x3xf64>) -> tensor<3x3xf64>
  // CHECK-NEXT: %[[XTY:.+]] = stablehlo.dot_general %[[X]], %[[Y]], contracting_dims = [0] x [0] : (tensor<20x3xf64>, tensor<20xf64>) -> tensor<3xf64>
  // CHECK-NEXT: %[[YTY:.+]] = stablehlo.dot_general %[[Y]], %[[Y]], contracting_dims = [0] x [0] : (tensor<20xf64>, tensor<20xf64>) -> tensor<f64>
  // CHECK-NEXT: %[[GB:.+]] = stablehlo.dot_general %[[G]], %[[BETA]], contracting_dims = [1] x [0] : (tensor<3x3xf64>, tensor<3xf64>) -> tensor<3xf64>
  // CHECK-NEXT: %[[BGB:.+]] = stablehlo.dot_general %[[BETA]], %[[GB]], contracting_dims = [0] x [0] : (tensor<3xf64>, tensor<3xf64>) -> tensor<f64>
  // CHECK-NEXT: %[[BXTY:.+]] = stablehlo.dot_general %[[BETA]], %[[XTY]], contracting_dims = [0] x [0] : (tensor<3xf64>, tensor<3xf64>) -> tensor<f64>
  // CHECK-NEXT: %[[TWOBXTY:.+]] = stablehlo.multiply %{{.+}}, %[[BXTY]] : tensor<f64>
  // CHECK-NEXT: %[[DIFF:.+]] = stablehlo.subtract %[[YTY]], %[[TWOBXTY]] : tensor<f64>
  // CHECK-NEXT: %[[CORE:.+]] = stablehlo.add %[[DIFF]], %[[BGB]] : tensor<f64>
  // CHECK-NEXT: %[[SIG2:.+]] = stablehlo.multiply %[[TAU]]#1, %[[TAU]]#1 : tensor<f64>
  // CHECK-NEXT: %[[SUMSQ:.+]] = stablehlo.divide %[[CORE]], %[[SIG2]] : tensor<f64>
  // CHECK-NOT: enzymexla.linalg.qr
  func.func @linreg_gram(%rng : tensor<2xui64>, %X : tensor<20x3xf64>, %y : tensor<20xf64>, %trace : tensor<1x1xf64>) -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>) {
    %step_size = stablehlo.constant dense<0.1> : tensor<f64>
    %std = stablehlo.constant dense<1.0> : tensor<f64>
    %res:8 = impulse.infer_region @model(%rng, %X, %y) given %trace init_step_size = %step_size sampler {
    ^bb0(%arg_rng: tensor<2xui64>, %arg_X: tensor<20x3xf64>, %arg_y: tensor<20xf64>):
      %tau:2 = impulse.sample_region(%arg_rng, %std) sampler {
      ^bb0(%r: tensor<2xui64>, %s: tensor<f64>):
        impulse.yield %r, %s : tensor<2xui64>, tensor<f64>
      } logpdf {
      ^bb0(%lx: tensor<f64>, %s: tensor<f64>):
        %lp = stablehlo.negate %lx : tensor<f64>
        impulse.yield %lp : tensor<f64>
      } {address = [#impulse.symbol<1>], fn = @normal_sampler, logpdf_fn = @normal_logpdf, support = #impulse.support<REAL>} : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
      %beta = stablehlo.broadcast_in_dim %tau#1, dims = [] : (tensor<f64>) -> tensor<3xf64>
      %mu = stablehlo.dot_general %arg_X, %beta, contracting_dims = [1] x [0] : (tensor<20x3xf64>, tensor<3xf64>) -> tensor<20xf64>
      %cen = stablehlo.subtract %arg_y, %mu : tensor<20xf64>
      %sigma_b = stablehlo.broadcast_in_dim %tau#1, dims = [] : (tensor<f64>) -> tensor<20xf64>
      %z = stablehlo.divide %cen, %sigma_b : tensor<20xf64>
      %z2 = stablehlo.multiply %z, %z : tensor<20xf64>
      %log2pi = stablehlo.constant dense<1.8378770664093453> : tensor<20xf64>
      %sm = stablehlo.add %z2, %log2pi : tensor<20xf64>
      %neghalf = stablehlo.constant dense<-5.000000e-01> : tensor<f64>
      %neghalf_b = stablehlo.broadcast_in_dim %neghalf, dims = [] : (tensor<f64>) -> tensor<20xf64>
      %kernel = stablehlo.multiply %sm, %neghalf_b : tensor<20xf64>
      %ones = stablehlo.constant dense<1.000000e+00> : tensor<20xf64>
      %ss = stablehlo.dot_general %kernel, %ones, contracting_dims = [0] x [0] : (tensor<20xf64>, tensor<20xf64>) -> tensor<f64>
      %ss_b = stablehlo.broadcast_in_dim %ss, dims = [] : (tensor<f64>) -> tensor<3xf64>
      impulse.yield %tau#0, %ss_b : tensor<2xui64>, tensor<3xf64>
    } logpdf {
    } {all_addresses = [[#impulse.symbol<1>]], hmc_config = #impulse.hmc_config<trajectory_length = 1.0>, selection = [[#impulse.symbol<1>]]} : (tensor<2xui64>, tensor<20x3xf64>, tensor<20xf64>, tensor<1x1xf64>, tensor<f64>) -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>, tensor<1x1xf64>, tensor<1x1xf64>, tensor<f64>, tensor<f64>, tensor<1x1xf64>)
    return %res#0, %res#1, %res#2 : tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>
  }
}
