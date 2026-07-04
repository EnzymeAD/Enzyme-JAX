// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=impulse.sicm_dot_absorb_fft" --transform-interpreter --enzyme-hlo-remove-transform | FileCheck %s

// x sample-dependent, single last contracting dim of extent prod(fft_length).
module {
  func.func private @normal_sampler(%rng : tensor<2xui64>, %std : tensor<4xf64>) -> (tensor<2xui64>, tensor<4xf64>) {
    %sample = stablehlo.abs %std : tensor<4xf64>
    return %rng, %sample : tensor<2xui64>, tensor<4xf64>
  }
  func.func private @normal_logpdf(%x : tensor<4xf64>, %std : tensor<4xf64>) -> tensor<4xf64> {
    %neg = stablehlo.negate %x : tensor<4xf64>
    return %neg : tensor<4xf64>
  }

  // CHECK-LABEL: func.func @absorb_fft
  func.func @absorb_fft(%rng : tensor<2xui64>, %A : tensor<2x4xcomplex<f64>>, %trace : tensor<1x1xf64>) -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>) {
    %step_size = stablehlo.constant dense<0.1> : tensor<f64>
    %std = stablehlo.constant dense<1.0> : tensor<4xf64>
    // CHECK: ^bb0(%{{.+}}: tensor<2xui64>, %[[A:.+]]: tensor<2x4xcomplex<f64>>):
    %res:8 = impulse.infer_region @model(%rng, %A) given %trace init_step_size = %step_size sampler {
    ^bb0(%arg_rng: tensor<2xui64>, %arg_A: tensor<2x4xcomplex<f64>>):
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
      %xc = stablehlo.complex %x#1, %x#1 : tensor<4xcomplex<f64>>
      // CHECK: %[[XC:.+]] = stablehlo.complex %[[X]]#1, %[[X]]#1 : tensor<4xcomplex<f64>>
      // CHECK-NEXT: %[[ELHS:.+]] = stablehlo.reshape %[[A]] : (tensor<2x4xcomplex<f64>>) -> tensor<2x4xcomplex<f64>>
      // CHECK-NEXT: %[[FFTA:.+]] = stablehlo.fft %[[ELHS]], type = FFT, length = [4] : (tensor<2x4xcomplex<f64>>) -> tensor<2x4xcomplex<f64>>
      // CHECK-NEXT: %[[NLHS:.+]] = stablehlo.reshape %[[FFTA]] : (tensor<2x4xcomplex<f64>>) -> tensor<2x4xcomplex<f64>>
      // CHECK-NEXT: %[[DOT:.+]] = stablehlo.dot_general %[[NLHS]], %[[XC]], contracting_dims = [1] x [0] : (tensor<2x4xcomplex<f64>>, tensor<4xcomplex<f64>>) -> tensor<2xcomplex<f64>>
      %f = stablehlo.fft %xc, type = FFT, length = [4] : (tensor<4xcomplex<f64>>) -> tensor<4xcomplex<f64>>
      %y = stablehlo.dot_general %arg_A, %f, contracting_dims = [1] x [0] : (tensor<2x4xcomplex<f64>>, tensor<4xcomplex<f64>>) -> tensor<2xcomplex<f64>>
      // CHECK: impulse.yield %[[X]]#0, %[[DOT]] : tensor<2xui64>, tensor<2xcomplex<f64>>
      impulse.yield %x#0, %y : tensor<2xui64>, tensor<2xcomplex<f64>>
    } logpdf {
    } {all_addresses = [[#impulse.symbol<1>]], hmc_config = #impulse.hmc_config<trajectory_length = 1.0>, selection = [[#impulse.symbol<1>]]} : (tensor<2xui64>, tensor<2x4xcomplex<f64>>, tensor<1x1xf64>, tensor<f64>) -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>, tensor<1x1xf64>, tensor<1x1xf64>, tensor<f64>, tensor<f64>, tensor<1x1xf64>)
    return %res#0, %res#1, %res#2 : tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>
  }
}
