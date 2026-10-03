// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=impulse.sicm_dot_absorb_fft" --transform-interpreter --enzyme-hlo-remove-transform | FileCheck %s

module {
  func.func private @normal_sampler(%rng : tensor<2xui64>, %std : tensor<4xf64>) -> (tensor<2xui64>, tensor<4xf64>) {
    %sample = stablehlo.abs %std : tensor<4xf64>
    return %rng, %sample : tensor<2xui64>, tensor<4xf64>
  }
  func.func private @normal_logpdf(%x : tensor<4xf64>, %std : tensor<4xf64>) -> tensor<4xf64> {
    %neg = stablehlo.negate %x : tensor<4xf64>
    return %neg : tensor<4xf64>
  }

  func.func private @normal_sampler3(%rng : tensor<2xui64>, %std : tensor<3xf64>) -> (tensor<2xui64>, tensor<3xf64>) {
    %sample = stablehlo.abs %std : tensor<3xf64>
    return %rng, %sample : tensor<2xui64>, tensor<3xf64>
  }
  func.func private @normal_logpdf3(%x : tensor<3xf64>, %std : tensor<3xf64>) -> tensor<3xf64> {
    %neg = stablehlo.negate %x : tensor<3xf64>
    return %neg : tensor<3xf64>
  }
  func.func private @normal_sampler4x2(%rng : tensor<2xui64>, %std : tensor<4x2xf64>) -> (tensor<2xui64>, tensor<4x2xf64>) {
    %sample = stablehlo.abs %std : tensor<4x2xf64>
    return %rng, %sample : tensor<2xui64>, tensor<4x2xf64>
  }
  func.func private @normal_logpdf4x2(%x : tensor<4x2xf64>, %std : tensor<4x2xf64>) -> tensor<4x2xf64> {
    %neg = stablehlo.negate %x : tensor<4x2xf64>
    return %neg : tensor<4x2xf64>
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

  // Must not fire: IRFFT passes the extent check but is not complex-to-complex.
  // CHECK-LABEL: func.func @neg_irfft
  // CHECK: ^bb0(%{{.+}}: tensor<2xui64>, %[[A:.+]]: tensor<2x4xf64>):
  // CHECK: %[[XC:.+]] = stablehlo.complex %{{.+}}, %{{.+}} : tensor<3xcomplex<f64>>
  // CHECK-NEXT: %[[F:.+]] = stablehlo.fft %[[XC]], type = IRFFT, length = [4] : (tensor<3xcomplex<f64>>) -> tensor<4xf64>
  // CHECK-NEXT: %[[DOT:.+]] = stablehlo.dot_general %[[A]], %[[F]], contracting_dims = [1] x [0] : (tensor<2x4xf64>, tensor<4xf64>) -> tensor<2xf64>
  // CHECK-NEXT: impulse.yield %{{.+}}, %[[DOT]] : tensor<2xui64>, tensor<2xf64>
  func.func @neg_irfft(%rng : tensor<2xui64>, %A : tensor<2x4xf64>, %trace : tensor<1x1xf64>) -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>) {
    %step_size = stablehlo.constant dense<0.1> : tensor<f64>
    %std = stablehlo.constant dense<1.0> : tensor<3xf64>
    %res:8 = impulse.infer_region @model(%rng, %A) given %trace init_step_size = %step_size sampler {
    ^bb0(%arg_rng: tensor<2xui64>, %arg_A: tensor<2x4xf64>):
      %x:2 = impulse.sample_region(%arg_rng, %std) sampler {
      ^bb0(%r: tensor<2xui64>, %s: tensor<3xf64>):
        %v = stablehlo.abs %s : tensor<3xf64>
        impulse.yield %r, %v : tensor<2xui64>, tensor<3xf64>
      } logpdf {
      ^bb0(%px: tensor<3xf64>, %s: tensor<3xf64>):
        %lp = stablehlo.negate %px : tensor<3xf64>
        impulse.yield %lp : tensor<3xf64>
      } {address = [#impulse.symbol<1>], fn = @normal_sampler3, logpdf_fn = @normal_logpdf3, support = #impulse.support<REAL>} : (tensor<2xui64>, tensor<3xf64>) -> (tensor<2xui64>, tensor<3xf64>)
      %xc = stablehlo.complex %x#1, %x#1 : tensor<3xcomplex<f64>>
      %f = stablehlo.fft %xc, type = IRFFT, length = [4] : (tensor<3xcomplex<f64>>) -> tensor<4xf64>
      %y = stablehlo.dot_general %arg_A, %f, contracting_dims = [1] x [0] : (tensor<2x4xf64>, tensor<4xf64>) -> tensor<2xf64>
      impulse.yield %x#0, %y : tensor<2xui64>, tensor<2xf64>
    } logpdf {
    } {all_addresses = [[#impulse.symbol<1>]], hmc_config = #impulse.hmc_config<trajectory_length = 1.0>, selection = [[#impulse.symbol<1>]]} : (tensor<2xui64>, tensor<2x4xf64>, tensor<1x1xf64>, tensor<f64>) -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>, tensor<1x1xf64>, tensor<1x1xf64>, tensor<f64>, tensor<f64>, tensor<1x1xf64>)
    return %res#0, %res#1, %res#2 : tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>
  }

  // Must not fire: RFFT passes the extent check but takes real input.
  // CHECK-LABEL: func.func @neg_rfft
  // CHECK: ^bb0(%{{.+}}: tensor<2xui64>, %[[A:.+]]: tensor<2x8xcomplex<f64>>):
  // CHECK: %[[X:.+]]:2 = impulse.sample_region
  // CHECK: %[[F:.+]] = stablehlo.fft %[[X]]#1, type = RFFT, length = [4, 2] : (tensor<4x2xf64>) -> tensor<4x2xcomplex<f64>>
  // CHECK-NEXT: %[[FR:.+]] = stablehlo.reshape %[[F]] : (tensor<4x2xcomplex<f64>>) -> tensor<8xcomplex<f64>>
  // CHECK-NEXT: %[[DOT:.+]] = stablehlo.dot_general %[[A]], %[[FR]], contracting_dims = [1] x [0] : (tensor<2x8xcomplex<f64>>, tensor<8xcomplex<f64>>) -> tensor<2xcomplex<f64>>
  // CHECK-NEXT: impulse.yield %[[X]]#0, %[[DOT]] : tensor<2xui64>, tensor<2xcomplex<f64>>
  func.func @neg_rfft(%rng : tensor<2xui64>, %A : tensor<2x8xcomplex<f64>>, %trace : tensor<1x1xf64>) -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>) {
    %step_size = stablehlo.constant dense<0.1> : tensor<f64>
    %std = stablehlo.constant dense<1.0> : tensor<4x2xf64>
    %res:8 = impulse.infer_region @model(%rng, %A) given %trace init_step_size = %step_size sampler {
    ^bb0(%arg_rng: tensor<2xui64>, %arg_A: tensor<2x8xcomplex<f64>>):
      %x:2 = impulse.sample_region(%arg_rng, %std) sampler {
      ^bb0(%r: tensor<2xui64>, %s: tensor<4x2xf64>):
        %v = stablehlo.abs %s : tensor<4x2xf64>
        impulse.yield %r, %v : tensor<2xui64>, tensor<4x2xf64>
      } logpdf {
      ^bb0(%px: tensor<4x2xf64>, %s: tensor<4x2xf64>):
        %lp = stablehlo.negate %px : tensor<4x2xf64>
        impulse.yield %lp : tensor<4x2xf64>
      } {address = [#impulse.symbol<1>], fn = @normal_sampler4x2, logpdf_fn = @normal_logpdf4x2, support = #impulse.support<REAL>} : (tensor<2xui64>, tensor<4x2xf64>) -> (tensor<2xui64>, tensor<4x2xf64>)
      %f = stablehlo.fft %x#1, type = RFFT, length = [4, 2] : (tensor<4x2xf64>) -> tensor<4x2xcomplex<f64>>
      %fr = stablehlo.reshape %f : (tensor<4x2xcomplex<f64>>) -> tensor<8xcomplex<f64>>
      %y = stablehlo.dot_general %arg_A, %fr, contracting_dims = [1] x [0] : (tensor<2x8xcomplex<f64>>, tensor<8xcomplex<f64>>) -> tensor<2xcomplex<f64>>
      impulse.yield %x#0, %y : tensor<2xui64>, tensor<2xcomplex<f64>>
    } logpdf {
    } {all_addresses = [[#impulse.symbol<1>]], hmc_config = #impulse.hmc_config<trajectory_length = 1.0>, selection = [[#impulse.symbol<1>]]} : (tensor<2xui64>, tensor<2x8xcomplex<f64>>, tensor<1x1xf64>, tensor<f64>) -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>, tensor<1x1xf64>, tensor<1x1xf64>, tensor<f64>, tensor<f64>, tensor<1x1xf64>)
    return %res#0, %res#1, %res#2 : tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>
  }
}
