// RUN: enzymexlamlir-opt %s --inline-infer-regions --sicm | FileCheck %s

module {
  func.func private @model_logpdf(%position : tensor<8xf64>,
                                  %B : tensor<4x8xcomplex<f64>>) -> tensor<f64> {
    %zero = stablehlo.constant dense<0.0> : tensor<8xf64>
    %xc = stablehlo.complex %position, %zero : tensor<8xcomplex<f64>>
    %fft = stablehlo.fft %xc, type = FFT, length = [8] : (tensor<8xcomplex<f64>>) -> tensor<8xcomplex<f64>>
    %dot = stablehlo.dot_general %B, %fft, contracting_dims = [1] x [0] : (tensor<4x8xcomplex<f64>>, tensor<8xcomplex<f64>>) -> tensor<4xcomplex<f64>>
    %re = stablehlo.real %dot : (tensor<4xcomplex<f64>>) -> tensor<4xf64>
    %sliced = stablehlo.slice %re [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %result = stablehlo.reshape %sliced : (tensor<1xf64>) -> tensor<f64>
    return %result : tensor<f64>
  }

  // CHECK-LABEL:  func.func @main(
  // CHECK-SAME:       %[[RNG:.+]]: tensor<2xui64>, %[[B:.+]]: tensor<4x8xcomplex<f64>>)
  // CHECK:        %[[BR0:.+]] = stablehlo.reshape %[[B]] : (tensor<4x8xcomplex<f64>>) -> tensor<4x8xcomplex<f64>>
  // CHECK-NEXT:   %[[BF:.+]] = stablehlo.fft %[[BR0]], type = FFT, length = [8] : (tensor<4x8xcomplex<f64>>) -> tensor<4x8xcomplex<f64>>
  // CHECK-NEXT:   %[[BR1:.+]] = stablehlo.reshape %[[BF]] : (tensor<4x8xcomplex<f64>>) -> tensor<4x8xcomplex<f64>>
  // CHECK:        impulse.infer_region(%[[RNG]], %[[B]])
  // CHECK-SAME:   sampler {
  // CHECK-NEXT:   } logpdf {
  // CHECK-NEXT:   ^bb0(%[[POS:.+]]: tensor<8xf64>):
  // CHECK-NOT:    stablehlo.fft
  // CHECK:        %[[XC:.+]] = stablehlo.complex %[[POS]], %{{.+}} : tensor<8xcomplex<f64>>
  // CHECK-NEXT:   %[[DOT:.+]] = stablehlo.dot_general %[[BR1]], %[[XC]], contracting_dims = [1] x [0] : (tensor<4x8xcomplex<f64>>, tensor<8xcomplex<f64>>) -> tensor<4xcomplex<f64>>
  // CHECK-NEXT:   %[[RE:.+]] = stablehlo.real %[[DOT]] : (tensor<4xcomplex<f64>>) -> tensor<4xf64>
  // CHECK-NEXT:   %[[SL:.+]] = stablehlo.slice %[[RE]] [0:1] : (tensor<4xf64>) -> tensor<1xf64>
  // CHECK-NEXT:   %[[RS:.+]] = stablehlo.reshape %[[SL]] : (tensor<1xf64>) -> tensor<f64>
  // CHECK-NEXT:   impulse.yield %[[RS]] : tensor<f64>
  func.func @main(%rng : tensor<2xui64>,
                  %B : tensor<4x8xcomplex<f64>>)
      -> (tensor<1x8xf64>, tensor<1x2xi1>, tensor<1xf64>, tensor<2xui64>) {
    %init_pos = stablehlo.constant dense<0.0> : tensor<1x8xf64>
    %step_size = stablehlo.constant dense<0.1> : tensor<f64>
    %imm = stablehlo.constant dense<1.0> : tensor<8xf64>
    %res:4 = "impulse.infer"(%rng, %B, %imm, %step_size, %init_pos) {
      logpdf_fn = @model_logpdf,
      nuts_config = #impulse.nuts_config<max_tree_depth = 3, max_delta_energy = 1000.0, adapt_step_size = false, adapt_mass_matrix = false>,
      selection = [],
      all_addresses = [],
      num_warmup = 0,
      num_samples = 1,
      operand_segment_sizes = array<i32: 2, 0, 0, 0, 1, 1, 1, 0>
    } : (tensor<2xui64>, tensor<4x8xcomplex<f64>>, tensor<8xf64>, tensor<f64>, tensor<1x8xf64>)
        -> (tensor<1x8xf64>, tensor<1x2xi1>, tensor<1xf64>, tensor<2xui64>)
    return %res#0, %res#1, %res#2, %res#3 : tensor<1x8xf64>, tensor<1x2xi1>, tensor<1xf64>, tensor<2xui64>
  }
}
