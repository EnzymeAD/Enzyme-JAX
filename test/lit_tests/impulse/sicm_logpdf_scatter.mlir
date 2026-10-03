// RUN: enzymexlamlir-opt %s --inline-infer-regions --sicm | FileCheck %s

// The last-occurrence mask depends only on the indices and is hoisted.

module {
  func.func private @model_logpdf(%position : tensor<8xf64>,
                                  %B : tensor<16x3xcomplex<f64>>,
                                  %idx : tensor<8x2xi64>) -> tensor<f64> {
    %p = stablehlo.reshape %position : (tensor<8xf64>) -> tensor<1x8xf64>
    %neg = stablehlo.negate %p : tensor<1x8xf64>
    %upd = stablehlo.concatenate %p, %neg, dim = 0 : (tensor<1x8xf64>, tensor<1x8xf64>) -> tensor<2x8xf64>
    %grid0 = stablehlo.constant dense<0.0> : tensor<2x4x4xf64>
    %grid = "stablehlo.scatter"(%grid0, %idx, %upd) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [1, 2], scatter_dims_to_operand_dims = [1, 2], index_vector_dim = 1>}> ({
    ^bb0(%old: tensor<f64>, %new: tensor<f64>):
      stablehlo.return %new : tensor<f64>
    }) : (tensor<2x4x4xf64>, tensor<8x2xi64>, tensor<2x8xf64>) -> tensor<2x4x4xf64>
    %gt = stablehlo.transpose %grid, dims = [0, 2, 1] : (tensor<2x4x4xf64>) -> tensor<2x4x4xf64>
    %re3 = stablehlo.slice %gt [0:1, 0:4, 0:4] : (tensor<2x4x4xf64>) -> tensor<1x4x4xf64>
    %re = stablehlo.reshape %re3 : (tensor<1x4x4xf64>) -> tensor<4x4xf64>
    %im3 = stablehlo.slice %gt [1:2, 0:4, 0:4] : (tensor<2x4x4xf64>) -> tensor<1x4x4xf64>
    %im = stablehlo.reshape %im3 : (tensor<1x4x4xf64>) -> tensor<4x4xf64>
    %img = stablehlo.complex %re, %im : tensor<4x4xcomplex<f64>>
    %f = stablehlo.fft %img, type = FFT, length = [4, 4] : (tensor<4x4xcomplex<f64>>) -> tensor<4x4xcomplex<f64>>
    %flat = stablehlo.reshape %f : (tensor<4x4xcomplex<f64>>) -> tensor<16xcomplex<f64>>
    %vis = stablehlo.dot_general %B, %flat, contracting_dims = [0] x [0] : (tensor<16x3xcomplex<f64>>, tensor<16xcomplex<f64>>) -> tensor<3xcomplex<f64>>
    %r = stablehlo.real %vis : (tensor<3xcomplex<f64>>) -> tensor<3xf64>
    %sliced = stablehlo.slice %r [0:1] : (tensor<3xf64>) -> tensor<1xf64>
    %result = stablehlo.reshape %sliced : (tensor<1xf64>) -> tensor<f64>
    return %result : tensor<f64>
  }

  // CHECK-LABEL: func.func @main(
  // CHECK-SAME: %[[RNG:.+]]: tensor<2xui64>, %[[B:.+]]: tensor<16x3xcomplex<f64>>, %[[IDX:.+]]: tensor<8x2xi64>)
  // CHECK-DAG: %[[NONE:.+]] = stablehlo.constant dense<-1> : tensor<16xi64>
  // CHECK: %[[BF:.+]] = stablehlo.fft %{{.+}}, type = FFT, length = [4, 4] : (tensor<3x4x4xcomplex<f64>>) -> tensor<3x4x4xcomplex<f64>>
  // CHECK: %[[BFT:.+]] = stablehlo.reshape %{{.+}} : (tensor<4x4x3xcomplex<f64>>) -> tensor<16x3xcomplex<f64>>
  // CHECK-NEXT: %[[I0:.+]] = stablehlo.slice %[[IDX]] [0:8, 0:1] : (tensor<8x2xi64>) -> tensor<8x1xi64>
  // CHECK-NEXT: %[[S0:.+]] = stablehlo.multiply %[[I0]], %{{.+}} : tensor<8x1xi64>
  // CHECK-NEXT: %[[L0:.+]] = stablehlo.add %{{.+}}, %[[S0]] : tensor<8x1xi64>
  // CHECK-NEXT: %[[I1:.+]] = stablehlo.slice %[[IDX]] [0:8, 1:2] : (tensor<8x2xi64>) -> tensor<8x1xi64>
  // CHECK-NEXT: %[[S1:.+]] = stablehlo.multiply %[[I1]], %{{.+}} : tensor<8x1xi64>
  // CHECK-NEXT: %[[LIN:.+]] = stablehlo.add %[[L0]], %[[S1]] : tensor<8x1xi64>
  // CHECK-NEXT: %[[G:.+]] = "stablehlo.gather"(%[[BFT]], %[[LIN]]) <{{.+}}> : (tensor<16x3xcomplex<f64>>, tensor<8x1xi64>) -> tensor<8x3xcomplex<f64>>
  // CHECK-NEXT: %[[POS:.+]] = stablehlo.iota dim = 0 : tensor<8xi64>
  // CHECK-NEXT: %[[WIN:.+]] = "stablehlo.scatter"(%[[NONE]], %[[LIN]], %[[POS]])
  // CHECK: stablehlo.maximum
  // CHECK: %[[WINAT:.+]] = "stablehlo.gather"(%[[WIN]], %[[LIN]])
  // CHECK-NEXT: %[[LAST:.+]] = stablehlo.compare EQ, %[[WINAT]], %[[POS]] : (tensor<8xi64>, tensor<8xi64>) -> tensor<8xi1>
  // CHECK-NEXT: %[[KEEP:.+]] = stablehlo.convert %[[LAST]] : (tensor<8xi1>) -> tensor<8xcomplex<f64>>
  // CHECK-NEXT: %[[KEEPB:.+]] = stablehlo.broadcast_in_dim %[[KEEP]], dims = [0] : (tensor<8xcomplex<f64>>) -> tensor<8x3xcomplex<f64>>
  // CHECK-NEXT: %[[OP:.+]] = stablehlo.multiply %[[G]], %[[KEEPB]] : tensor<8x3xcomplex<f64>>
  // CHECK-NEXT: impulse.infer_region(%[[RNG]], %[[B]], %[[IDX]])
  // CHECK-SAME: sampler {
  // CHECK-NEXT: } logpdf {
  // CHECK-NEXT: ^bb0(%[[POSN:.+]]: tensor<8xf64>):
  // CHECK-NEXT: %[[P:.+]] = stablehlo.reshape %[[POSN]] : (tensor<8xf64>) -> tensor<1x8xf64>
  // CHECK-NEXT: %[[NEG:.+]] = stablehlo.negate %[[P]] : tensor<1x8xf64>
  // CHECK-NEXT: %[[UPD:.+]] = stablehlo.concatenate %[[P]], %[[NEG]], dim = 0 : (tensor<1x8xf64>, tensor<1x8xf64>) -> tensor<2x8xf64>
  // CHECK-NEXT: %[[RE2:.+]] = stablehlo.slice %[[UPD]] [0:1, 0:8] : (tensor<2x8xf64>) -> tensor<1x8xf64>
  // CHECK-NEXT: %[[IM2:.+]] = stablehlo.slice %[[UPD]] [1:2, 0:8] : (tensor<2x8xf64>) -> tensor<1x8xf64>
  // CHECK-NEXT: %[[RE:.+]] = stablehlo.reshape %[[RE2]] : (tensor<1x8xf64>) -> tensor<8xf64>
  // CHECK-NEXT: %[[IM:.+]] = stablehlo.reshape %[[IM2]] : (tensor<1x8xf64>) -> tensor<8xf64>
  // CHECK-NEXT: %[[XC:.+]] = stablehlo.complex %[[RE]], %[[IM]] : tensor<8xcomplex<f64>>
  // CHECK-NEXT: %[[XR:.+]] = stablehlo.reshape %[[XC]] : (tensor<8xcomplex<f64>>) -> tensor<8xcomplex<f64>>
  // CHECK-NEXT: %[[VIS:.+]] = stablehlo.dot_general %[[OP]], %[[XR]], contracting_dims = [0] x [0] : (tensor<8x3xcomplex<f64>>, tensor<8xcomplex<f64>>) -> tensor<3xcomplex<f64>>
  // CHECK-NEXT: %[[VR:.+]] = stablehlo.real %[[VIS]] : (tensor<3xcomplex<f64>>) -> tensor<3xf64>
  // CHECK-NEXT: %[[S:.+]] = stablehlo.slice %[[VR]] [0:1] : (tensor<3xf64>) -> tensor<1xf64>
  // CHECK-NEXT: %[[LP:.+]] = stablehlo.reshape %[[S]] : (tensor<1xf64>) -> tensor<f64>
  // CHECK-NEXT: impulse.yield %[[LP]] : tensor<f64>
  func.func @main(%rng : tensor<2xui64>,
                  %B : tensor<16x3xcomplex<f64>>,
                  %idx : tensor<8x2xi64>)
      -> (tensor<1x8xf64>, tensor<1x2xi1>, tensor<1xf64>, tensor<2xui64>) {
    %init_pos = stablehlo.constant dense<0.0> : tensor<1x8xf64>
    %step_size = stablehlo.constant dense<0.1> : tensor<f64>
    %imm = stablehlo.constant dense<1.0> : tensor<8xf64>
    %res:4 = "impulse.infer"(%rng, %B, %idx, %imm, %step_size, %init_pos) {
      logpdf_fn = @model_logpdf,
      nuts_config = #impulse.nuts_config<max_tree_depth = 3, max_delta_energy = 1000.0, adapt_step_size = false, adapt_mass_matrix = false>,
      selection = [],
      all_addresses = [],
      num_warmup = 0,
      num_samples = 1,
      operand_segment_sizes = array<i32: 3, 0, 0, 0, 1, 1, 1, 0>
    } : (tensor<2xui64>, tensor<16x3xcomplex<f64>>, tensor<8x2xi64>, tensor<8xf64>, tensor<f64>, tensor<1x8xf64>)
        -> (tensor<1x8xf64>, tensor<1x2xi1>, tensor<1xf64>, tensor<2xui64>)
    return %res#0, %res#1, %res#2, %res#3 : tensor<1x8xf64>, tensor<1x2xi1>, tensor<1xf64>, tensor<2xui64>
  }
}
