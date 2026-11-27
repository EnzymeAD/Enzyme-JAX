// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s
// RUN: enzymexlamlir-opt --enzyme-hlo-opt="all_finite=true" %s | FileCheck %s --check-prefix=FINITE

// CHECK-LABEL: func.func @topk_k1_values_and_indices(
// CHECK-SAME:    %[[X:.*]]: tensor<3x5xf32>) -> (tensor<3x1xf32>, tensor<3x1xi32>) {
// CHECK:         %[[IOTA:.*]] = stablehlo.constant dense<{{\[\[}}0, 1, 2, 3, 4], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4{{\]\]}}> : tensor<3x5xi32>
// CHECK:         %[[RED:.*]]:2 = stablehlo.reduce(%[[X]] init: %{{.*}}), (%[[IOTA]] init: %{{.*}}) across dimensions = [1] : (tensor<3x5xf32>, tensor<3x5xi32>, tensor<f32>, tensor<i32>) -> (tensor<3xf32>, tensor<3xi32>)
// CHECK:         %[[RV:.*]] = stablehlo.reshape %[[RED]]#0 : (tensor<3xf32>) -> tensor<3x1xf32>
// CHECK:         %[[RI:.*]] = stablehlo.reshape %[[RED]]#1 : (tensor<3xi32>) -> tensor<3x1xi32>
// CHECK:         return %[[RV]], %[[RI]] : tensor<3x1xf32>, tensor<3x1xi32>
func.func @topk_k1_values_and_indices(%x: tensor<3x5xf32>) -> (tensor<3x1xf32>, tensor<3x1xi32>) {
  %v, %i = chlo.top_k(%x, k = 1) : tensor<3x5xf32> -> (tensor<3x1xf32>, tensor<3x1xi32>)
  return %v, %i : tensor<3x1xf32>, tensor<3x1xi32>
}

// CHECK-LABEL: func.func @topk_k1_values_only_int(
// CHECK-SAME:    %[[X:.*]]: tensor<4x6xi32>) -> tensor<4x1xi32> {
// CHECK-NOT:     chlo.top_k
// CHECK-NOT:     stablehlo.iota
// CHECK:         %[[RED:.*]] = stablehlo.reduce(%[[X]] init: %{{.*}}) applies stablehlo.maximum across dimensions = [1] : (tensor<4x6xi32>, tensor<i32>) -> tensor<4xi32>
// CHECK:         %[[RV:.*]] = stablehlo.reshape %[[RED]] : (tensor<4xi32>) -> tensor<4x1xi32>
// CHECK:         return %[[RV]] : tensor<4x1xi32>
func.func @topk_k1_values_only_int(%x: tensor<4x6xi32>) -> tensor<4x1xi32> {
  %v, %i = chlo.top_k(%x, k = 1) : tensor<4x6xi32> -> (tensor<4x1xi32>, tensor<4x1xi32>)
  return %v : tensor<4x1xi32>
}

// CHECK-LABEL: func.func @topk_k1_dim1_trivial(
// CHECK-SAME:    %[[X:.*]]: tensor<3x1xf32>) -> (tensor<3x1xf32>, tensor<3x1xi32>) {
// CHECK-NEXT:    %[[Z:.*]] = stablehlo.constant dense<0> : tensor<3x1xi32>
// CHECK-NEXT:    return %[[X]], %[[Z]] : tensor<3x1xf32>, tensor<3x1xi32>
func.func @topk_k1_dim1_trivial(%x: tensor<3x1xf32>) -> (tensor<3x1xf32>, tensor<3x1xi32>) {
  %v, %i = chlo.top_k(%x, k = 1) : tensor<3x1xf32> -> (tensor<3x1xf32>, tensor<3x1xi32>)
  return %v, %i : tensor<3x1xf32>, tensor<3x1xi32>
}

// CHECK-LABEL: func.func @topk_k2_untouched(
// CHECK:         chlo.top_k(%{{.*}}, k = 2)
func.func @topk_k2_untouched(%x: tensor<3x5xf32>) -> (tensor<3x2xf32>, tensor<3x2xi32>) {
  %v, %i = chlo.top_k(%x, k = 2) : tensor<3x5xf32> -> (tensor<3x2xf32>, tensor<3x2xi32>)
  return %v, %i : tensor<3x2xf32>, tensor<3x2xi32>
}

// CHECK-LABEL: func.func @fft_factor_scalar(
// CHECK-SAME:    %[[A:.*]]: tensor<64x64xcomplex<f64>>, %[[S:.*]]: tensor<complex<f64>>) -> tensor<64x64xcomplex<f64>> {
// CHECK-DAG:     %[[FFT:.*]] = stablehlo.fft %[[A]], type =  FFT, length = [64, 64]
// CHECK-DAG:     %[[BCAST:.*]] = stablehlo.broadcast_in_dim %[[S]], dims = []{{.*}} : (tensor<complex<f64>>) -> tensor<64x64xcomplex<f64>>
// CHECK:         %[[MUL:.*]] = stablehlo.multiply %[[BCAST]], %[[FFT]] : tensor<64x64xcomplex<f64>>
// CHECK:         return %[[MUL]] : tensor<64x64xcomplex<f64>>
func.func @fft_factor_scalar(%A: tensor<64x64xcomplex<f64>>, %s: tensor<complex<f64>>) -> tensor<64x64xcomplex<f64>> {
  %b = stablehlo.broadcast_in_dim %s, dims = [] : (tensor<complex<f64>>) -> tensor<64x64xcomplex<f64>>
  %m = stablehlo.multiply %A, %b : tensor<64x64xcomplex<f64>>
  %f = stablehlo.fft %m, type = FFT, length = [64, 64] : (tensor<64x64xcomplex<f64>>) -> tensor<64x64xcomplex<f64>>
  return %f : tensor<64x64xcomplex<f64>>
}

// CHECK-LABEL: func.func @rfft_factor_scalar(
// CHECK-SAME:    %[[A:.*]]: tensor<8x8xf32>, %[[S:.*]]: tensor<f32>) -> tensor<8x5xcomplex<f32>> {
// CHECK-DAG:     %[[ZERO:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-DAG:     %[[FFT:.*]] = stablehlo.fft %[[A]], type =  RFFT, length = [8, 8]
// CHECK-DAG:     %[[CS:.*]] = stablehlo.complex %[[S]], %[[ZERO]]
// CHECK-DAG:     %[[BCAST:.*]] = stablehlo.broadcast_in_dim %[[CS]], dims = []{{.*}} : (tensor<complex<f32>>) -> tensor<8x5xcomplex<f32>>
// CHECK:         %[[MUL:.*]] = stablehlo.multiply %[[BCAST]], %[[FFT]] : tensor<8x5xcomplex<f32>>
// CHECK:         return %[[MUL]] : tensor<8x5xcomplex<f32>>
func.func @rfft_factor_scalar(%A: tensor<8x8xf32>, %s: tensor<f32>) -> tensor<8x5xcomplex<f32>> {
  %b = stablehlo.broadcast_in_dim %s, dims = [] : (tensor<f32>) -> tensor<8x8xf32>
  %m = stablehlo.multiply %A, %b : tensor<8x8xf32>
  %f = stablehlo.fft %m, type = RFFT, length = [8, 8] : (tensor<8x8xf32>) -> tensor<8x5xcomplex<f32>>
  return %f : tensor<8x5xcomplex<f32>>
}

// CHECK-LABEL: func.func @irfft_complex_scalar_untouched(
// CHECK:         %[[MUL:.*]] = stablehlo.multiply
// CHECK-NEXT:    %[[FFT:.*]] = stablehlo.fft %[[MUL]], type =  IRFFT
func.func @irfft_complex_scalar_untouched(%A: tensor<8x5xcomplex<f32>>, %s: tensor<complex<f32>>) -> tensor<8x8xf32> {
  %b = stablehlo.broadcast_in_dim %s, dims = [] : (tensor<complex<f32>>) -> tensor<8x5xcomplex<f32>>
  %m = stablehlo.multiply %A, %b : tensor<8x5xcomplex<f32>>
  %f = stablehlo.fft %m, type = IRFFT, length = [8, 8] : (tensor<8x5xcomplex<f32>>) -> tensor<8x8xf32>
  return %f : tensor<8x8xf32>
}

// CHECK-LABEL: func.func @fft_preserves_input_fusion_when_no_output_consumer(
// CHECK:         %[[ADD:.*]] = stablehlo.add
// CHECK:         %[[MUL:.*]] = stablehlo.multiply
// CHECK-NEXT:    %[[FFT:.*]] = stablehlo.fft %[[MUL]], type =  FFT
func.func @fft_preserves_input_fusion_when_no_output_consumer(
    %A: tensor<16xcomplex<f32>>, %B: tensor<16xcomplex<f32>>, %s: tensor<complex<f32>>) -> tensor<16xcomplex<f32>> {
  %sum = stablehlo.add %A, %B : tensor<16xcomplex<f32>>
  %b = stablehlo.broadcast_in_dim %s, dims = [] : (tensor<complex<f32>>) -> tensor<16xcomplex<f32>>
  %m = stablehlo.multiply %sum, %b : tensor<16xcomplex<f32>>
  %f = stablehlo.fft %m, type = FFT, length = [16] : (tensor<16xcomplex<f32>>) -> tensor<16xcomplex<f32>>
  return %f : tensor<16xcomplex<f32>>
}

// CHECK-LABEL: func.func @compare_neg_inf_default(
// CHECK:         stablehlo.compare EQ
// FINITE-LABEL: func.func @compare_neg_inf_default(
// FINITE-NEXT:   %[[FALSE:.*]] = stablehlo.constant dense<false> : tensor<i1>
// FINITE-NEXT:   return %[[FALSE]] : tensor<i1>
func.func @compare_neg_inf_default(%arg0: tensor<f64>) -> tensor<i1> {
  %cst = stablehlo.constant dense<0xFFF0000000000000> : tensor<f64>
  %0 = stablehlo.compare EQ, %arg0, %cst : (tensor<f64>, tensor<f64>) -> tensor<i1>
  return %0 : tensor<i1>
}

// FINITE-LABEL: func.func @compare_inf_directions(
// FINITE-DAG:    %[[TRUE:.*]] = stablehlo.constant dense<true> : tensor<4xi1>
// FINITE-DAG:    %[[FALSE:.*]] = stablehlo.constant dense<false> : tensor<4xi1>
// FINITE:        return %[[TRUE]], %[[TRUE]], %[[FALSE]], %[[TRUE]] : tensor<4xi1>, tensor<4xi1>, tensor<4xi1>, tensor<4xi1>
func.func @compare_inf_directions(%x: tensor<4xf32>) -> (tensor<4xi1>, tensor<4xi1>, tensor<4xi1>, tensor<4xi1>) {
  %pinf = stablehlo.constant dense<0x7F800000> : tensor<4xf32>
  %ninf = stablehlo.constant dense<0xFF800000> : tensor<4xf32>
  %ne = stablehlo.compare NE, %x, %pinf : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
  %lt_pinf = stablehlo.compare LT, %x, %pinf : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
  %lt_ninf = stablehlo.compare LT, %x, %ninf : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
  %pinf_gt_x = stablehlo.compare GT, %pinf, %x : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
  return %ne, %lt_pinf, %lt_ninf, %pinf_gt_x : tensor<4xi1>, tensor<4xi1>, tensor<4xi1>, tensor<4xi1>
}

// CHECK-LABEL: func.func @compare_inf_guaranteed_attr(
// CHECK-NEXT:    %[[FALSE:.*]] = stablehlo.constant dense<false> : tensor<4xi1>
// CHECK-NEXT:    return %[[FALSE]] : tensor<4xi1>
func.func @compare_inf_guaranteed_attr(
    %x: tensor<4xf32> {enzymexla.finite = #enzymexla.guaranteed<GUARANTEED>, enzymexla.no_nan = #enzymexla.guaranteed<GUARANTEED>}) -> tensor<4xi1> {
  %ninf = stablehlo.constant dense<0xFF800000> : tensor<4xf32>
  %eq = stablehlo.compare EQ, %x, %ninf : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
  return %eq : tensor<4xi1>
}
