// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

module {
  func.func @main(%arg0: tensor<274xcomplex<f64>> {enzymexla.memory_effects = [], tf.aliasing_output = 0 : i32}, %arg1: tensor<128x128xcomplex<f64>> {enzymexla.memory_effects = [], tf.aliasing_output = 1 : i32}, %arg2: tensor<64x64xcomplex<f64>> {enzymexla.memory_effects = []}, %arg3: tensor<4096xi64> {enzymexla.memory_effects = []}, %arg4: tensor<4096xcomplex<f64>> {enzymexla.memory_effects = []}, %arg5: tensor<274x16384xcomplex<f64>> {enzymexla.memory_effects = []}, %arg6: tensor<64x64xf64> {enzymexla.memory_effects = []}) -> (tensor<274xcomplex<f64>>, tensor<128x128xcomplex<f64>>) attributes {enzymexla.memory_effects = []} {
    %c = stablehlo.constant dense<1> : tensor<4096x1xi64>
    %c_0 = stablehlo.constant dense<128> : tensor<4096x1xi64>
    %cst = stablehlo.constant dense<(0.000000e+00,5.000000e+00)> : tensor<128x128xcomplex<f64>>
    %c_1 = stablehlo.constant dense<128> : tensor<4096x2xi64>
    %0 = stablehlo.reshape %arg3 : (tensor<4096xi64>) -> tensor<4096x1xi64>
    %1 = stablehlo.convert %arg6 : (tensor<64x64xf64>) -> tensor<64x64xcomplex<f64>>
    %2 = stablehlo.reshape %1 : (tensor<64x64xcomplex<f64>>) -> tensor<4096xcomplex<f64>>
    %3 = stablehlo.multiply %2, %arg4 {enzymexla.symmetric_matrix = [#enzymexla<guaranteed NOTGUARANTEED>]} : tensor<4096xcomplex<f64>>
    %4 = stablehlo.subtract %0, %c : tensor<4096x1xi64>
    %5 = stablehlo.divide %4, %c_0 : tensor<4096x1xi64>
    %6 = stablehlo.concatenate %4, %5, dim = 1 : (tensor<4096x1xi64>, tensor<4096x1xi64>) -> tensor<4096x2xi64>
    %7 = stablehlo.remainder %6, %c_1 : tensor<4096x2xi64>
    %8 = "stablehlo.scatter"(%cst, %7, %3) <{scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>}> ({
    ^bb0(%arg7: tensor<complex<f64>>, %arg8: tensor<complex<f64>>):
      stablehlo.return %arg8 : tensor<complex<f64>>
    }) {enzymexla.symmetric_matrix = [#enzymexla<guaranteed NOTGUARANTEED>]} : (tensor<128x128xcomplex<f64>>, tensor<4096x2xi64>, tensor<4096xcomplex<f64>>) -> tensor<128x128xcomplex<f64>>
    %9 = stablehlo.transpose %8, dims = [1, 0] : (tensor<128x128xcomplex<f64>>) -> tensor<128x128xcomplex<f64>>
    %10 = stablehlo.fft %9, type =  FFT, length = [128, 128] : (tensor<128x128xcomplex<f64>>) -> tensor<128x128xcomplex<f64>>
    %11 = stablehlo.reshape %10 : (tensor<128x128xcomplex<f64>>) -> tensor<16384x1xcomplex<f64>>
    %12 = stablehlo.dot_general %arg5, %11, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<274x16384xcomplex<f64>>, tensor<16384x1xcomplex<f64>>) -> tensor<274x1xcomplex<f64>>
    %13 = stablehlo.reshape %12 : (tensor<274x1xcomplex<f64>>) -> tensor<274xcomplex<f64>>
    return %13, %10 : tensor<274xcomplex<f64>>, tensor<128x128xcomplex<f64>>
  }
}

// CHECK: %cst = stablehlo.constant dense<5.000000e+00> : tensor<128x128xf64>
// CHECK: %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<128x128xf64>
// CHECK: %8 = stablehlo.real %3 : (tensor<4096xcomplex<f64>>) -> tensor<4096xf64>
// CHECK-NEXT: %9 = stablehlo.imag %3 : (tensor<4096xcomplex<f64>>) -> tensor<4096xf64>
// CHECK-NEXT: %10 = "stablehlo.scatter"(%cst_0, %7, %8) <{scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>}> ({
// CHECK-NEXT: ^bb0(%arg7: tensor<f64>, %arg8: tensor<f64>):
// CHECK-NEXT:   stablehlo.return %arg8 : tensor<f64>
// CHECK-NEXT: }) {enzymexla.symmetric_matrix = [#enzymexla<guaranteed NOTGUARANTEED>]} : (tensor<128x128xf64>, tensor<4096x2xi64>, tensor<4096xf64>) -> tensor<128x128xf64>
// CHECK-NEXT: %11 = "stablehlo.scatter"(%cst, %7, %9) <{scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>}> ({
// CHECK-NEXT: ^bb0(%arg7: tensor<f64>, %arg8: tensor<f64>):
// CHECK-NEXT:   stablehlo.return %arg8 : tensor<f64>
// CHECK-NEXT: }) : (tensor<128x128xf64>, tensor<4096x2xi64>, tensor<4096xf64>) -> tensor<128x128xf64>
// CHECK-NEXT: %12 = stablehlo.complex %10, %11 {enzymexla.symmetric_matrix = [#enzymexla<guaranteed NOTGUARANTEED>]} : tensor<128x128xcomplex<f64>>
