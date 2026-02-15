// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

func.func @main(%arg0: tensor<1024xf32>, %arg1: tensor<1024x32xf32>, %arg2: tensor<1024x32xf32>) -> (tensor<24xf32>, tensor<1024x32xf32>) {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<24x1024xf32>
    %0 = stablehlo.iota dim = 0 : tensor<24x2xi64>
    %1 = stablehlo.dot_general %cst, %arg1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<24x1024xf32>, tensor<1024x32xf32>) -> tensor<24x32xf32>
    %2 = stablehlo.broadcast_in_dim %arg0, dims = [0] : (tensor<1024xf32>) -> tensor<1024x32xf32>
    %3 = "stablehlo.gather"(%1, %0) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1], start_index_map = [0, 1], index_vector_dim = 1>, slice_sizes = array<i64: 1, 1>}> : (tensor<24x32xf32>, tensor<24x2xi64>) -> tensor<24xf32>
    return %3, %2 : tensor<24xf32>, tensor<1024x32xf32>
}

// CHECK:   func.func @main(%arg0: tensor<1024xf32>, %arg1: tensor<1024x32xf32>, %arg2: tensor<1024x32xf32>) -> (tensor<24xf32>, tensor<1024x32xf32>) {
// CHECK-NEXT{LITERAL}:     %c = stablehlo.constant dense<[[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7], [8, 8], [9, 9], [10, 10], [11, 11], [12, 12], [13, 13], [14, 14], [15, 15], [16, 16], [17, 17], [18, 18], [19, 19], [20, 20], [21, 21], [22, 22], [23, 23]]> : tensor<24x2xi64>
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:     %0 = stablehlo.reduce(%arg1 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<1024x32xf32>, tensor<f32>) -> tensor<32xf32>
// CHECK-NEXT:     %1 = stablehlo.broadcast_in_dim %0, dims = [1] : (tensor<32xf32>) -> tensor<24x32xf32>
// CHECK-NEXT:     %2 = stablehlo.broadcast_in_dim %arg0, dims = [0] : (tensor<1024xf32>) -> tensor<1024x32xf32>
// CHECK-NEXT:     %3 = "stablehlo.gather"(%1, %c) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1], start_index_map = [0, 1], index_vector_dim = 1>, slice_sizes = array<i64: 1, 1>}> : (tensor<24x32xf32>, tensor<24x2xi64>) -> tensor<24xf32>
// CHECK-NEXT:     return %3, %2 : tensor<24xf32>, tensor<1024x32xf32>
// CHECK-NEXT:   }

func.func @main2(%arg0: tensor<1024xf32>, %arg1: tensor<1024x32xf32>, %arg2: tensor<1024x32xf32>) -> (tensor<24xf32>, tensor<1024x32xf32>) {
    %cst = stablehlo.constant dense<5.000000e+00> : tensor<24x1024xf32>
    %0 = stablehlo.iota dim = 0 : tensor<24x2xi64>
    %1 = stablehlo.dot_general %cst, %arg1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<24x1024xf32>, tensor<1024x32xf32>) -> tensor<24x32xf32>
    %2 = stablehlo.broadcast_in_dim %arg0, dims = [0] : (tensor<1024xf32>) -> tensor<1024x32xf32>
    %3 = "stablehlo.gather"(%1, %0) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1], start_index_map = [0, 1], index_vector_dim = 1>, slice_sizes = array<i64: 1, 1>}> : (tensor<24x32xf32>, tensor<24x2xi64>) -> tensor<24xf32>
    return %3, %2 : tensor<24xf32>, tensor<1024x32xf32>
}

// CHECK: func.func @main2(%arg0: tensor<1024xf32>, %arg1: tensor<1024x32xf32>, %arg2: tensor<1024x32xf32>) -> (tensor<24xf32>, tensor<1024x32xf32>) {
// CHECK-NEXT:     %cst = stablehlo.constant dense<5.000000e+00> : tensor<24xf32>
// CHECK-NEXT{LITERAL}:     %c = stablehlo.constant dense<[[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7], [8, 8], [9, 9], [10, 10], [11, 11], [12, 12], [13, 13], [14, 14], [15, 15], [16, 16], [17, 17], [18, 18], [19, 19], [20, 20], [21, 21], [22, 22], [23, 23]]> : tensor<24x2xi64>
// CHECK-NEXT:     %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:     %0 = stablehlo.reduce(%arg1 init: %cst_0) applies stablehlo.add across dimensions = [0] : (tensor<1024x32xf32>, tensor<f32>) -> tensor<32xf32>
// CHECK-NEXT:     %1 = stablehlo.broadcast_in_dim %0, dims = [1] : (tensor<32xf32>) -> tensor<24x32xf32>
// CHECK-NEXT:     %2 = stablehlo.broadcast_in_dim %arg0, dims = [0] : (tensor<1024xf32>) -> tensor<1024x32xf32>
// CHECK-NEXT:     %3 = "stablehlo.gather"(%1, %c) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1], start_index_map = [0, 1], index_vector_dim = 1>, slice_sizes = array<i64: 1, 1>}> : (tensor<24x32xf32>, tensor<24x2xi64>) -> tensor<24xf32>
// CHECK-NEXT:     %4 = stablehlo.multiply %3, %cst : tensor<24xf32>
// CHECK-NEXT:     return %4, %2 : tensor<24xf32>, tensor<1024x32xf32>
// CHECK-NEXT: }

func.func @main3(%arg0: tensor<2x2xcomplex<f64>> {enzymexla.memory_effects = []}, %arg1: tensor<2x2xcomplex<f64>> {enzymexla.memory_effects = []}) -> (tensor<2x2xcomplex<f64>>, tensor<2x2xcomplex<f64>>, tensor<2x2xcomplex<f64>>) attributes {enzymexla.memory_effects = []} {
    %cst = stablehlo.constant dense<(1.000000e+00,-0.000000e+00)> : tensor<2x2xcomplex<f64>>
    %0 = stablehlo.dot_general %arg1, %arg0, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x2xcomplex<f64>>, tensor<2x2xcomplex<f64>>) -> tensor<2x2xcomplex<f64>>
    %1 = stablehlo.dot_general %cst, %arg0, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2x2xcomplex<f64>>, tensor<2x2xcomplex<f64>>) -> tensor<2x2xcomplex<f64>>
    %2 = chlo.conj %1 : tensor<2x2xcomplex<f64>> -> tensor<2x2xcomplex<f64>>
    %3 = stablehlo.dot_general %arg1, %cst, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x2xcomplex<f64>>, tensor<2x2xcomplex<f64>>) -> tensor<2x2xcomplex<f64>>
    %4 = chlo.conj %3 : tensor<2x2xcomplex<f64>> -> tensor<2x2xcomplex<f64>>
    return %4, %2, %0 : tensor<2x2xcomplex<f64>>, tensor<2x2xcomplex<f64>>, tensor<2x2xcomplex<f64>>
}

// CHECK: func.func @main3(%arg0: tensor<2x2xcomplex<f64>> {enzymexla.memory_effects = []}, %arg1: tensor<2x2xcomplex<f64>> {enzymexla.memory_effects = []}) -> (tensor<2x2xcomplex<f64>>, tensor<2x2xcomplex<f64>>, tensor<2x2xcomplex<f64>>) attributes {enzymexla.memory_effects = []} {
// CHECK-NEXT:     %cst = stablehlo.constant dense<(1.000000e+00,-0.000000e+00)> : tensor<2xcomplex<f64>>
// CHECK-NEXT:     %cst_0 = stablehlo.constant dense<(0.000000e+00,0.000000e+00)> : tensor<complex<f64>>
// CHECK-NEXT:     %0 = stablehlo.dot_general %arg1, %arg0, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x2xcomplex<f64>>, tensor<2x2xcomplex<f64>>) -> tensor<2x2xcomplex<f64>>
// CHECK-NEXT:     %1 = stablehlo.reduce(%arg0 init: %cst_0) applies stablehlo.add across dimensions = [1] {enzymexla.complex_is_purely_real = [#enzymexla<guaranteed NOTGUARANTEED>]} : (tensor<2x2xcomplex<f64>>, tensor<complex<f64>>) -> tensor<2xcomplex<f64>>
// CHECK-NEXT:     %2 = stablehlo.multiply %1, %cst {enzymexla.complex_is_purely_real = [#enzymexla<guaranteed NOTGUARANTEED>]} : tensor<2xcomplex<f64>>
// CHECK-NEXT:     %3 = stablehlo.broadcast_in_dim %2, dims = [1] {enzymexla.complex_is_purely_real = [#enzymexla<guaranteed NOTGUARANTEED>]} : (tensor<2xcomplex<f64>>) -> tensor<2x2xcomplex<f64>>
// CHECK-NEXT:     %4 = chlo.conj %3 : tensor<2x2xcomplex<f64>> -> tensor<2x2xcomplex<f64>>
// CHECK-NEXT:     %5 = stablehlo.reduce(%arg1 init: %cst_0) applies stablehlo.add across dimensions = [0] {enzymexla.complex_is_purely_real = [#enzymexla<guaranteed NOTGUARANTEED>]} : (tensor<2x2xcomplex<f64>>, tensor<complex<f64>>) -> tensor<2xcomplex<f64>>
// CHECK-NEXT:     %6 = stablehlo.multiply %5, %cst {enzymexla.complex_is_purely_real = [#enzymexla<guaranteed NOTGUARANTEED>]} : tensor<2xcomplex<f64>>
// CHECK-NEXT:     %7 = stablehlo.broadcast_in_dim %6, dims = [0] {enzymexla.complex_is_purely_real = [#enzymexla<guaranteed NOTGUARANTEED>]} : (tensor<2xcomplex<f64>>) -> tensor<2x2xcomplex<f64>>
// CHECK-NEXT:     %8 = chlo.conj %7 : tensor<2x2xcomplex<f64>> -> tensor<2x2xcomplex<f64>>
// CHECK-NEXT:     return %8, %4, %0 : tensor<2x2xcomplex<f64>>, tensor<2x2xcomplex<f64>>, tensor<2x2xcomplex<f64>>
// CHECK-NEXT: }

func.func @main4(%arg0: tensor<2x16xf32> {enzymexla.memory_effects = []}, %arg1: tensor<16xf32> {enzymexla.memory_effects = []}, %arg2: tensor<16x16xf32> {enzymexla.memory_effects = []}, %arg3: tensor<16xf32> {enzymexla.memory_effects = []}, %arg4: tensor<16x1xf32> {enzymexla.memory_effects = []}, %arg5: tensor<1xf32> {enzymexla.memory_effects = []}, %arg6: tensor<2xf32> {enzymexla.memory_effects = []}) -> tensor<2xf32> attributes {enzymexla.memory_effects = []} {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<1xf32>
    %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<16xf32>
    %0 = stablehlo.reshape %arg4 : (tensor<16x1xf32>) -> tensor<1x16xf32>
    %1 = stablehlo.dot_general %arg0, %arg6, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x16xf32>, tensor<2xf32>) -> tensor<16xf32>
    %2 = stablehlo.add %1, %arg1 : tensor<16xf32>
    %3 = stablehlo.tanh %2 : tensor<16xf32>
    %4 = stablehlo.dot_general %arg2, %3, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<16x16xf32>, tensor<16xf32>) -> tensor<16xf32>
    %5 = stablehlo.add %4, %arg3 : tensor<16xf32>
    %6 = stablehlo.dot_general %cst, %0, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1xf32>, tensor<1x16xf32>) -> tensor<16xf32>
    %7 = stablehlo.tanh %5 : tensor<16xf32>
    %8 = stablehlo.multiply %7, %7 : tensor<16xf32>
    %9 = stablehlo.subtract %cst_0, %8 : tensor<16xf32>
    %10 = stablehlo.multiply %6, %9 : tensor<16xf32>
    %11 = stablehlo.dot_general %10, %arg2, contracting_dims = [0] x [1], precision = [DEFAULT, DEFAULT] : (tensor<16xf32>, tensor<16x16xf32>) -> tensor<16xf32>
    %12 = stablehlo.multiply %3, %3 : tensor<16xf32>
    %13 = stablehlo.subtract %cst_0, %12 : tensor<16xf32>
    %14 = stablehlo.multiply %11, %13 : tensor<16xf32>
    %15 = stablehlo.dot_general %14, %arg0, contracting_dims = [0] x [1], precision = [DEFAULT, DEFAULT] : (tensor<16xf32>, tensor<2x16xf32>) -> tensor<2xf32>
    return %15 : tensor<2xf32>
}

// CHECK: func.func @main4(%arg0: tensor<2x16xf32> {enzymexla.memory_effects = []}, %arg1: tensor<16xf32> {enzymexla.memory_effects = []}, %arg2: tensor<16x16xf32> {enzymexla.memory_effects = []}, %arg3: tensor<16xf32> {enzymexla.memory_effects = []}, %arg4: tensor<16x1xf32> {enzymexla.memory_effects = []}, %arg5: tensor<1xf32> {enzymexla.memory_effects = []}, %arg6: tensor<2xf32> {enzymexla.memory_effects = []}) -> tensor<2xf32> attributes {enzymexla.memory_effects = []} {
// CHECK-NEXT:     %cst = stablehlo.constant dense<1.000000e+00> : tensor<16xf32>
// CHECK-NEXT:     %0 = stablehlo.dot_general %arg0, %arg6, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x16xf32>, tensor<2xf32>) -> tensor<16xf32>
// CHECK-NEXT:     %1 = stablehlo.add %0, %arg1 : tensor<16xf32>
// CHECK-NEXT:     %2 = stablehlo.tanh %1 : tensor<16xf32>
// CHECK-NEXT:     %3 = stablehlo.dot_general %arg2, %2, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<16x16xf32>, tensor<16xf32>) -> tensor<16xf32>
// CHECK-NEXT:     %4 = stablehlo.add %3, %arg3 : tensor<16xf32>
// CHECK-NEXT:     %5 = stablehlo.reshape %arg4 : (tensor<16x1xf32>) -> tensor<16xf32>
// CHECK-NEXT:     %6 = stablehlo.tanh %4 : tensor<16xf32>
// CHECK-NEXT:     %7 = stablehlo.multiply %6, %6 : tensor<16xf32>
// CHECK-NEXT:     %8 = stablehlo.subtract %cst, %7 : tensor<16xf32>
// CHECK-NEXT:     %9 = stablehlo.multiply %5, %8 : tensor<16xf32>
// CHECK-NEXT:     %10 = stablehlo.dot_general %9, %arg2, contracting_dims = [0] x [1], precision = [DEFAULT, DEFAULT] : (tensor<16xf32>, tensor<16x16xf32>) -> tensor<16xf32>
// CHECK-NEXT:     %11 = stablehlo.multiply %2, %2 : tensor<16xf32>
// CHECK-NEXT:     %12 = stablehlo.subtract %cst, %11 : tensor<16xf32>
// CHECK-NEXT:     %13 = stablehlo.multiply %10, %12 : tensor<16xf32>
// CHECK-NEXT:     %14 = stablehlo.dot_general %13, %arg0, contracting_dims = [0] x [1], precision = [DEFAULT, DEFAULT] : (tensor<16xf32>, tensor<2x16xf32>) -> tensor<2xf32>
// CHECK-NEXT:     return %14 : tensor<2xf32>
// CHECK-NEXT: }
