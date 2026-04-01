// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt | FileCheck %s

func.func @main_add1(%arg0: tensor<8x2xf64>) -> tensor<2xf64> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<8x2xf64>) -> tensor<2x8xf64>
    %1 = stablehlo.slice %0 [0:2, 0:2] : (tensor<2x8xf64>) -> tensor<2x2xf64>
    %4 = stablehlo.slice %0 [0:2, 2:3] : (tensor<2x8xf64>) -> tensor<2x1xf64>
    %6 = stablehlo.slice %0 [0:2, 3:4] : (tensor<2x8xf64>) -> tensor<2x1xf64>
    %8 = stablehlo.slice %0 [0:2, 4:5] : (tensor<2x8xf64>) -> tensor<2x1xf64>
    %10 = stablehlo.slice %0 [0:2, 5:6] : (tensor<2x8xf64>) -> tensor<2x1xf64>
    %12 = stablehlo.slice %0 [0:2, 6:7] : (tensor<2x8xf64>) -> tensor<2x1xf64>
    %14 = stablehlo.slice %0 [0:2, 7:8] : (tensor<2x8xf64>) -> tensor<2x1xf64>
    %2 = stablehlo.reduce(%1 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<2x2xf64>, tensor<f64>) -> tensor<2xf64>
    %3 = stablehlo.reshape %2 : (tensor<2xf64>) -> tensor<2x1xf64>
    %5 = stablehlo.add %3, %4 : tensor<2x1xf64>
    %7 = stablehlo.add %5, %6 : tensor<2x1xf64>
    %9 = stablehlo.add %8, %12 : tensor<2x1xf64>
    %11 = stablehlo.add %10, %14 : tensor<2x1xf64>
    %13 = stablehlo.add %11, %9 : tensor<2x1xf64>
    %15 = stablehlo.add %13, %7 : tensor<2x1xf64>
    %16 = stablehlo.reshape %15 : (tensor<2x1xf64>) -> tensor<2xf64>
    return %16 : tensor<2xf64>
}

// CHECK: func.func @main_add1(%arg0: tensor<8x2xf64>) -> tensor<2xf64> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:     %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<8x2xf64>, tensor<f64>) -> tensor<2xf64>
// CHECK-NEXT:     return %0 : tensor<2xf64>
// CHECK-NEXT: }

func.func @main_add2(%arg0: tensor<6x2xf32>, %arg1: tensor<2x4xf32>) -> tensor<4xf32> {
    %0 = stablehlo.slice %arg0 [4:5, 0:2] : (tensor<6x2xf32>) -> tensor<1x2xf32>
    %1 = stablehlo.slice %arg0 [2:3, 0:2] : (tensor<6x2xf32>) -> tensor<1x2xf32>
    %2 = stablehlo.slice %arg0 [0:1, 0:2] : (tensor<6x2xf32>) -> tensor<1x2xf32>
    %3 = stablehlo.slice %arg0 [1:2, 0:2] : (tensor<6x2xf32>) -> tensor<1x2xf32>
    %4 = stablehlo.slice %arg0 [3:4, 0:2] : (tensor<6x2xf32>) -> tensor<1x2xf32>
    %5 = stablehlo.slice %arg0 [5:6, 0:2] : (tensor<6x2xf32>) -> tensor<1x2xf32>
    %6 = stablehlo.dot_general %arg1, %2, contracting_dims = [0] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2x4xf32>, tensor<1x2xf32>) -> tensor<4x1xf32>
    %7 = stablehlo.dot_general %arg1, %3, contracting_dims = [0] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2x4xf32>, tensor<1x2xf32>) -> tensor<4x1xf32>
    %8 = stablehlo.dot_general %arg1, %1, contracting_dims = [0] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2x4xf32>, tensor<1x2xf32>) -> tensor<4x1xf32>
    %9 = stablehlo.dot_general %arg1, %4, contracting_dims = [0] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2x4xf32>, tensor<1x2xf32>) -> tensor<4x1xf32>
    %10 = stablehlo.dot_general %arg1, %0, contracting_dims = [0] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2x4xf32>, tensor<1x2xf32>) -> tensor<4x1xf32>
    %11 = stablehlo.dot_general %arg1, %5, contracting_dims = [0] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2x4xf32>, tensor<1x2xf32>) -> tensor<4x1xf32>
    %12 = stablehlo.add %6, %7 : tensor<4x1xf32>
    %13 = stablehlo.add %12, %8 : tensor<4x1xf32>
    %14 = stablehlo.add %13, %9 : tensor<4x1xf32>
    %15 = stablehlo.add %14, %10 : tensor<4x1xf32>
    %16 = stablehlo.add %15, %11 : tensor<4x1xf32>
    %17 = stablehlo.reshape %16 : (tensor<4x1xf32>) -> tensor<4xf32>
    return %17 : tensor<4xf32>
}

// CHECK: func.func @main_add2(%arg0: tensor<6x2xf32>, %arg1: tensor<2x4xf32>) -> tensor<4xf32> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:     %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<6x2xf32>, tensor<f32>) -> tensor<2xf32>
// CHECK-NEXT:     %1 = stablehlo.dot_general %arg1, %0, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x4xf32>, tensor<2xf32>) -> tensor<4xf32>
// CHECK-NEXT:     return %1 : tensor<4xf32>
// CHECK-NEXT: }

func.func @main_add3(%arg0: tensor<16x2xf64>) -> tensor<2xf64> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<16x2xf64>) -> tensor<2x16xf64>
    %1 = stablehlo.slice %0 [0:2, 0:4:2] : (tensor<2x16xf64>) -> tensor<2x2xf64>
    %4 = stablehlo.slice %0 [0:2, 4:6:2] : (tensor<2x16xf64>) -> tensor<2x1xf64>
    %6 = stablehlo.slice %0 [0:2, 6:8:2] : (tensor<2x16xf64>) -> tensor<2x1xf64>
    %8 = stablehlo.slice %0 [0:2, 8:10:2] : (tensor<2x16xf64>) -> tensor<2x1xf64>
    %10 = stablehlo.slice %0 [0:2, 10:12:2] : (tensor<2x16xf64>) -> tensor<2x1xf64>
    %12 = stablehlo.slice %0 [0:2, 12:14:2] : (tensor<2x16xf64>) -> tensor<2x1xf64>
    %14 = stablehlo.slice %0 [0:2, 14:16:2] : (tensor<2x16xf64>) -> tensor<2x1xf64>
    %2 = stablehlo.reduce(%1 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<2x2xf64>, tensor<f64>) -> tensor<2xf64>
    %3 = stablehlo.reshape %2 : (tensor<2xf64>) -> tensor<2x1xf64>
    %5 = stablehlo.add %3, %4 : tensor<2x1xf64>
    %7 = stablehlo.add %5, %6 : tensor<2x1xf64>
    %9 = stablehlo.add %8, %12 : tensor<2x1xf64>
    %11 = stablehlo.add %10, %14 : tensor<2x1xf64>
    %13 = stablehlo.add %11, %9 : tensor<2x1xf64>
    %15 = stablehlo.add %13, %7 : tensor<2x1xf64>
    %16 = stablehlo.reshape %15 : (tensor<2x1xf64>) -> tensor<2xf64>
    return %16 : tensor<2xf64>
}

// CHECK: func.func @main_add3(%arg0: tensor<16x2xf64>) -> tensor<2xf64> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:     %0 = stablehlo.slice %arg0 [0:16:2, 0:2] : (tensor<16x2xf64>) -> tensor<8x2xf64>
// CHECK-NEXT:     %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<8x2xf64>, tensor<f64>) -> tensor<2xf64>
// CHECK-NEXT:     return %1 : tensor<2xf64>
// CHECK-NEXT: }

func.func @main_mul1(%arg0: tensor<8x2xf64>) -> tensor<2xf64> {
    %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<8x2xf64>) -> tensor<2x8xf64>
    %1 = stablehlo.slice %0 [0:2, 0:1] : (tensor<2x8xf64>) -> tensor<2x1xf64>
    %17 = stablehlo.slice %0 [0:2, 1:2] : (tensor<2x8xf64>) -> tensor<2x1xf64>
    %4 = stablehlo.slice %0 [0:2, 2:3] : (tensor<2x8xf64>) -> tensor<2x1xf64>
    %6 = stablehlo.slice %0 [0:2, 3:4] : (tensor<2x8xf64>) -> tensor<2x1xf64>
    %8 = stablehlo.slice %0 [0:2, 4:5] : (tensor<2x8xf64>) -> tensor<2x1xf64>
    %10 = stablehlo.slice %0 [0:2, 5:6] : (tensor<2x8xf64>) -> tensor<2x1xf64>
    %12 = stablehlo.slice %0 [0:2, 6:7] : (tensor<2x8xf64>) -> tensor<2x1xf64>
    %14 = stablehlo.slice %0 [0:2, 7:8] : (tensor<2x8xf64>) -> tensor<2x1xf64>
    %3 = stablehlo.multiply %17, %1 : tensor<2x1xf64>
    %5 = stablehlo.multiply %3, %4 : tensor<2x1xf64>
    %7 = stablehlo.multiply %5, %6 : tensor<2x1xf64>
    %9 = stablehlo.multiply %8, %12 : tensor<2x1xf64>
    %11 = stablehlo.multiply %10, %14 : tensor<2x1xf64>
    %13 = stablehlo.multiply %11, %9 : tensor<2x1xf64>
    %15 = stablehlo.multiply %13, %7 : tensor<2x1xf64>
    %16 = stablehlo.reshape %15 : (tensor<2x1xf64>) -> tensor<2xf64>
    return %16 : tensor<2xf64>
}

// CHECK: func.func @main_mul1(%arg0: tensor<8x2xf64>) -> tensor<2xf64> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<1.000000e+00> : tensor<f64>
// CHECK-NEXT:     %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.multiply across dimensions = [0] : (tensor<8x2xf64>, tensor<f64>) -> tensor<2xf64>
// CHECK-NEXT:     return %0 : tensor<2xf64>
// CHECK-NEXT: }

func.func @main_max1(%arg0: tensor<8x2xf64>) -> tensor<2xf64> {
    %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<8x2xf64>) -> tensor<2x8xf64>
    %1 = stablehlo.slice %0 [0:2, 0:1] : (tensor<2x8xf64>) -> tensor<2x1xf64>
    %17 = stablehlo.slice %0 [0:2, 1:2] : (tensor<2x8xf64>) -> tensor<2x1xf64>
    %4 = stablehlo.slice %0 [0:2, 2:3] : (tensor<2x8xf64>) -> tensor<2x1xf64>
    %6 = stablehlo.slice %0 [0:2, 3:4] : (tensor<2x8xf64>) -> tensor<2x1xf64>
    %8 = stablehlo.slice %0 [0:2, 4:5] : (tensor<2x8xf64>) -> tensor<2x1xf64>
    %10 = stablehlo.slice %0 [0:2, 5:6] : (tensor<2x8xf64>) -> tensor<2x1xf64>
    %12 = stablehlo.slice %0 [0:2, 6:7] : (tensor<2x8xf64>) -> tensor<2x1xf64>
    %14 = stablehlo.slice %0 [0:2, 7:8] : (tensor<2x8xf64>) -> tensor<2x1xf64>
    %3 = stablehlo.maximum %17, %1 : tensor<2x1xf64>
    %5 = stablehlo.maximum %3, %4 : tensor<2x1xf64>
    %7 = stablehlo.maximum %5, %6 : tensor<2x1xf64>
    %9 = stablehlo.maximum %8, %12 : tensor<2x1xf64>
    %11 = stablehlo.maximum %10, %14 : tensor<2x1xf64>
    %13 = stablehlo.maximum %11, %9 : tensor<2x1xf64>
    %15 = stablehlo.maximum %13, %7 : tensor<2x1xf64>
    %16 = stablehlo.reshape %15 : (tensor<2x1xf64>) -> tensor<2xf64>
    return %16 : tensor<2xf64>
}

// CHECK: func.func @main_max1(%arg0: tensor<8x2xf64>) -> tensor<2xf64> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0xFFF0000000000000> : tensor<f64>
// CHECK-NEXT:     %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.maximum across dimensions = [0] : (tensor<8x2xf64>, tensor<f64>) -> tensor<2xf64>
// CHECK-NEXT:     return %0 : tensor<2xf64>
// CHECK-NEXT: }

func.func @main_min1(%arg0: tensor<8x2xf64>) -> tensor<2xf64> {
    %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<8x2xf64>) -> tensor<2x8xf64>
    %1 = stablehlo.slice %0 [0:2, 0:1] : (tensor<2x8xf64>) -> tensor<2x1xf64>
    %17 = stablehlo.slice %0 [0:2, 1:2] : (tensor<2x8xf64>) -> tensor<2x1xf64>
    %4 = stablehlo.slice %0 [0:2, 2:3] : (tensor<2x8xf64>) -> tensor<2x1xf64>
    %6 = stablehlo.slice %0 [0:2, 3:4] : (tensor<2x8xf64>) -> tensor<2x1xf64>
    %8 = stablehlo.slice %0 [0:2, 4:5] : (tensor<2x8xf64>) -> tensor<2x1xf64>
    %10 = stablehlo.slice %0 [0:2, 5:6] : (tensor<2x8xf64>) -> tensor<2x1xf64>
    %12 = stablehlo.slice %0 [0:2, 6:7] : (tensor<2x8xf64>) -> tensor<2x1xf64>
    %14 = stablehlo.slice %0 [0:2, 7:8] : (tensor<2x8xf64>) -> tensor<2x1xf64>
    %3 = stablehlo.minimum %17, %1 : tensor<2x1xf64>
    %5 = stablehlo.minimum %3, %4 : tensor<2x1xf64>
    %7 = stablehlo.minimum %5, %6 : tensor<2x1xf64>
    %9 = stablehlo.minimum %8, %12 : tensor<2x1xf64>
    %11 = stablehlo.minimum %10, %14 : tensor<2x1xf64>
    %13 = stablehlo.minimum %11, %9 : tensor<2x1xf64>
    %15 = stablehlo.minimum %13, %7 : tensor<2x1xf64>
    %16 = stablehlo.reshape %15 : (tensor<2x1xf64>) -> tensor<2xf64>
    return %16 : tensor<2xf64>
}

// CHECK: func.func @main_min1(%arg0: tensor<8x2xf64>) -> tensor<2xf64> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0x7FF0000000000000> : tensor<f64>
// CHECK-NEXT:     %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.minimum across dimensions = [0] : (tensor<8x2xf64>, tensor<f64>) -> tensor<2xf64>
// CHECK-NEXT:     return %0 : tensor<2xf64>
// CHECK-NEXT: }

func.func @main_with_reshape(%arg0: tensor<7x5x3xf32>) -> tensor<7x3xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.slice %arg0 [0:7, 0:2, 0:3] : (tensor<7x5x3xf32>) -> tensor<7x2x3xf32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<7x2x3xf32>, tensor<f32>) -> tensor<7x3xf32>
    %2 = stablehlo.slice %arg0 [0:7, 2:3, 0:3] : (tensor<7x5x3xf32>) -> tensor<7x1x3xf32>
    %3 = stablehlo.reshape %2 : (tensor<7x1x3xf32>) -> tensor<7x3xf32>
    %4 = stablehlo.add %1, %3 : tensor<7x3xf32>
    %5 = stablehlo.slice %arg0 [0:7, 3:4, 0:3] : (tensor<7x5x3xf32>) -> tensor<7x1x3xf32>
    %6 = stablehlo.reshape %5 : (tensor<7x1x3xf32>) -> tensor<7x3xf32>
    %7 = stablehlo.add %4, %6 : tensor<7x3xf32>
    %8 = stablehlo.slice %arg0 [0:7, 4:5, 0:3] : (tensor<7x5x3xf32>) -> tensor<7x1x3xf32>
    %9 = stablehlo.reshape %8 : (tensor<7x1x3xf32>) -> tensor<7x3xf32>
    %10 = stablehlo.add %7, %9 : tensor<7x3xf32>
    return %10 : tensor<7x3xf32>
}

// CHECK: func.func @main_with_reshape(%arg0: tensor<7x5x3xf32>) -> tensor<7x3xf32> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:     %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<7x5x3xf32>, tensor<f32>) -> tensor<7x3xf32>
// CHECK-NEXT:     return %0 : tensor<7x3xf32>
// CHECK-NEXT: }

func.func @main_with_reshape2(%arg0: tensor<7x5x3xf32>) -> tensor<7x3xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.slice %arg0 [0:7, 0:2, 0:3] : (tensor<7x5x3xf32>) -> tensor<7x2x3xf32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<7x2x3xf32>, tensor<f32>) -> tensor<7x3xf32>
    %2 = stablehlo.slice %arg0 [0:7, 2:5, 0:3] : (tensor<7x5x3xf32>) -> tensor<7x3x3xf32>
    %3 = stablehlo.reduce(%2 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<7x3x3xf32>, tensor<f32>) -> tensor<7x3xf32>
    %4 = stablehlo.add %3, %1 : tensor<7x3xf32>
    return %4 : tensor<7x3xf32>
}

// CHECK: func.func @main_with_reshape2(%arg0: tensor<7x5x3xf32>) -> tensor<7x3xf32> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:     %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<7x5x3xf32>, tensor<f32>) -> tensor<7x3xf32>
// CHECK-NEXT:     return %0 : tensor<7x3xf32>
// CHECK-NEXT: }

func.func @main_complex_add(%arg0: tensor<8x2xcomplex<f64>>) -> tensor<2xcomplex<f64>> {
    %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<8x2xcomplex<f64>>) -> tensor<2x8xcomplex<f64>>
    %1 = stablehlo.slice %0 [0:2, 0:1] : (tensor<2x8xcomplex<f64>>) -> tensor<2x1xcomplex<f64>>
    %17 = stablehlo.slice %0 [0:2, 1:2] : (tensor<2x8xcomplex<f64>>) -> tensor<2x1xcomplex<f64>>
    %4 = stablehlo.slice %0 [0:2, 2:3] : (tensor<2x8xcomplex<f64>>) -> tensor<2x1xcomplex<f64>>
    %6 = stablehlo.slice %0 [0:2, 3:4] : (tensor<2x8xcomplex<f64>>) -> tensor<2x1xcomplex<f64>>
    %8 = stablehlo.slice %0 [0:2, 4:5] : (tensor<2x8xcomplex<f64>>) -> tensor<2x1xcomplex<f64>>
    %10 = stablehlo.slice %0 [0:2, 5:6] : (tensor<2x8xcomplex<f64>>) -> tensor<2x1xcomplex<f64>>
    %12 = stablehlo.slice %0 [0:2, 6:7] : (tensor<2x8xcomplex<f64>>) -> tensor<2x1xcomplex<f64>>
    %14 = stablehlo.slice %0 [0:2, 7:8] : (tensor<2x8xcomplex<f64>>) -> tensor<2x1xcomplex<f64>>
    %3 = stablehlo.add %17, %1 : tensor<2x1xcomplex<f64>>
    %5 = stablehlo.add %3, %4 : tensor<2x1xcomplex<f64>>
    %7 = stablehlo.add %5, %6 : tensor<2x1xcomplex<f64>>
    %9 = stablehlo.add %8, %12 : tensor<2x1xcomplex<f64>>
    %11 = stablehlo.add %10, %14 : tensor<2x1xcomplex<f64>>
    %13 = stablehlo.add %11, %9 : tensor<2x1xcomplex<f64>>
    %15 = stablehlo.add %13, %7 : tensor<2x1xcomplex<f64>>
    %16 = stablehlo.reshape %15 : (tensor<2x1xcomplex<f64>>) -> tensor<2xcomplex<f64>>
    return %16 : tensor<2xcomplex<f64>>
}

// CHECK: func.func @main_complex_add(%arg0: tensor<8x2xcomplex<f64>>) -> tensor<2xcomplex<f64>> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<(0.000000e+00,0.000000e+00)> : tensor<complex<f64>>
// CHECK-NEXT:     %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<8x2xcomplex<f64>>, tensor<complex<f64>>) -> tensor<2xcomplex<f64>>
// CHECK-NEXT:     return %0 : tensor<2xcomplex<f64>>
// CHECK-NEXT: }

func.func @main_complex_mul(%arg0: tensor<8x2xcomplex<f64>>) -> tensor<2xcomplex<f64>> {
    %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<8x2xcomplex<f64>>) -> tensor<2x8xcomplex<f64>>
    %1 = stablehlo.slice %0 [0:2, 0:1] : (tensor<2x8xcomplex<f64>>) -> tensor<2x1xcomplex<f64>>
    %17 = stablehlo.slice %0 [0:2, 1:2] : (tensor<2x8xcomplex<f64>>) -> tensor<2x1xcomplex<f64>>
    %4 = stablehlo.slice %0 [0:2, 2:3] : (tensor<2x8xcomplex<f64>>) -> tensor<2x1xcomplex<f64>>
    %6 = stablehlo.slice %0 [0:2, 3:4] : (tensor<2x8xcomplex<f64>>) -> tensor<2x1xcomplex<f64>>
    %8 = stablehlo.slice %0 [0:2, 4:5] : (tensor<2x8xcomplex<f64>>) -> tensor<2x1xcomplex<f64>>
    %10 = stablehlo.slice %0 [0:2, 5:6] : (tensor<2x8xcomplex<f64>>) -> tensor<2x1xcomplex<f64>>
    %12 = stablehlo.slice %0 [0:2, 6:7] : (tensor<2x8xcomplex<f64>>) -> tensor<2x1xcomplex<f64>>
    %14 = stablehlo.slice %0 [0:2, 7:8] : (tensor<2x8xcomplex<f64>>) -> tensor<2x1xcomplex<f64>>
    %3 = stablehlo.multiply %17, %1 : tensor<2x1xcomplex<f64>>
    %5 = stablehlo.multiply %3, %4 : tensor<2x1xcomplex<f64>>
    %7 = stablehlo.multiply %5, %6 : tensor<2x1xcomplex<f64>>
    %9 = stablehlo.multiply %8, %12 : tensor<2x1xcomplex<f64>>
    %11 = stablehlo.multiply %10, %14 : tensor<2x1xcomplex<f64>>
    %13 = stablehlo.multiply %11, %9 : tensor<2x1xcomplex<f64>>
    %15 = stablehlo.multiply %13, %7 : tensor<2x1xcomplex<f64>>
    %16 = stablehlo.reshape %15 : (tensor<2x1xcomplex<f64>>) -> tensor<2xcomplex<f64>>
    return %16 : tensor<2xcomplex<f64>>
}

// CHECK: func.func @main_complex_mul(%arg0: tensor<8x2xcomplex<f64>>) -> tensor<2xcomplex<f64>> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<(1.000000e+00,0.000000e+00)> : tensor<complex<f64>>
// CHECK-NEXT:     %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.multiply across dimensions = [0] : (tensor<8x2xcomplex<f64>>, tensor<complex<f64>>) -> tensor<2xcomplex<f64>>
// CHECK-NEXT:     return %0 : tensor<2xcomplex<f64>>
// CHECK-NEXT: }

