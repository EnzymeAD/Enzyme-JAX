// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

func.func @main1(%arg0: tensor<9x8x7xf64>, %arg1: tensor<i64>) -> tensor<1x4x3xf64> {
  %c = stablehlo.constant dense<0> : tensor<i32>
  %c_0 = stablehlo.constant dense<1> : tensor<i32>
  %0 = stablehlo.convert %arg1 : (tensor<i64>) -> tensor<i32>
  %1 = stablehlo.subtract %0, %c_0 : tensor<i32>
  %2 = stablehlo.slice %arg0 [4:6, 1:5, 0:4] : (tensor<9x8x7xf64>) -> tensor<2x4x4xf64>
  %3 = stablehlo.dynamic_slice %2, %1, %c, %1, sizes = [1, 4, 3] : (tensor<2x4x4xf64>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x4x3xf64>
  return %3 : tensor<1x4x3xf64>
}

// CHECK: func.func @main1(%arg0: tensor<9x8x7xf64>, %arg1: tensor<i64>) -> tensor<1x4x3xf64> {
// CHECK-NEXT:   %c = stablehlo.constant dense<3> : tensor<i32>
// CHECK-NEXT:   %c_0 = stablehlo.constant dense<1> : tensor<i32>
// CHECK-NEXT:   %0 = stablehlo.convert %arg1 : (tensor<i64>) -> tensor<i32>
// CHECK-NEXT:   %1 = stablehlo.subtract %0, %c_0 : tensor<i32>
// CHECK-NEXT:   %2 = stablehlo.add %0, %c : tensor<i32>
// CHECK-NEXT:   %3 = stablehlo.dynamic_slice %arg0, %2, %c_0, %1, sizes = [1, 4, 3] : (tensor<9x8x7xf64>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x4x3xf64>
// CHECK-NEXT:   return %3 : tensor<1x4x3xf64>
// CHECK-NEXT: }

func.func @main2(%arg0: tensor<9x8x7xf64>, %arg1: tensor<i64>) -> tensor<1x4x2xf64> {
  %c = stablehlo.constant dense<1> : tensor<i32>
  %c_0 = stablehlo.constant dense<2> : tensor<i64>
  %0 = stablehlo.add %arg1, %c_0 : tensor<i64>
  %1 = stablehlo.convert %arg1 : (tensor<i64>) -> tensor<i32>
  %2 = stablehlo.subtract %1, %c : tensor<i32>
  %3 = stablehlo.convert %0 : (tensor<i64>) -> tensor<i32>
  %4 = stablehlo.subtract %3, %c : tensor<i32>
  %5 = stablehlo.dynamic_slice %arg0, %4, %c, %2, sizes = [2, 4, 4] : (tensor<9x8x7xf64>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<2x4x4xf64>
  %6 = stablehlo.slice %5 [0:1, 0:4, 0:2] : (tensor<2x4x4xf64>) -> tensor<1x4x2xf64>
  return %6 : tensor<1x4x2xf64>
}

// CHECK: func.func @main2(%arg0: tensor<9x8x7xf64>, %arg1: tensor<i64>) -> tensor<1x4x2xf64> {
// CHECK-NEXT:   %c = stablehlo.constant dense<1> : tensor<i32>
// CHECK-NEXT:   %c_0 = stablehlo.constant dense<2> : tensor<i64>
// CHECK-NEXT:   %0 = stablehlo.add %arg1, %c_0 : tensor<i64>
// CHECK-NEXT:   %1 = stablehlo.convert %arg1 : (tensor<i64>) -> tensor<i32>
// CHECK-NEXT:   %2 = stablehlo.subtract %1, %c : tensor<i32>
// CHECK-NEXT:   %3 = stablehlo.convert %0 : (tensor<i64>) -> tensor<i32>
// CHECK-NEXT:   %4 = stablehlo.subtract %3, %c : tensor<i32>
// CHECK-NEXT:   %5 = stablehlo.dynamic_slice %arg0, %4, %c, %2, sizes = [1, 4, 2] : (tensor<9x8x7xf64>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x4x2xf64>
// CHECK-NEXT:   return %5 : tensor<1x4x2xf64>
// CHECK-NEXT: }

func.func @main3(%arg0: tensor<9x8x7xf64>, %arg1: tensor<i64>) -> tensor<1x2x1xf64> {
  %c = stablehlo.constant dense<1> : tensor<i32>
  %c_0 = stablehlo.constant dense<2> : tensor<i64>
  %0 = stablehlo.transpose %arg0, dims = [2, 1, 0] : (tensor<9x8x7xf64>) -> tensor<7x8x9xf64>
  %1 = stablehlo.add %arg1, %c_0 : tensor<i64>
  %2 = stablehlo.convert %arg1 : (tensor<i64>) -> tensor<i32>
  %3 = stablehlo.subtract %2, %c : tensor<i32>
  %4 = stablehlo.convert %1 : (tensor<i64>) -> tensor<i32>
  %5 = stablehlo.subtract %4, %c : tensor<i32>
  %6 = stablehlo.dynamic_slice %0, %3, %c, %5, sizes = [4, 4, 2] : (tensor<7x8x9xf64>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<4x4x2xf64>
  %7 = stablehlo.dynamic_slice %6, %3, %3, %c, sizes = [1, 2, 1] : (tensor<4x4x2xf64>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x2x1xf64>
  return %7 : tensor<1x2x1xf64>
}

// CHECK: func.func @main3(%arg0: tensor<9x8x7xf64>, %arg1: tensor<i64>) -> tensor<1x2x1xf64> {
// CHECK-NEXT:   %c = stablehlo.constant dense<1> : tensor<i32>
// CHECK-NEXT:   %c_0 = stablehlo.constant dense<2> : tensor<i64>
// CHECK-NEXT:   %0 = stablehlo.add %arg1, %c_0 : tensor<i64>
// CHECK-NEXT:   %1 = stablehlo.convert %arg1 : (tensor<i64>) -> tensor<i32>
// CHECK-NEXT:   %2 = stablehlo.subtract %1, %c : tensor<i32>
// CHECK-NEXT:   %3 = stablehlo.convert %0 : (tensor<i64>) -> tensor<i32>
// CHECK-NEXT:   %4 = stablehlo.subtract %3, %c : tensor<i32>
// CHECK-NEXT:   %5 = stablehlo.add %2, %2 : tensor<i32>
// CHECK-NEXT:   %6 = stablehlo.add %c, %4 : tensor<i32>
// CHECK-NEXT:   %7 = stablehlo.dynamic_slice %arg0, %6, %1, %5, sizes = [1, 2, 1] : (tensor<9x8x7xf64>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x2x1xf64>
// CHECK-NEXT:   return %7 : tensor<1x2x1xf64>
// CHECK-NEXT: }
