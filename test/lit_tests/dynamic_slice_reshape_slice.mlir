// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=slice_reshape_dynamic_slice;dynamic_slice_reshape_dynamic_slice;dynamic_slice_reshape_slice" --transform-interpreter --enzyme-hlo-remove-transform --enzyme-hlo-opt %s | FileCheck %s

func.func @main1(%arg0: tensor<9x8x7xf64>, %arg1: tensor<i64>) -> tensor<2x1xf64> {
  %c = stablehlo.constant dense<1> : tensor<i32>
  %c_0 = stablehlo.constant dense<2> : tensor<i64>
  %0 = stablehlo.transpose %arg0, dims = [2, 1, 0] : (tensor<9x8x7xf64>) -> tensor<7x8x9xf64>
  %1 = stablehlo.add %arg1, %c_0 : tensor<i64>
  %2 = stablehlo.convert %arg1 : (tensor<i64>) -> tensor<i32>
  %3 = stablehlo.subtract %2, %c : tensor<i32>
  %4 = stablehlo.convert %1 : (tensor<i64>) -> tensor<i32>
  %5 = stablehlo.subtract %4, %c : tensor<i32>
  %6 = stablehlo.dynamic_slice %0, %3, %c, %5, sizes = [4, 4, 1] : (tensor<7x8x9xf64>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<4x4x1xf64>
  %7 = stablehlo.reshape %6 : (tensor<4x4x1xf64>) -> tensor<4x4xf64>
  %8 = stablehlo.slice %7 [1:3, 2:3] : (tensor<4x4xf64>) -> tensor<2x1xf64>
  return %8 : tensor<2x1xf64>
}

// CHECK: func.func @main1(%arg0: tensor<9x8x7xf64>, %arg1: tensor<i64>) -> tensor<2x1xf64> {
// CHECK-NEXT:   %c = stablehlo.constant dense<3> : tensor<i32>
// CHECK-NEXT:   %c_0 = stablehlo.constant dense<1> : tensor<i32>
// CHECK-NEXT:   %c_1 = stablehlo.constant dense<2> : tensor<i64>
// CHECK-NEXT:   %0 = stablehlo.transpose %arg0, dims = [2, 1, 0] : (tensor<9x8x7xf64>) -> tensor<7x8x9xf64>
// CHECK-NEXT:   %1 = stablehlo.add %arg1, %c_1 : tensor<i64>
// CHECK-NEXT:   %2 = stablehlo.convert %arg1 : (tensor<i64>) -> tensor<i32>
// CHECK-NEXT:   %3 = stablehlo.subtract %2, %c_0 : tensor<i32>
// CHECK-NEXT:   %4 = stablehlo.convert %1 : (tensor<i64>) -> tensor<i32>
// CHECK-NEXT:   %5 = stablehlo.subtract %4, %c_0 : tensor<i32>
// CHECK-NEXT:   %6 = stablehlo.add %c_0, %3 : tensor<i32>
// CHECK-NEXT:   %7 = stablehlo.dynamic_slice %0, %6, %c, %5, sizes = [2, 1, 1] : (tensor<7x8x9xf64>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<2x1x1xf64>
// CHECK-NEXT:   %8 = stablehlo.reshape %7 : (tensor<2x1x1xf64>) -> tensor<2x1xf64>
// CHECK-NEXT:   return %8 : tensor<2x1xf64>
// CHECK-NEXT: }

func.func @main2(%arg0: tensor<9x8x7xf64>, %arg1: tensor<i64>) -> tensor<1x3xf64> {
  %c = stablehlo.constant dense<1> : tensor<i32>
  %0 = stablehlo.slice %arg0 [4:6, 1:2, 0:4] : (tensor<9x8x7xf64>) -> tensor<2x1x4xf64>
  %1 = stablehlo.reshape %0 : (tensor<2x1x4xf64>) -> tensor<2x4xf64>
  %2 = stablehlo.convert %arg1 : (tensor<i64>) -> tensor<i32>
  %3 = stablehlo.subtract %2, %c : tensor<i32>
  %4 = stablehlo.dynamic_slice %1, %3, %3, sizes = [1, 3] : (tensor<2x4xf64>, tensor<i32>, tensor<i32>) -> tensor<1x3xf64>
  return %4 : tensor<1x3xf64>
}

// CHECK: func.func @main2(%arg0: tensor<9x8x7xf64>, %arg1: tensor<i64>) -> tensor<1x3xf64> {
// CHECK-NEXT:   %c = stablehlo.constant dense<2> : tensor<i32>
// CHECK-NEXT:   %c_0 = stablehlo.constant dense<7> : tensor<i32>
// CHECK-NEXT:   %c_1 = stablehlo.constant dense<1> : tensor<i32>
// CHECK-NEXT:   %0 = stablehlo.convert %arg1 : (tensor<i64>) -> tensor<i32>
// CHECK-NEXT:   %1 = stablehlo.subtract %0, %c_1 : tensor<i32>
// CHECK-NEXT:   %2 = stablehlo.add %0, %c_0 : tensor<i32>
// CHECK-NEXT:   %3 = stablehlo.dynamic_slice %arg0, %2, %c, %1, sizes = [1, 1, 3] : (tensor<9x8x7xf64>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x1x3xf64>
// CHECK-NEXT:   %4 = stablehlo.reshape %3 : (tensor<1x1x3xf64>) -> tensor<1x3xf64>
// CHECK-NEXT:   return %4 : tensor<1x3xf64>
// CHECK-NEXT: }

func.func @main3(%arg0: tensor<9x8x7xf64>, %arg1: tensor<i64>) -> tensor<2x1xf64> {
  %c = stablehlo.constant dense<1> : tensor<i32>
  %c_0 = stablehlo.constant dense<2> : tensor<i64>
  %0 = stablehlo.transpose %arg0, dims = [2, 1, 0] : (tensor<9x8x7xf64>) -> tensor<7x8x9xf64>
  %1 = stablehlo.add %arg1, %c_0 : tensor<i64>
  %2 = stablehlo.convert %arg1 : (tensor<i64>) -> tensor<i32>
  %3 = stablehlo.subtract %2, %c : tensor<i32>
  %4 = stablehlo.convert %1 : (tensor<i64>) -> tensor<i32>
  %5 = stablehlo.subtract %4, %c : tensor<i32>
  %6 = stablehlo.dynamic_slice %0, %3, %c, %5, sizes = [4, 4, 1] : (tensor<7x8x9xf64>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<4x4x1xf64>
  %7 = stablehlo.reshape %6 : (tensor<4x4x1xf64>) -> tensor<4x4xf64>
  %8 = stablehlo.dynamic_slice %7, %3, %3, sizes = [1, 2] : (tensor<4x4xf64>, tensor<i32>, tensor<i32>) -> tensor<1x2xf64>
  %9 = stablehlo.reshape %8 : (tensor<1x2xf64>) -> tensor<2x1xf64>
  return %9 : tensor<2x1xf64>
}

// CHECK: func.func @main3(%arg0: tensor<9x8x7xf64>, %arg1: tensor<i64>) -> tensor<2x1xf64> {
// CHECK-NEXT:   %c = stablehlo.constant dense<1> : tensor<i32>
// CHECK-NEXT:   %c_0 = stablehlo.constant dense<2> : tensor<i64>
// CHECK-NEXT:   %0 = stablehlo.transpose %arg0, dims = [2, 1, 0] : (tensor<9x8x7xf64>) -> tensor<7x8x9xf64>
// CHECK-NEXT:   %1 = stablehlo.add %arg1, %c_0 : tensor<i64>
// CHECK-NEXT:   %2 = stablehlo.convert %arg1 : (tensor<i64>) -> tensor<i32>
// CHECK-NEXT:   %3 = stablehlo.subtract %2, %c : tensor<i32>
// CHECK-NEXT:   %4 = stablehlo.convert %1 : (tensor<i64>) -> tensor<i32>
// CHECK-NEXT:   %5 = stablehlo.subtract %4, %c : tensor<i32>
// CHECK-NEXT:   %6 = stablehlo.add %3, %3 : tensor<i32>
// CHECK-NEXT:   %7 = stablehlo.dynamic_slice %0, %6, %2, %5, sizes = [1, 2, 1] : (tensor<7x8x9xf64>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x2x1xf64>
// CHECK-NEXT:   %8 = stablehlo.reshape %7 : (tensor<1x2x1xf64>) -> tensor<2x1xf64>
// CHECK-NEXT:   return %8 : tensor<2x1xf64>
// CHECK-NEXT: }
