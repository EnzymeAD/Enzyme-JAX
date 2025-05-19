// RUN: enzymexlamlir-opt %s --enzyme-batch | FileCheck %s

func.func private @slice(%arg0: tensor<3x4xf64>) -> (tensor<2x2xf64>) {
    %1 = stablehlo.slice %arg0 [0:2, 1:3] : (tensor<3x4xf64>) -> tensor<2x2xf64>
    return %1 : tensor<2x2xf64>
}

func.func private @dynamic_slice(%arg0: tensor<3x4xf64>) -> (tensor<2x2xf64>) {
    %c_0 = stablehlo.constant dense<0> : tensor<i32>
    %c_1 = stablehlo.constant dense<1> : tensor<i32>
    %1 = stablehlo.dynamic_slice %arg0, %c_0, %c_1, sizes = [2, 2] : (tensor<3x4xf64>, tensor<i32>, tensor<i32>) -> tensor<2x2xf64>
    return %1 : tensor<2x2xf64>
}

func.func private @dynamic_slice2(%arg0: tensor<3x4xf64>) -> (tensor<2x2xf64>) {
    %c_0 = stablehlo.constant dense<0> : tensor<i64>
    %c_1 = stablehlo.constant dense<1> : tensor<i64>
    %1 = stablehlo.dynamic_slice %arg0, %c_0, %c_1, sizes = [2, 2] : (tensor<3x4xf64>, tensor<i64>, tensor<i64>) -> tensor<2x2xf64>
    return %1 : tensor<2x2xf64>
}

func.func @main(%arg0: tensor<2x5x3x4xf64>) -> (tensor<2x5x2x2xf64>, tensor<2x5x2x2xf64>, tensor<2x5x2x2xf64>) {
    %1 = enzyme.batch @slice(%arg0) {batch_shape = array<i64: 2, 5>} : (tensor<2x5x3x4xf64>) -> (tensor<2x5x2x2xf64>)
    %2 = enzyme.batch @dynamic_slice(%arg0) {batch_shape = array<i64: 2, 5>} : (tensor<2x5x3x4xf64>) -> (tensor<2x5x2x2xf64>)
    %3 = enzyme.batch @dynamic_slice2(%arg0) {batch_shape = array<i64: 2, 5>} : (tensor<2x5x3x4xf64>) -> (tensor<2x5x2x2xf64>)
    return %1, %2, %3 : tensor<2x5x2x2xf64>, tensor<2x5x2x2xf64>, tensor<2x5x2x2xf64>
}

// CHECK: func.func private @batched_slice(%arg0: tensor<2x5x3x4xf64>) -> tensor<2x5x2x2xf64> {
// CHECK-NEXT:     %0 = stablehlo.slice %arg0 [0:2, 0:5, 0:2, 1:3] : (tensor<2x5x3x4xf64>) -> tensor<2x5x2x2xf64>
// CHECK-NEXT:     return %0 : tensor<2x5x2x2xf64>
// CHECK-NEXT: }
// CHECK:  func.func private @batched_dynamic_slice(%arg0: tensor<2x5x3x4xf64>) -> tensor<2x5x2x2xf64> {
// CHECK-NEXT:    %c = stablehlo.constant dense<0> : tensor<2x5xi32>
// CHECK-NEXT:    %c_0 = stablehlo.constant dense<1> : tensor<2x5xi32>
// CHECK-NEXT:    %c_1 = stablehlo.constant dense<0> : tensor<i32>
// CHECK-NEXT:    %0 = stablehlo.slice %c [0:1, 0:1] : (tensor<2x5xi32>) -> tensor<1x1xi32>
// CHECK-NEXT:    %1 = stablehlo.reshape %0 : (tensor<1x1xi32>) -> tensor<i32>
// CHECK-NEXT:    %2 = stablehlo.slice %c_0 [0:1, 0:1] : (tensor<2x5xi32>) -> tensor<1x1xi32>
// CHECK-NEXT:    %3 = stablehlo.reshape %2 : (tensor<1x1xi32>) -> tensor<i32>
// CHECK-NEXT:    %4 = stablehlo.dynamic_slice %arg0, %c_1, %c_1, %1, %3, sizes = [2, 5, 2, 2] : (tensor<2x5x3x4xf64>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<2x5x2x2xf64>
// CHECK-NEXT:    return %4 : tensor<2x5x2x2xf64>
// CHECK-NEXT:  }
// CHECK:  func.func private @batched_dynamic_slice2(%arg0: tensor<2x5x3x4xf64>) -> tensor<2x5x2x2xf64> {
// CHECK-NEXT:    %c = stablehlo.constant dense<0> : tensor<2x5xi64>
// CHECK-NEXT:    %c_0 = stablehlo.constant dense<1> : tensor<2x5xi64>
// CHECK-NEXT:    %c_1 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %0 = stablehlo.slice %c [0:1, 0:1] : (tensor<2x5xi64>) -> tensor<1x1xi64>
// CHECK-NEXT:    %1 = stablehlo.reshape %0 : (tensor<1x1xi64>) -> tensor<i64>
// CHECK-NEXT:    %2 = stablehlo.slice %c_0 [0:1, 0:1] : (tensor<2x5xi64>) -> tensor<1x1xi64>
// CHECK-NEXT:    %3 = stablehlo.reshape %2 : (tensor<1x1xi64>) -> tensor<i64>
// CHECK-NEXT:    %4 = stablehlo.dynamic_slice %arg0, %c_1, %c_1, %1, %3, sizes = [2, 5, 2, 2] : (tensor<2x5x3x4xf64>, tensor<i64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<2x5x2x2xf64>
// CHECK-NEXT:    return %4 : tensor<2x5x2x2xf64>
// CHECK-NEXT:  }
