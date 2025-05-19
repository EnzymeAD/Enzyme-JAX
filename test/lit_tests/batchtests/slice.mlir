// RUN: enzymexlamlir-opt %s --enzyme-batch | FileCheck %s

func.func private @slice(%arg0: tensor<3x4xf64>) -> (tensor<2x2xf64>) {
    %1 = stablehlo.slice %arg0 [0:2, 1:3] : (tensor<3x4xf64>) -> tensor<2x2xf64>
    return %1 : tensor<2x2xf64>
}
func.func @main(%arg0: tensor<2x5x3x4xf64>) -> (tensor<2x5x2x2xf64>) {
    %1 = enzyme.batch @slice(%arg0) {batch_shape = array<i64: 2, 5>} : (tensor<2x5x3x4xf64>) -> (tensor<2x5x2x2xf64>)
    return %1 : tensor<2x5x2x2xf64>
}

// CHECK: func.func private @batched_slice(%arg0: tensor<2x5x3x4xf64>) -> tensor<2x5x2x2xf64> {
// CHECK-NEXT:     %0 = stablehlo.slice %arg0 [0:2, 0:5, 0:2, 1:3] : (tensor<2x5x3x4xf64>) -> tensor<2x5x2x2xf64>
// CHECK-NEXT:     return %0 : tensor<2x5x2x2xf64>
// CHECK-NEXT: }
