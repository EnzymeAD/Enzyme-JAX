// RUN: enzymexlamlir-opt %s --enzyme-batch | FileCheck %s

func.func private @concat(%arg0: tensor<3x4xf64>) -> (tensor<3x8xf64>) {
    %1 = stablehlo.concatenate %arg0, %arg0, dim = 1 : (tensor<3x4xf64>, tensor<3x4xf64>) -> tensor<3x8xf64>
    return %1 : tensor<3x8xf64>
}
func.func @main(%arg0: tensor<2x5x3x4xf64>) -> (tensor<2x5x3x8xf64>) {
    %1 = enzyme.batch @concat(%arg0) {batch_shape = array<i64: 2, 5>} : (tensor<2x5x3x4xf64>) -> (tensor<2x5x3x8xf64>)
    return %1 : tensor<2x5x3x8xf64>
}

// CHECK:  func.func private @batched_concat(%arg0: tensor<2x5x3x4xf64>) -> tensor<2x5x3x8xf64> {
// CHECK-NEXT:    %0 = stablehlo.concatenate %arg0, %arg0, dim = 3 : (tensor<2x5x3x4xf64>, tensor<2x5x3x4xf64>) -> tensor<2x5x3x8xf64>
// CHECK-NEXT:    return %0 : tensor<2x5x3x8xf64>
// CHECK-NEXT:  }
