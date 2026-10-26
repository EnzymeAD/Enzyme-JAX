// RUN:enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=transpose_reduce" --transform-interpreter --enzyme-hlo-remove-transform | FileCheck %s

module {
  func.func @main(%11: tensor<1x12x8x3x8xf32>) -> tensor<1x8x3xf32> {
    %cst_1 = stablehlo.constant dense<0.0> : tensor<f32>
    %16 = stablehlo.reduce(%11 init: %cst_1) applies stablehlo.add across dimensions = [2, 1]
            : (tensor<1x12x8x3x8xf32>, tensor<f32>) -> tensor<1x3x8xf32>
    %17 = stablehlo.transpose %16, dims = [0, 2, 1] : (tensor<1x3x8xf32>) -> tensor<1x8x3xf32>
    return %17 : tensor<1x8x3xf32>
  }
}

// CHECK:  func.func @main(%[[ARG:.+]]: tensor<1x12x8x3x8xf32>) -> tensor<1x8x3xf32> {
// CHECK-NEXT:    %[[ZERO:.+]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:    %[[TRAN:.+]] = stablehlo.transpose %[[ARG]], dims = [0, 1, 2, 4, 3] : (tensor<1x12x8x3x8xf32>) -> tensor<1x12x8x8x3xf32>
// CHECK-NEXT:    %[[REDU:.+]] = stablehlo.reduce(%[[TRAN]] init: %[[ZERO]]) applies stablehlo.add across dimensions = [1, 2] : (tensor<1x12x8x8x3xf32>, tensor<f32>) -> tensor<1x8x3xf32>
// CHECK-NEXT:    return %[[REDU]] : tensor<1x8x3xf32>
// CHECK-NEXT:  }
