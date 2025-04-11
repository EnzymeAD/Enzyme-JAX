// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=transpose_wrap" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s

module {
  func.func @transpose_wrap_test(%arg0: tensor<3x4x5xf32>) -> tensor<9x3x4xf32> {
    // Create a wrap operation and then transpose it
    %0 = "enzymexla.wrap"(%arg0) <{dimension = 2 : i64, lhs = 3 : i64, rhs = 1 : i64}> : (tensor<3x4x5xf32>) -> tensor<3x4x9xf32>
    %1 = stablehlo.transpose %0, dims = [2, 0, 1] : (tensor<3x4x9xf32>) -> tensor<9x3x4xf32>
    return %1 : tensor<9x3x4xf32>
  }
}

// CHECK:      func.func @transpose_wrap_test(%arg0: tensor<3x4x5xf32>) -> tensor<9x3x4xf32> {
// CHECK-NEXT:   %0 = stablehlo.transpose %arg0, dims = [2, 0, 1] : (tensor<3x4x5xf32>) -> tensor<5x3x4xf32>
// CHECK-NEXT:   %1 = "enzymexla.wrap"(%0) <{dimension = 0 : i64, lhs = 3 : i64, rhs = 1 : i64}> : (tensor<5x3x4xf32>) -> tensor<9x3x4xf32>
// CHECK-NEXT:   return %1 : tensor<9x3x4xf32>
// CHECK-NEXT: }