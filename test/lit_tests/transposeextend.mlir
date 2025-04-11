// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=transpose_extend" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s

module {
  func.func @transpose_extend_test(%arg0: tensor<3x4x5xf32>) -> (tensor<4x3x7xf32>, tensor<6x3x7xf32>) {
    // Create an extend operation and then transpose it
    %0 = "enzymexla.extend"(%arg0) <{dimension = 2 : i64, lhs = 1 : i64, rhs = 1 : i64}> : (tensor<3x4x5xf32>) -> tensor<3x4x7xf32>
    %1 = stablehlo.transpose %0, dims = [1, 0, 2] : (tensor<3x4x7xf32>) -> tensor<4x3x7xf32>
    
    // Add a second example with different dimensions to fully test permutation handling
    %2 = "enzymexla.extend"(%1) <{dimension = 0 : i64, lhs = 1 : i64, rhs = 1 : i64}> : (tensor<4x3x7xf32>) -> tensor<6x3x7xf32>
    %3 = stablehlo.transpose %2, dims = [0, 1, 2] : (tensor<6x3x7xf32>) -> tensor<6x3x7xf32>
    return %1, %3 : tensor<4x3x7xf32>, tensor<6x3x7xf32>
  }
}

// CHECK:      func.func @transpose_extend_test(%arg0: tensor<3x4x5xf32>) -> (tensor<4x3x7xf32>, tensor<6x3x7xf32>) {
// CHECK-NEXT:   %0 = stablehlo.transpose %arg0, dims = [1, 0, 2] : (tensor<3x4x5xf32>) -> tensor<4x3x5xf32>
// CHECK-NEXT:   %1 = "enzymexla.extend"(%0) <{dimension = 2 : i64, lhs = 1 : i64, rhs = 1 : i64}> : (tensor<4x3x5xf32>) -> tensor<4x3x7xf32>
// CHECK-NEXT:   %2 = stablehlo.transpose %0, dims = [0, 1, 2] : (tensor<4x3x5xf32>) -> tensor<4x3x5xf32>
// CHECK-NEXT:   %3 = "enzymexla.extend"(%2) <{dimension = 2 : i64, lhs = 1 : i64, rhs = 1 : i64}> : (tensor<4x3x5xf32>) -> tensor<4x3x7xf32>
// CHECK-NEXT:   %4 = "enzymexla.extend"(%3) <{dimension = 0 : i64, lhs = 1 : i64, rhs = 1 : i64}> : (tensor<4x3x7xf32>) -> tensor<6x3x7xf32>
// CHECK-NEXT:   return %1, %4 : tensor<4x3x7xf32>, tensor<6x3x7xf32>
// CHECK-NEXT: }
