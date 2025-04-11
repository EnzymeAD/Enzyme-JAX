// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td=patterns=extend_elementwise --transform-interpreter --enzyme-hlo-remove-transform | FileCheck %s

module {
  func.func @main(%arg0: tensor<10xf32>, %arg1: tensor<10xf32>) -> tensor<14xf32> {
    %lhs = "enzymexla.extend"(%arg0) <{dimension = 0 : i64, lhs = 2 : i64, rhs = 2 : i64}> : (tensor<10xf32>) -> tensor<14xf32>
    %rhs = "enzymexla.extend"(%arg1) <{dimension = 0 : i64, lhs = 2 : i64, rhs = 2 : i64}> : (tensor<10xf32>) -> tensor<14xf32>

    %res = stablehlo.add %lhs, %rhs : tensor<14xf32>
    return %res : tensor<14xf32>
  }
}

// CHECK:  func.func @main(%[[ARG0:.+]]: tensor<10xf32>, %[[ARG1:.+]]: tensor<10xf32>) -> tensor<14xf32> {
// CHECK-NEXT:    %[[ELEM:.+]] = stablehlo.add %[[ARG0]], %[[ARG1]] : tensor<10xf32>
// CHECK-NEXT:    %[[EXTEND:.+]] = "enzymexla.extend"(%[[ELEM]]) <{dimension = 0 : i64, lhs = 2 : i64, rhs = 2 : i64}> : (tensor<10xf32>) -> tensor<14xf32>
// CHECK-NEXT:    return %[[EXTEND]] : tensor<14xf32>
// CHECK-NEXT:  }
