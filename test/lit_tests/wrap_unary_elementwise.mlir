// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td='patterns=wrap_unary_elementwise(1)' --transform-interpreter --enzyme-hlo-remove-transform | FileCheck %s

module {
  func.func @main(%arg0: tensor<10xf32>) -> tensor<14xf32> {
    %0 = "enzymexla.wrap"(%arg0) <{ lhs = 2 : i64, rhs = 2 : i64, dimension = 0 : i64 }> : (tensor<10xf32>) -> tensor<14xf32>
    %1 = "math.sqrt"(%0) : (tensor<14xf32>) -> tensor<14xf32>
    return %1 : tensor<14xf32>
  }
}

// CHECK:  func.func @main(%[[ARG:.+]]: tensor<10xf32>) -> tensor<14xf32> {
// CHECK-NEXT:    %[[SQRT:.+]] = math.sqrt %[[ARG]] : tensor<10xf32>
// CHECK-NEXT:    %[[WRAPPED:.+]] = "enzymexla.wrap"(%[[SQRT]]) <{dimension = 0 : i64, lhs = 2 : i64, rhs = 2 : i64}> : (tensor<10xf32>) -> tensor<14xf32>
// CHECK-NEXT:    return %[[WRAPPED]] : tensor<14xf32>
// CHECK-NEXT:  }
