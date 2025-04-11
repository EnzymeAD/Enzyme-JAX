// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=reshape_extend" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s

module {
  func.func @main(%arg0: tensor<1x10x80xf64>) -> tensor<10x96x1x1xf64> {
    %0 = "enzymexla.extend"(%arg0) <{dimension = 2 : i64, lhs = 7 : i64, rhs = 9 : i64}> : (tensor<1x10x80xf64>) -> tensor<1x10x96xf64>
    %1 = stablehlo.reshape %0 : (tensor<1x10x96xf64>) -> tensor<10x96x1x1xf64>
    return %1 : tensor<10x96x1x1xf64>
  }
}

// CHECK:      func.func @main(%arg0: tensor<1x10x80xf64>) -> tensor<10x96x1x1xf64> {
// CHECK-NEXT:   %0 = stablehlo.reshape %arg0 : (tensor<1x10x80xf64>) -> tensor<10x80x1x1xf64>
// CHECK-NEXT:   %1 = "enzymexla.extend"(%0) <{dimension = 1 : i64, lhs = 7 : i64, rhs = 9 : i64}> : (tensor<10x80x1x1xf64>) -> tensor<10x96x1x1xf64>
// CHECK-NEXT:   return %1 : tensor<10x96x1x1xf64>
// CHECK-NEXT: }
