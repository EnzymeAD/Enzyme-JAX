// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=reshape_rotate" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s

module {
  func.func @main(%arg0: tensor<1x10x80xf64>) -> tensor<10x80x1x1xf64> {
    %0 = "enzymexla.rotate"(%arg0) <{dimension = 1 : i32, amount = 9 : i32}> : (tensor<1x10x80xf64>) -> tensor<1x10x80xf64>
    %1 = stablehlo.reshape %0 : (tensor<1x10x80xf64>) -> tensor<10x80x1x1xf64>
    return %1 : tensor<10x80x1x1xf64>
  }
}

// CHECK:      func.func @main(%arg0: tensor<1x10x80xf64>) -> tensor<10x80x1x1xf64> {
// CHECK-NEXT:   %0 = stablehlo.reshape %arg0 : (tensor<1x10x80xf64>) -> tensor<10x80x1x1xf64>
// CHECK-NEXT:   %1 = "enzymexla.rotate"(%0) <{amount = 9 : i32, dimension = 0 : i32}> : (tensor<10x80x1x1xf64>) -> tensor<10x80x1x1xf64>
// CHECK-NEXT:   return %1 : tensor<10x80x1x1xf64>
// CHECK-NEXT: }
