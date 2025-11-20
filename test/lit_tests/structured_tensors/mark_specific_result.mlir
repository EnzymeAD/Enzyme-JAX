// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

module {
  func.func @main(%arg0: tensor<8xf32>) -> (tensor<8x8xf32>, tensor<1xf32>, tensor<1x1xf32>) {
    %0 = stablehlo.reshape %arg0 : (tensor<8xf32>) -> tensor<8x1xf32>
    %U, %S, %Vt, %info = enzymexla.linalg.svd %0 {full = true} : (tensor<8x1xf32>) -> (tensor<8x8xf32>, tensor<1xf32>, tensor<1x1xf32>, tensor<i32>)
    %1 = stablehlo.transpose %U, dims = [1, 0] : (tensor<8x8xf32>) -> tensor<8x8xf32>
    return %1, %S, %Vt : tensor<8x8xf32>, tensor<1xf32>, tensor<1x1xf32>
  }
}

// CHECK: func.func @main(%arg0: tensor<8xf32>) -> (tensor<8x8xf32>, tensor<1xf32>, tensor<1x1xf32>) {
// CHECK-NEXT:   %0 = stablehlo.reshape %arg0 : (tensor<8xf32>) -> tensor<8x1xf32>
// CHECK-NEXT:   %U, %S, %Vt, %info = enzymexla.linalg.svd %0 {enzymexla.guaranteed_symmetric = [#enzymexla<guaranteed NOTGUARANTEED>, #enzymexla<guaranteed UNKNOWN>, #enzymexla<guaranteed UNKNOWN>, #enzymexla<guaranteed UNKNOWN>], full = true} : (tensor<8x1xf32>) -> (tensor<8x8xf32>, tensor<1xf32>, tensor<1x1xf32>, tensor<i32>)
// CHECK-NEXT:   %1 = stablehlo.transpose %U, dims = [1, 0] : (tensor<8x8xf32>) -> tensor<8x8xf32>
// CHECK-NEXT:   return %1, %S, %Vt : tensor<8x8xf32>, tensor<1xf32>, tensor<1x1xf32>
// CHECK-NEXT: }
