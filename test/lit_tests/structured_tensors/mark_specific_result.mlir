// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

module {
  func.func @main(%arg0: tensor<8x8xf32>) -> (tensor<8x8xf32>, tensor<8xf32>, tensor<8x8xf32>) {
    %U, %S, %Vt, %info = enzymexla.linalg.svd %arg0 {full = true} : (tensor<8x8xf32>) -> (tensor<8x8xf32>, tensor<8xf32>, tensor<8x8xf32>, tensor<i32>)
    %0 = stablehlo.transpose %Vt, dims = [1, 0] : (tensor<8x8xf32>) -> tensor<8x8xf32>
    return %U, %S, %0 : tensor<8x8xf32>, tensor<8xf32>, tensor<8x8xf32>
  }
}

// CHECK: func.func @main(%arg0: tensor<8x8xf32>) -> (tensor<8x8xf32>, tensor<8xf32>, tensor<8x8xf32>) {
// CHECK-NEXT:    %U, %S, %Vt, %info = enzymexla.linalg.svd %arg0 {enzymexla.symmetric_matrix = [#enzymexla<guaranteed UNKNOWN>, #enzymexla<guaranteed UNKNOWN>, #enzymexla<guaranteed NOTGUARANTEED>, #enzymexla<guaranteed UNKNOWN>], full = true} : (tensor<8x8xf32>) -> (tensor<8x8xf32>, tensor<8xf32>, tensor<8x8xf32>, tensor<i32>)
// CHECK-NEXT:    %0 = stablehlo.transpose %Vt, dims = [1, 0] : (tensor<8x8xf32>) -> tensor<8x8xf32>
// CHECK-NEXT:    return %U, %S, %0 : tensor<8x8xf32>, tensor<8xf32>, tensor<8x8xf32>
// CHECK-NEXT:  }
