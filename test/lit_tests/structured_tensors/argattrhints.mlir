// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

func.func @main(%arg0: tensor<8x8xf32> {enzymexla.symmetric_matrix = #enzymexla<guaranteed GUARANTEED>}) -> (tensor<8x8xf32>) {
    %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<8x8xf32>) -> tensor<8x8xf32>
    return %0 : tensor<8x8xf32>
}

// CHECK: func.func @main(%arg0: tensor<8x8xf32> {enzymexla.symmetric_matrix = #enzymexla<guaranteed GUARANTEED>}) -> tensor<8x8xf32> {
// CHECK-NEXT:   return %arg0 : tensor<8x8xf32>
// CHECK-NEXT: }
