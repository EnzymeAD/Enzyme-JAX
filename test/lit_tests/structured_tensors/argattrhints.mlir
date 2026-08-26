// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

func.func @main1(%arg0: tensor<8x8xf32> {enzymexla.symmetric_matrix = #enzymexla<guaranteed GUARANTEED>}) -> (tensor<8x8xf32>) {
    %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<8x8xf32>) -> tensor<8x8xf32>
    return %0 : tensor<8x8xf32>
}

// CHECK: func.func @main1(%arg0: tensor<8x8xf32> {enzymexla.symmetric_matrix = #enzymexla<guaranteed GUARANTEED>}) -> tensor<8x8xf32> {
// CHECK-NEXT:   return %arg0 : tensor<8x8xf32>
// CHECK-NEXT: }

func.func @main2(%arg0: tensor<2x2xf32> {enzymexla.symmetric_matrix = #enzymexla<guaranteed GUARANTEED>}, %arg1: tensor<2x2xf32> {enzymexla.symmetric_matrix = #enzymexla<guaranteed GUARANTEED>}) -> tensor<2x2xf32> {
  %0 = stablehlo.add %arg0, %arg1 : tensor<2x2xf32>
  %1 = stablehlo.transpose %0, dims = [1, 0] : (tensor<2x2xf32>) -> tensor<2x2xf32>
  return %1 : tensor<2x2xf32>
}

// CHECK: func.func @main2(%arg0: tensor<2x2xf32> {enzymexla.symmetric_matrix = #enzymexla<guaranteed GUARANTEED>}, %arg1: tensor<2x2xf32> {enzymexla.symmetric_matrix = #enzymexla<guaranteed GUARANTEED>}) -> tensor<2x2xf32> {
// CHECK-NEXT:   %0 = stablehlo.add %arg0, %arg1 {enzymexla.symmetric_matrix = [#enzymexla<guaranteed GUARANTEED>]} : tensor<2x2xf32>
// CHECK-NEXT:   return %0 : tensor<2x2xf32>
// CHECK-NEXT: }
