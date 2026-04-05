// RUN: enzymexlamlir-opt %s -enzyme-hlo-opt | FileCheck %s

func.func @main(%iterArg_200: tensor<42x266x266xf32>) -> (tensor<32x256x256xf32>, tensor<32x256x256xf32>) {
      %611 = stablehlo.slice %iterArg_200 [5:37, 5:261, 5:261] {enzymexla.finite = [#enzymexla<guaranteed NOTGUARANTEED>]} : (tensor<42x266x266xf32>) -> tensor<32x256x256xf32>
      %612 = stablehlo.slice %iterArg_200 [5:37, 5:261, 5:261] : (tensor<42x266x266xf32>) -> tensor<32x256x256xf32>
      return %611, %612 : tensor<32x256x256xf32>, tensor<32x256x256xf32>
}

// CHECK: func.func @main(%arg0: tensor<42x266x266xf32>) -> (tensor<32x256x256xf32>, tensor<32x256x256xf32>) {
// CHECK-NEXT:   %0 = stablehlo.slice %arg0 [5:37, 5:261, 5:261] {enzymexla.finite = [#enzymexla<guaranteed NOTGUARANTEED>]} : (tensor<42x266x266xf32>) -> tensor<32x256x256xf32>
// CHECK-NEXT:   return %0, %0 : tensor<32x256x256xf32>, tensor<32x256x256xf32>
// CHECK-NEXT: }
