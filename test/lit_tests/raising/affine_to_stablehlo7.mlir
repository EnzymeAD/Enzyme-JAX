// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(raise-affine-to-stablehlo,arith-raise,enzyme-hlo-opt{max_constant_expansion=1})" | FileCheck %s

module {
  func.func @main(%arg0: memref<180x180xf32>, %arg1: memref<180x180xf32>) {
    %c90 = arith.constant 90 : index
    affine.parallel (%i, %j) = (0, 0) to (180, 180) {
      %1 = affine.load %arg0[%i, 0] : memref<180x180xf32>
      %2 = affine.load %arg1[%i, %j] : memref<180x180xf32>
      %3 = arith.addf %1, %2 : f32
      affine.store %3, %arg1[%j, %i] : memref<180x180xf32>
    }
    return
  }
}

// CHECK:  func.func private @main_raised(%arg0: tensor<180x180xf32>, %arg1: tensor<180x180xf32>) -> (tensor<180x180xf32>, tensor<180x180xf32>) {
// CHECK-NEXT:    %0 = stablehlo.slice %arg0 [0:180, 0:1] : (tensor<180x180xf32>) -> tensor<180x1xf32>
// CHECK-NEXT:    %1 = stablehlo.reshape %0 : (tensor<180x1xf32>) -> tensor<180xf32>
// CHECK-NEXT:    %2 = stablehlo.broadcast_in_dim %1, dims = [0] {enzymexla.guaranteed_symmetric = [#enzymexla<guaranteed NOTGUARANTEED>]} : (tensor<180xf32>) -> tensor<180x180xf32>
// CHECK-NEXT:    %3 = stablehlo.add %2, %arg1 {enzymexla.guaranteed_symmetric = [#enzymexla<guaranteed NOTGUARANTEED>]} : tensor<180x180xf32>
// CHECK-NEXT:    %4 = stablehlo.transpose %3, dims = [1, 0] : (tensor<180x180xf32>) -> tensor<180x180xf32>
// CHECK-NEXT:    return %arg0, %4 : tensor<180x180xf32>, tensor<180x180xf32>
// CHECK-NEXT:  }
