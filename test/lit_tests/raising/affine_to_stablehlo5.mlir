// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(raise-affine-to-stablehlo,arith-raise,enzyme-hlo-opt{max_constant_expansion=1})" | FileCheck %s

module {
  func.func @main(%arg0: memref<180xf32>, %arg1: memref<180xf32>, %arg2: memref<180xf32>) {
    %c90 = arith.constant 90 : index
    affine.parallel (%i) = (0) to (180) {
      %1 = affine.load %arg0[%i] : memref<180xf32>
      %2 = affine.load %arg1[%i] : memref<180xf32>
      %cond = arith.cmpi sgt, %i, %c90 : index
      %3 = arith.select %cond, %1, %2 : f32
      affine.store %3, %arg2[%i] : memref<180xf32>
    }
    return
  }
}

// CHECK:  func.func private @main_raised(%arg0: tensor<180xf32>, %arg1: tensor<180xf32>, %arg2: tensor<180xf32>) -> (tensor<180xf32>, tensor<180xf32>, tensor<180xf32>) {
// CHECK-NEXT:    %0 = stablehlo.slice %arg1 [0:91] : (tensor<180xf32>) -> tensor<91xf32> 
// CHECK-NEXT:    %1 = stablehlo.slice %arg0 [91:180] : (tensor<180xf32>) -> tensor<89xf32>
// CHECK-NEXT:    %2 = stablehlo.concatenate %0, %1, dim = 0 : (tensor<91xf32>, tensor<89xf32>) -> tensor<180xf32> 
// CHECK-NEXT:    return %arg0, %arg1, %2 : tensor<180xf32>, tensor<180xf32>, tensor<180xf32> 
// CHECK-NEXT:  }
