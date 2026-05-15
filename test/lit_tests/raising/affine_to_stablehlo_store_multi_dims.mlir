// RUN: enzymexlamlir-opt %s --raise-affine-to-stablehlo --canonicalize --enzyme-hlo-opt=max_constant_expansion=0 | FileCheck %s

module {
  func.func @main(%arg0: memref<100xf32>, %arg1: memref<100xf32>) {
    affine.parallel (%i, %j) = (0, 0) to (10, 10) step (1, 1) {
      %0 = affine.load %arg1[%i * 10 + %j] : memref<100xf32>
      affine.store %0, %arg0[%i * 10 + %j] : memref<100xf32>
    }
    return
  }
}

// CHECK:  func.func private @main_raised(%arg0: tensor<100xf32>, %arg1: tensor<100xf32>) -> (tensor<100xf32>, tensor<100xf32>) {
// CHECK-NEXT:    return %arg1, %arg1 : tensor<100xf32>, tensor<100xf32>
// CHECK-NEXT:  }
