// RUN: enzymexlamlir-opt %s --raise-affine-to-stablehlo --arith-raise --enzyme-hlo-opt | FileCheck %s

module {
  func.func private @kernel(%arg0: memref<187x194xf64, 1>, %arg1: memref<187x194xf64, 1>) {
    %cst = arith.constant -1.000000e+00 : f64
    affine.parallel (%arg2, %arg3) = (0, 0) to (194, 187) {
      %0 = affine.load %arg0[-%arg3, -%arg2] : memref<187x194xf64, 1>
      affine.store %0, %arg1[%arg3, %arg2] : memref<187x194xf64, 1>
    }
    return
  }
}

// CHECK:  func.func private @kernel_raised(%[[src:.+]]: tensor<187x194xf64>, %[[dst:.+]]: tensor<187x194xf64>) -> (tensor<187x194xf64>, tensor<187x194xf64>) {
// CHECK-NEXT:    %[[rev:.+]] = stablehlo.reverse %[[src]], dims = [0, 1] : tensor<187x194xf64>
// CHECK-NEXT:    return %[[src]], %[[rev]] : tensor<187x194xf64>, tensor<187x194xf64>
// CHECK-NEXT:  }
