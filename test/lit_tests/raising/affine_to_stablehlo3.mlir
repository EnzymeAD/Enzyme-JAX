// RUN: enzymexlamlir-opt %s --affine-raise --enzyme-hlo-opt | FileCheck %s

module {
  func.func private @double_kernel(%arg0: memref<100x100xf32, 1>) {
    %cst = arith.constant 2.0 : f32
    affine.parallel (%arg1, %arg2) = (0, 0) to (100, 100) {
      %0 = affine.load %arg0[%arg1, %arg2] : memref<100x100xf32, 1>
      %1 = arith.mulf %0, %cst : f32
      affine.store %1, %arg0[%arg1, %arg2] : memref<100x100xf32, 1>
    }
    return
  }
}

// CHECK:  func.func private @double_kernel_raised(%[[ARG:.+]]: tensor<100x100xf32>) -> tensor<100x100xf32> {
// CHECK-NEXT:    %[[CST:.+]] = stablehlo.constant dense<2.000000e+00> : tensor<100x100xf32>
// CHECK-NEXT:    %[[RES:.+]] = arith.mulf %[[ARG]], %[[CST]] : tensor<100x100xf32>
// CHECK-NEXT:    return %[[RES]] : tensor<100x100xf32>
// CHECK-NEXT:  }
