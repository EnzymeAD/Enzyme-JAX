// RUN: enzymexlamlir-opt --affine-cfg --split-input-file --allow-unregistered-dialect %s | FileCheck %s

  func.func @f(%0: f64, %arg0: memref<32x128x128xf64, 1>) {
    affine.parallel (%arg1) = (0) to (16) {
      affine.store %0, %arg0[(%arg1 floordiv 8) * 16, (1 + %arg1 * 16) mod 128, 0] : memref<32x128x128xf64, 1>
    }
    return
  }

// CHECK:  func.func @f(%arg0: f64, %arg1: memref<32x128x128xf64, 1>) {
// CHECK-NEXT:    affine.parallel (%arg2, %arg3) = (0, 0) to (2, 8) {
// CHECK-NEXT:      affine.store %arg0, %arg1[%arg2 * 16, %arg3 * 16 + 1, 0] : memref<32x128x128xf64, 1>
// CHECK-NEXT:    }
// CHECK-NEXT:    return
// CHECK-NEXT:  }

