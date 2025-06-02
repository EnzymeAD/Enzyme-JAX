// RUN: enzymexlamlir-opt --affine-cfg --split-input-file --allow-unregistered-dialect %s | FileCheck %s

#set = affine_set<(d0, d1) : (-d0 - d1 * 16 + 86 >= 0)>
module {
  func.func @ran(%arg3: memref<1x99xf64, 1>) {
    %cst = arith.constant 2.731500e+02 : f64
    affine.parallel (%arg15, %arg16) = (0, 0) to (6, 16) {
      affine.if #set(%arg16, %arg15) {
        affine.store %cst, %arg3[0, %arg16 + %arg15 * 16] : memref<1x99xf64, 1>
      }
    }
    return
  }
}

// CHECK:  func.func @ran(%arg0: memref<1x99xf64, 1>) {
// CHECK-NEXT:    %cst = arith.constant 2.731500e+02 : f64
// CHECK-NEXT:    affine.parallel (%arg1) = (0) to (87) {
// CHECK-NEXT:      affine.store %cst, %arg0[0, %arg1] : memref<1x99xf64, 1>
// CHECK-NEXT:    }
// CHECK-NEXT:    return
// CHECK-NEXT:  }
