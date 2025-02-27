// RUN: enzymexlamlir-opt --affine-cfg --split-input-file --allow-unregistered-dialect %s | FileCheck %s

#set = affine_set<(d0, d1, d2, d3) : (-d0 - d1 * 16 + 86 >= 0, d2 * 16 + d3 >= 0, d2 * -16 - d3 + 181 >= 0)>
module {
  func.func @ran(%arg0: memref<1x99x194xf64, 1>) {
    %cst = arith.constant 2.731500e+02 : f64
    affine.parallel (%arg15, %arg16, %arg17, %arg18) = (0, 0, 0, 0) to (6, 16, 12, 16) {
      affine.if #set(%arg16, %arg15, %arg17, %arg18) {
        affine.store %cst, %arg0[0, %arg16 + %arg15 * 16 + 6, %arg18 + %arg17 * 16 + 6] : memref<1x99x194xf64, 1>
      }
    }
    return
  }
}

// CHECK:  func.func @ran(%arg0: memref<1x99x194xf64, 1>) {
// CHECK-NEXT:    %cst = arith.constant 2.731500e+02 : f64
// CHECK-NEXT:    affine.parallel (%arg1, %arg2) = (0, 0) to (87, 192) {
// CHECK-NEXT:        affine.store %cst, %arg0[0, %arg1 + 6, %arg2 + 6] : memref<1x99x194xf64, 1>
// CHECK-NEXT:    }
// CHECK-NEXT:    return
// CHECK-NEXT:  }
