// RUN: enzymexlamlir-opt --affine-cfg --split-input-file --allow-unregistered-dialect %s | FileCheck %s

func.func @ran(%arg0: memref<1x99x194xf64, 1>, %arg1: index) {
  %cst = arith.constant 2.731500e+02 : f64
  affine.parallel (%arg2, %arg3) = (0, 0) to (87, 192) {
    affine.if affine_set<(d0)[s0] : (d0 * 3 + s0 >= 0)>(%arg3)[%arg1] {
      affine.store %cst, %arg0[0, %arg2 + 6, %arg3 + 6] : memref<1x99x194xf64, 1>
    }
  }
  return
}

//CHECK: #set = affine_set<(d0)[s0] : (d0 * 3 + s0 >= 0)>
// CHECK-LABEL: @ran
// CHECK:    affine.parallel (%arg2, %arg3) = (0, 0) to (87, 192) {
// CHECK-NEXT:      affine.if #set(%arg3)[%arg1] {
// CHECK-NEXT:        affine.store %cst, %arg0[0, %arg2 + 6, %arg3 + 6] : memref<1x99x194xf64, 1>
// CHECK-NEXT:      }
// CHECK-NEXT:    }


func.func @foo(%arg0: memref<1x99x194xf64, 1>, %arg1: index) {
  %cst = arith.constant 2.731500e+02 : f64
  affine.parallel (%arg2, %arg3) = (0, 0) to (87, 192) {
    affine.if affine_set<(d0)[s0] : (d0 * 3 - 99 >= 0)>(%arg3)[%arg1] {
      affine.store %cst, %arg0[0, %arg2 + 6, %arg3 + 6] : memref<1x99x194xf64, 1>
    }
  }
  return
}

// CHECK-LABEL: @foo
// CHECK:    affine.parallel (%arg2, %arg3) = (33, 0) to (87, 192) {
// CHECK-NEXT:        affine.store %cst, %arg0[0, %arg2 + 6, %arg3 + 6] : memref<1x99x194xf64, 1>
// CHECK-NEXT:    }
