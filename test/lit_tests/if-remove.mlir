// RUN: enzymexlamlir-opt %s --split-input-file -allow-unregistered-dialect --affine-cfg | FileCheck %s


// CHECK-LABEL: only_then
// CHECK-NOT: affine.if
// CHECK: test1
// CHECK: test2

#set = affine_set<(d0, d1, d2, d3) : (-d0 + 89 >= 0)>
func.func private @"only_then"(%arg0: memref<34x99x194xf64, 1>, %arg1: memref<34x99x194xf64, 1>, %arg2: memref<34x99x194xf64, 1>, %arg3: memref<34x99x194xf64, 1>, %arg4: memref<34x99x194xf64, 1>) {
  affine.parallel (%arg5, %arg6, %arg7, %arg8) = (0, 0, 0, 0) to (2, 16, 12, 16) {
    "test.test1"(): () -> ()
    affine.if #set(%arg6, %arg5, %arg7, %arg8) {
      "test.test2"(): () -> ()
      affine.yield
    }
    affine.yield
  }
  func.return
}

// -----

// CHECK-LABEL: only_then
// CHECK-NOT: affine.if
// CHECK: test1
// CHECK-NOT: test2

#set = affine_set<(d0, d1, d2, d3) : (-d0 - 1 >= 0)>
func.func private @"only_then"(%arg0: memref<34x99x194xf64, 1>, %arg1: memref<34x99x194xf64, 1>, %arg2: memref<34x99x194xf64, 1>, %arg3: memref<34x99x194xf64, 1>, %arg4: memref<34x99x194xf64, 1>) {
  affine.parallel (%arg5, %arg6, %arg7, %arg8) = (0, 0, 0, 0) to (2, 16, 12, 16) {
    "test.test1"(): () -> ()
    affine.if #set(%arg6, %arg5, %arg7, %arg8) {
      "test.test2"(): () -> ()
      affine.yield
    }
    affine.yield
  }
  func.return
}

