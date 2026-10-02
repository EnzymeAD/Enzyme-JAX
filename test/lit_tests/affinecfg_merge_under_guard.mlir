// RUN: enzymexlamlir-opt --affine-cfg %s | FileCheck %s

// A guard over `%i + %j * 16` pins `%j` on one side (rows >= 1518 mean
// `%j` = 94), and simplifying the guarded load against the guard would
// rewrite it through `%i` alone. That single-induction form is what stops
// `%i` and `%j` from merging, so the load keeps both and the pair merges.

#set = affine_set<(d0, d1) : (-d0 - d1 * 16 + 1517 >= 0)>
module {
  func.func @merge_under_guard(%arg0: memref<1536x3072xf32, 1>, %arg1: f32, %arg2: f32) {
    %cst = arith.constant 0.000000e+00 : f32
    affine.parallel (%arg3, %arg4, %arg5) = (0, 0, 0) to (16, 95, 3056) {
      %0 = affine.if #set(%arg3, %arg4) -> f32 {
        %6 = affine.load %arg0[%arg3 + %arg4 * 16 + 8, %arg5 + 8] : memref<1536x3072xf32, 1>
        affine.yield %6 : f32
      } else {
        %6 = affine.load %arg0[%arg3 + %arg4 * 16 + 9, %arg5 + 8] : memref<1536x3072xf32, 1>
        affine.yield %6 : f32
      }
      %1 = affine.load %arg0[%arg3 + %arg4 * 16 + 7, %arg5 + 8] : memref<1536x3072xf32, 1>
      %2 = arith.addf %0, %1 : f32
      affine.store %2, %arg0[%arg3 + %arg4 * 16 + 8, %arg5 + 8] : memref<1536x3072xf32, 1>
    }
    return
  }
}

// CHECK:  #set = affine_set<(d0) : (-d0 + 1517 >= 0)>
// CHECK-NEXT:  module {
// CHECK-NEXT:    func.func @merge_under_guard(%arg0: memref<1536x3072xf32, 1>, %arg1: f32, %arg2: f32) {
// CHECK-NEXT:      affine.parallel (%arg3, %arg4) = (0, 0) to (1520, 3056) {
// CHECK-NEXT:        %0 = affine.if #set(%arg3) -> f32 {
// CHECK-NEXT:          %3 = affine.load %arg0[%arg3 + 8, %arg4 + 8] : memref<1536x3072xf32, 1>
// CHECK-NEXT:          affine.yield %3 : f32
// CHECK-NEXT:        } else {
// CHECK-NEXT:          %3 = affine.load %arg0[%arg3 + 9, %arg4 + 8] : memref<1536x3072xf32, 1>
// CHECK-NEXT:          affine.yield %3 : f32
// CHECK-NEXT:        }
// CHECK-NEXT:        %1 = affine.load %arg0[%arg3 + 7, %arg4 + 8] : memref<1536x3072xf32, 1>
// CHECK-NEXT:        %2 = arith.addf %0, %1 : f32
// CHECK-NEXT:        affine.store %2, %arg0[%arg3 + 8, %arg4 + 8] : memref<1536x3072xf32, 1>
// CHECK-NEXT:      }
// CHECK-NEXT:      return
// CHECK-NEXT:    }
// CHECK-NEXT:  }
