// RUN: enzymexlamlir-opt %s --affine-cfg | FileCheck %s

// CHECK-LABEL: func.func private @implicit_mod
// CHECK: affine.parallel (%[[Z:.+]], %[[Y:.+]], %[[X:.+]]) = (0, 0, 0) to (10, 40, 40) {
// CHECK-NEXT: %[[V:.+]] = affine.load %{{.*}}[%[[Z]] + 5, %[[Y]] + 4, %[[X]] + 5] : memref<20x50x50xf64, 1>
// CHECK-NEXT: affine.store %[[V]], %{{.*}}[%[[Z]] + 5, %[[Y]] + 5, %[[X]] + 5] : memref<20x50x50xf64, 1>

// The single parallel induction variable %arg2 packs three tile coordinates:
//   z = %arg2 floordiv 9
//   y = (%arg2 mod 9) floordiv 3, appearing as (%arg2 floordiv 3) * 16 - (%arg2 floordiv 9) * 48
//   x = %arg2 mod 3, appearing as %arg2 * 16 - (%arg2 floordiv 3) * 48
// There is no explicit mod anywhere; the remainders are implicit linear
// combinations of raw and floordiv uses. SplitParallelInductions should
// still delinearize the IV.

module {
  func.func private @implicit_mod(%arg0: memref<20x50x50xf64, 1>, %arg1: memref<20x50x50xf64, 1>) {
    affine.parallel (%arg2, %arg3, %arg4) = (0, 0, 0) to (90, 16, 16) {
      affine.if affine_set<(d0, d1, d2) : (d2 + d1 * 16 - (d1 floordiv 3) * 48 >= 0, -d2 - d1 * 16 + (d1 floordiv 3) * 48 + 39 >= 0, d0 + (d1 floordiv 3) * 16 - (d1 floordiv 9) * 48 >= 0, -d0 - (d1 floordiv 3) * 16 + (d1 floordiv 9) * 48 + 39 >= 0)>(%arg3, %arg2, %arg4) {
        %0 = affine.load %arg0[%arg2 floordiv 9 + 5, %arg3 + (%arg2 floordiv 3) * 16 - (%arg2 floordiv 9) * 48 + 4, %arg2 * 16 + %arg4 - (%arg2 floordiv 3) * 48 + 5] : memref<20x50x50xf64, 1>
        affine.store %0, %arg1[%arg2 floordiv 9 + 5, %arg3 + (%arg2 floordiv 3) * 16 - (%arg2 floordiv 9) * 48 + 5, %arg2 * 16 + %arg4 - (%arg2 floordiv 3) * 48 + 5] : memref<20x50x50xf64, 1>
      }
    }
    return
  }
}
