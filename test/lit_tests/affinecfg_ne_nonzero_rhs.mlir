// RUN: enzymexlamlir-opt --affine-cfg %s | FileCheck %s

module {
  func.func @ne_nonzero_rhs(%arg0: memref<1x16x18xf32, 1>, %arg1: memref<1x16x18xf32, 1>) {
    %c4 = arith.constant 4 : index
    %cst = arith.constant 0.000000e+00 : f32
    affine.parallel (%i) = (0) to (256) {
      %0 = arith.index_castui %i : index to i64
      %1 = arith.shrui %i, %c4 : index
      %2 = arith.shli %1, %c4 : index
      %3 = arith.index_castui %2 : index to i64
      %4 = affine.load %arg0[0, %i floordiv 16 + 4, %i mod 16 + 4] : memref<1x16x18xf32, 1>
      %5 = arith.cmpi ne, %3, %0 : i64
      %6 = arith.select %5, %4, %cst : f32
      affine.store %6, %arg1[0, %i floordiv 16 + 4, %i mod 16 + 4] : memref<1x16x18xf32, 1>
    }
    return
  }
}

// CHECK:       #[[SET:.+]] = affine_set<(d0) : (d0 - 1 >= 0)>
// CHECK-LABEL: func.func @ne_nonzero_rhs
// CHECK:         affine.parallel (%[[MAJOR:.+]], %[[MINOR:.+]]) = (0, 0) to (16, 16)
// CHECK-NEXT:      %[[LOAD:.+]] = affine.load %arg0[0, %[[MAJOR]] + 4, %[[MINOR]] + 4]
// CHECK-NEXT:      %[[SEL:.+]] = affine.if #[[SET]](%[[MINOR]]) -> f32 {
// CHECK-NEXT:        affine.yield %[[LOAD]]
// CHECK-NEXT:      } else {
// CHECK-NEXT:        affine.yield %cst
// CHECK-NEXT:      }
// CHECK-NEXT:      affine.store %[[SEL]], %arg1[0, %[[MAJOR]] + 4, %[[MINOR]] + 4]
