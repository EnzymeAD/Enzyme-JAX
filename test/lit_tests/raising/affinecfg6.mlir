// RUN: enzymexlamlir-opt %s --affine-cfg --mlir-print-local-scope | FileCheck %s

#map = affine_map<(d0) -> (d0 mod 5)>
#map1 = affine_map<(d0) -> (d0 floordiv 5)>
module {
  func.func @split_iv(%arg0: memref<10xi64>) {
    affine.parallel (%i) = (0) to (10) {
      %0 = affine.apply #map (%i)
      %1 = affine.apply #map1 (%i)
      %2 = arith.addi %0, %1 : index
      %3 = arith.index_cast %2 : index to i64
      affine.store %3, %arg0[%i] : memref<10xi64>
    }
    return
  }
// CHECK:  func.func @split_iv(%arg0: memref<10xi64>) {
// CHECK-NEXT:    affine.parallel (%arg1, %arg2) = (0, 0) to (2, 5) {
// CHECK-NEXT:      %0 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%arg1, %arg2)
// CHECK-NEXT:      %1 = arith.index_cast %0 : index to i64
// CHECK-NEXT:      affine.store %1, %arg0[%arg2 + %arg1 * 5] : memref<10xi64>
// CHECK-NEXT:    }
// CHECK-NEXT:    return
// CHECK-NEXT:  }

  func.func @merge_ivs(%arg0: memref<3xi64>) {
    affine.parallel (%c, %i, %j) = (0, 0, 0) to (3, 10, 2) {
      %0 = affine.apply affine_map<(d0, d1) -> (d0 * 2 + d1)> (%i, %j)
      %3 = arith.index_cast %0 : index to i64
      affine.store %3, %arg0[%c] : memref<3xi64>
    }
    return
  }
// CHECK:  func.func @merge_ivs(%arg0: memref<3xi64>) {
// CHECK-NEXT:    affine.parallel (%arg1, %arg2) = (0, 0) to (3, 20) {
// CHECK-NEXT:      %0 = arith.index_cast %arg2 : index to i64
// CHECK-NEXT:      affine.store %0, %arg0[%arg1] : memref<3xi64>
// CHECK-NEXT:    }
// CHECK-NEXT:    return
// CHECK-NEXT:  }
}

