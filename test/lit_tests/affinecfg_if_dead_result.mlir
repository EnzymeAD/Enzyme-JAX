// RUN: enzymexlamlir-opt %s --affine-cfg | FileCheck %s

// A periodic halo read, `i == 1 ? x[.., Nx] : x[.., i - 1]`, of a kernel over
// a flattened 16x16 thread index: the guard also yields the index it picks,
// which nothing reads. Dropping that result frees the thread index of the
// non-affine arithmetic computing it, so the index splits into its two
// dimensions.
#set = affine_set<(d0) : (d0 mod 16 == 0)>
func.func @periodic_halo(%x: memref<10x22x22xf64, 1>, %out: memref<10x22x22xf64, 1>) {
  %c16 = arith.constant 16 : index
  %c16_i64 = arith.constant 16 : i64
  affine.parallel (%i, %j) = (0, 0) to (4, 256) {
    %a = affine.load %x[%i + 3, %j floordiv 16 + 3, %j mod 16 + 3] : memref<10x22x22xf64, 1>
    %j64 = arith.index_castui %j : index to i64
    %q = arith.divui %j, %c16 : index
    %q16 = arith.muli %q, %c16 : index
    %q16_64 = arith.index_castui %q16 : index to i64
    %r = arith.subi %j64, %q16_64 overflow<nsw> : i64
    %v:2 = affine.if #set(%j) -> (i64, f64) {
      %b = affine.load %x[%i + 3, %j floordiv 16 + 3, 18] : memref<10x22x22xf64, 1>
      affine.yield %c16_i64, %b : i64, f64
    } else {
      %b = affine.load %x[%i + 3, %j floordiv 16 + 3, %j mod 16 + 2] : memref<10x22x22xf64, 1>
      affine.yield %r, %b : i64, f64
    }
    %d = arith.subf %a, %v#1 : f64
    affine.store %d, %out[%i + 3, %j floordiv 16 + 3, %j mod 16 + 3] : memref<10x22x22xf64, 1>
  }
  return
}

// CHECK:      #set = affine_set<(d0) : (d0 == 0)>
// CHECK:      func.func @periodic_halo(%arg0: memref<10x22x22xf64, 1>, %arg1: memref<10x22x22xf64, 1>) {
// CHECK-NEXT:   affine.parallel (%arg2, %arg3, %arg4) = (0, 0, 0) to (4, 16, 16) {
// CHECK-NEXT:     %0 = affine.load %arg0[%arg2 + 3, %arg3 + 3, %arg4 + 3] : memref<10x22x22xf64, 1>
// CHECK-NEXT:     %1 = affine.if #set(%arg4) -> f64 {
// CHECK-NEXT:       %3 = affine.load %arg0[%arg2 + 3, %arg3 + 3, 18] : memref<10x22x22xf64, 1>
// CHECK-NEXT:       affine.yield %3 : f64
// CHECK-NEXT:     } else {
// CHECK-NEXT:       %3 = affine.load %arg0[%arg2 + 3, %arg3 + 3, %arg4 + 2] : memref<10x22x22xf64, 1>
// CHECK-NEXT:       affine.yield %3 : f64
// CHECK-NEXT:     }
// CHECK-NEXT:     %2 = arith.subf %0, %1 : f64
// CHECK-NEXT:     affine.store %2, %arg1[%arg2 + 3, %arg3 + 3, %arg4 + 3] : memref<10x22x22xf64, 1>
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }
