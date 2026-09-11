// RUN: enzymexlamlir-opt --affine-cfg --allow-unregistered-dialect %s | FileCheck %s

// The unsigned compare's operand is (%blk * 16 + %tid + 1) + (%c - %blk) * 16,
// the shape a tiled kernel's halo mask arrives in: the block offset is added
// and subtracted around the thread offset. Nothing about the subtraction is
// non-negative on its own, so the direct test fails; composing first cancels
// %blk and leaves %tid + %c * 16 + 1, which the loop bounds settle. Only then
// does the `uge` read as a signed affine constraint.

// CHECK: #set = affine_set<(d0, d1) : (d0 + d1 * 16 - 4 >= 0)>
// CHECK-LABEL: func.func private @composed_nonneg
// CHECK-NOT: arith.cmpi
// CHECK: affine.if #set(%[[TID:.+]], %[[C:.+]]) -> f32 {
// CHECK-NOT: arith.cmpi

module {
  func.func private @composed_nonneg(%out: memref<10x16x3xf32, 1>) {
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %c16_i64 = arith.constant 16 : i64
    %c4_i64 = arith.constant 4 : i64
    %cst = arith.constant 1.000000e+00 : f32
    %cst_0 = arith.constant 2.000000e+00 : f32
    affine.parallel (%blk, %tid, %c) = (0, 0, 0) to (10, 16, 3) {
      %0 = arith.muli %blk, %c16 : index
      %1 = arith.addi %0, %tid : index
      %2 = arith.index_castui %1 : index to i64
      %3 = arith.subi %c, %blk : index
      %4 = arith.index_cast %3 : index to i64
      %5 = arith.muli %4, %c16_i64 : i64
      %6 = arith.addi %2, %5 : i64
      %7 = arith.cmpi uge, %6, %c4_i64 : i64
      %8 = arith.select %7, %cst, %cst_0 : f32
      affine.store %8, %out[%blk, %tid, %c] : memref<10x16x3xf32, 1>
    }
    return
  }
}
