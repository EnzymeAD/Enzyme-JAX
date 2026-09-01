// RUN: not enzymexlamlir-opt %s --raise-affine-to-stablehlo 2>&1 | FileCheck %s

// A scatter has nowhere to keep an ordering, so an atomic asking for more
// than atomicity itself does not raise into one: an acquire or a release is
// synchronising with something the scatter cannot express.

module {
  func.func private @seq_cst_atomic(%buf: memref<8xf64, 1>, %idx: memref<16xi32, 1>, %val: memref<16xf64, 1>) {
    affine.parallel (%t) = (0) to (16) {
      %i = affine.load %idx[%t] : memref<16xi32, 1>
      %iidx = arith.index_cast %i : i32 to index
      %v = affine.load %val[%t] : memref<16xf64, 1>
      %old = enzyme.atomic_rmw addf %v, %buf[%iidx] seq_cst : (f64, memref<8xf64, 1>) -> f64
    }
    return
  }
}

// CHECK: failed to raise func: func.func private @seq_cst_atomic
// CHECK: enzyme.atomic_rmw addf
