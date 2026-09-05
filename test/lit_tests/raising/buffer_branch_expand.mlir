// RUN: enzymexlamlir-opt %s --raise-affine-to-stablehlo | FileCheck %s

// A branch yielding whole buffers (mfem's `const_coeff ? c0 : c` ternary)
// cannot raise as a value; expand each access into the branch so only
// scalars cross the yield.

// CHECK-LABEL: @ifbuf_raised
// CHECK: stablehlo.select

module {
  func.func private @sel(%out: memref<16xf64, 1>, %a: memref<16xf64, 1>, %b: memref<16xf64, 1>, %flag: memref<1xi1, 1>) {
    affine.parallel (%t) = (0) to (16) {
      %c = affine.load %flag[0] : memref<1xi1, 1>
      %sel = arith.select %c, %a, %b : memref<16xf64, 1>
      %v = affine.load %sel[%t] : memref<16xf64, 1>
      affine.store %v, %out[%t] : memref<16xf64, 1>
    }
    return
  }

  // CHECK-LABEL: @sel_raised
  // CHECK: stablehlo.select
  // CHECK-NOT: arith.select
  func.func private @ifbuf(%out: memref<16xf64, 1>, %a: memref<16xf64, 1>, %b: memref<16xf64, 1>, %n: memref<1xi32, 1>) {
    %nv = affine.load %n[0] : memref<1xi32, 1>
    %ni = arith.index_cast %nv : i32 to index
    affine.parallel (%t) = (0) to (16) {
      %buf = affine.if affine_set<()[s0] : (s0 - 1 >= 0)>()[%ni] -> memref<16xf64, 1> {
        affine.yield %a : memref<16xf64, 1>
      } else {
        affine.yield %b : memref<16xf64, 1>
      }
      %v = affine.load %buf[%t] : memref<16xf64, 1>
      affine.store %v, %out[%t] : memref<16xf64, 1>
    }
    return
  }
}
