// RUN: enzymexlamlir-opt %s --raise-affine-to-stablehlo | FileCheck %s

// An optimizer hint carries no semantics a tensor program needs; a kernel
// carrying llvm.intr.assume still raises.

// CHECK-LABEL: @with_assume
// CHECK-NOT: llvm.intr.assume

module {
  func.func private @with_assume(%out: memref<16xf64, 1>, %in: memref<16xf64, 1>, %nbuf: memref<1xi32, 1>) {
    %c0_i32 = arith.constant 0 : i32
    affine.parallel (%t) = (0) to (16) {
      %n = affine.load %nbuf[0] : memref<1xi32, 1>
      %pos = arith.cmpi sgt, %n, %c0_i32 : i32
      "llvm.intr.assume"(%pos) <{op_bundle_sizes = array<i32>, op_bundle_tags = []}> : (i1) -> ()
      %v = affine.load %in[%t] : memref<16xf64, 1>
      affine.store %v, %out[%t] : memref<16xf64, 1>
    }
    return
  }
}
