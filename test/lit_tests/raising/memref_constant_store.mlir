// RUN: enzymexlamlir-opt %s --raise-affine-to-stablehlo | FileCheck %s
// XFAIL: *
// This test fails because it contains a race condition (parallel writes to the same constant location).

func.func @nested_constant_store(%arg0: memref<32x1xi32>) {
  affine.parallel (%arg3) = (0) to (32) {
    affine.for %arg5 = 0 to 19 step 32 {
      %0 = arith.index_cast %arg3 : index to i32
      %c0 = arith.constant 0 : index
      memref.store %0, %arg0[%c0, %c0] : memref<32x1xi32>
    }
  }
  return
}
