// RUN: enzymexlamlir-opt %s --raise-affine-to-stablehlo | FileCheck %s

// A view carved out of a buffer at a runtime element offset (gep feeding a
// typed pointer2memref) rebases onto the underlying buffer: the view's
// accesses index the base plus the offset, and raise as a gather.

// CHECK-LABEL: @viewed_raised
// CHECK: stablehlo.gather
// CHECK-NOT: llvm.getelementptr

module {
  func.func private @viewed(%out: memref<16xf64, 1>, %in: memref<64xf64, 1>, %idx: memref<16xi32, 1>) {
    affine.parallel (%t) = (0) to (16) {
      %i = affine.load %idx[%t] : memref<16xi32, 1>
      %i64 = arith.extsi %i : i32 to i64
      %p = "enzymexla.memref2pointer"(%in) : (memref<64xf64, 1>) -> !llvm.ptr<1>
      %g = llvm.getelementptr %p[%i64] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f64
      %view = "enzymexla.pointer2memref"(%g) : (!llvm.ptr<1>) -> memref<?xf64, 1>
      %v = affine.load %view[0] : memref<?xf64, 1>
      affine.store %v, %out[%t] : memref<16xf64, 1>
    }
    return
  }
}
