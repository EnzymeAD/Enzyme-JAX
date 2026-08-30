// RUN: enzymexlamlir-opt %s --raise-affine-to-stablehlo | FileCheck %s

// Data-dependent indexing (CSR-style) stays as raw gep+load; the access
// still addresses whole elements, so it converts to a flat memref access
// and raises as a gather.

// CHECK-LABEL: @csr_raised
// CHECK: stablehlo.gather
// CHECK-NOT: llvm.load

module {
  func.func private @csr(%out: memref<16xf64, 1>, %in: memref<16xf64, 1>, %idx: memref<16xi32, 1>) {
    affine.parallel (%t) = (0) to (16) {
      %i = affine.load %idx[%t] : memref<16xi32, 1>
      %i64 = arith.extsi %i : i32 to i64
      %p = "enzymexla.memref2pointer"(%in) : (memref<16xf64, 1>) -> !llvm.ptr<1>
      %g = llvm.getelementptr %p[%i64] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f64
      %v = llvm.load %g : !llvm.ptr<1> -> f64
      affine.store %v, %out[%t] : memref<16xf64, 1>
    }
    return
  }
}
