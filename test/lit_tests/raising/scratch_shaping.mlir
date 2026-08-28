// RUN: enzymexlamlir-opt %s --raise-affine-to-stablehlo | FileCheck %s

// Kernel scratch reached through typed views at byte offsets (a base
// pointer re-viewed part-way in) flattens onto the underlying alloca so
// raising sees plain scratch accesses.

// CHECK-LABEL: @struct_scratch_raised
// CHECK-NOT: llvm.struct

module {
  func.func private @viewed_scratch(%out: memref<16xf64, 1>, %in: memref<16xf64, 1>) {
    %cst = arith.constant 2.5 : f64
    affine.parallel (%t) = (0) to (16) {
      %sc = memref.alloca() : memref<8xf64>
      %p = "enzymexla.memref2pointer"(%sc) : (memref<8xf64>) -> !llvm.ptr
      %g = llvm.getelementptr %p[16] : (!llvm.ptr) -> !llvm.ptr, i8
      %view = "enzymexla.pointer2memref"(%g) : (!llvm.ptr) -> memref<?xf64>
      affine.store %cst, %view[0] : memref<?xf64>
      %r = affine.load %view[0] : memref<?xf64>
      %x = affine.load %in[%t] : memref<16xf64, 1>
      %r2 = arith.mulf %r, %x : f64
      affine.store %r2, %out[%t] : memref<16xf64, 1>
    }
    return
  }

  // CHECK-LABEL: @viewed_scratch_raised
  // CHECK-NOT: llvm.getelementptr
  func.func private @struct_scratch(%out: memref<16xf64, 1>, %in: memref<16xf64, 1>) {
    %cst = arith.constant 1.5 : f64
    affine.parallel (%t) = (0) to (16) {
      %sc = memref.alloca() : memref<4x!llvm.struct<(f64, f64)>>
      %p = "enzymexla.memref2pointer"(%sc) : (memref<4x!llvm.struct<(f64, f64)>>) -> !llvm.ptr
      %view = "enzymexla.pointer2memref"(%p) : (!llvm.ptr) -> memref<?xf64>
      affine.store %cst, %view[2] : memref<?xf64>
      %r = affine.load %view[2] : memref<?xf64>
      %x = affine.load %in[%t] : memref<16xf64, 1>
      %r2 = arith.mulf %r, %x : f64
      affine.store %r2, %out[%t] : memref<16xf64, 1>
    }
    return
  }
}
