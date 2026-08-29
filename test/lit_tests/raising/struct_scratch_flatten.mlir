// RUN: enzymexlamlir-opt %s --raise-affine-to-stablehlo | FileCheck %s

// Scratch declared as struct values (a union wrapping a register array)
// reaches the memref world as a memref of LLVM struct whose only consumers
// cast straight back to a pointer; it flattens to scalar scratch and the
// walked views rebase like any other buffer.

// CHECK-LABEL: @structscratch
// CHECK-NOT: llvm.struct
// CHECK-NOT: llvm.getelementptr

module {
  func.func private @structscratch(%out: memref<16xf64, 1>, %in: memref<16xf64, 1>) {
    %m = memref.alloca() {alignment = 8 : i64} : memref<4x!llvm.struct<"union.anon", (array<2 x array<2 x f64>>)>>
    %p = "enzymexla.memref2pointer"(%m) : (memref<4x!llvm.struct<"union.anon", (array<2 x array<2 x f64>>)>>) -> !llvm.ptr
    affine.parallel (%t) = (0) to (16) {
      %t64 = arith.index_castui %t : index to i64
      %g = llvm.getelementptr %p[%t64] : (!llvm.ptr, i64) -> !llvm.ptr, f64
      %view = "enzymexla.pointer2memref"(%g) : (!llvm.ptr) -> memref<?xf64>
      %v = affine.load %in[%t] : memref<16xf64, 1>
      affine.store %v, %view[0] : memref<?xf64>
      %r = affine.load %view[0] : memref<?xf64>
      affine.store %r, %out[%t] : memref<16xf64, 1>
    }
    return
  }
}
