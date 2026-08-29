// RUN: enzymexlamlir-opt %s --raise-affine-to-stablehlo | FileCheck %s

// Register-array scratch survives as an llvm.alloca of nested arrays with a
// zeroing memset and slab geps; flattened to flat scratch, the memset becomes
// element stores, and the walked views rebase like any other buffer.

// CHECK-LABEL: @regarray
// CHECK-NOT: llvm.alloca
// CHECK-NOT: llvm.intr.memset
// CHECK-NOT: llvm.getelementptr

module {
  func.func private @regarray(%out: memref<16xf64, 1>, %in: memref<16xf64, 1>) {
    %c1 = arith.constant 1 : i32
    %c0_i8 = arith.constant 0 : i8
    %c128 = arith.constant 128 : i64
    %slab = llvm.alloca %c1 x !llvm.array<4 x array<4 x f64>> : (i32) -> !llvm.ptr
    "llvm.intr.memset"(%slab, %c0_i8, %c128) <{isVolatile = false}> : (!llvm.ptr, i8, i64) -> ()
    affine.parallel (%t) = (0) to (16) {
      %t64 = arith.index_castui %t : index to i64
      %g = llvm.getelementptr %slab[%t64] : (!llvm.ptr, i64) -> !llvm.ptr, f64
      %view = "enzymexla.pointer2memref"(%g) : (!llvm.ptr) -> memref<?xf64>
      %v = affine.load %in[%t] : memref<16xf64, 1>
      %o = affine.load %view[0] : memref<?xf64>
      %a = arith.addf %o, %v : f64
      affine.store %a, %view[0] : memref<?xf64>
      %r = affine.load %view[0] : memref<?xf64>
      affine.store %r, %out[%t] : memref<16xf64, 1>
    }
    return
  }
}
