// RUN: enzymexlamlir-opt %s --raise-affine-to-stablehlo | FileCheck %s

// Staging helpers return null for empty buffers, so a captured device
// pointer arrives as `select(size > 0, ptr, null)`; the pointer is only
// dereferenced, so the select collapses to the real pointer and the kernel
// raises.
llvm.func @kern(%out: !llvm.ptr, %in: !llvm.ptr, %n: i32) {
  %c1 = arith.constant 1 : index
  %c0_i32 = arith.constant 0 : i32
  %null = llvm.mlir.zero : !llvm.ptr
  %ok = arith.cmpi sgt, %n, %c0_i32 : i32
  %p = arith.select %ok, %out, %null : !llvm.ptr
  %om = "enzymexla.pointer2memref"(%p) : (!llvm.ptr) -> memref<?xf64>
  %im = "enzymexla.pointer2memref"(%in) : (!llvm.ptr) -> memref<?xf64>
  %0 = "enzymexla.gpu_wrapper"(%c1, %c1, %c1, %c1, %c1, %c1) ({
    affine.parallel (%i) = (0) to (16) {
      %v = affine.load %im[%i] : memref<?xf64>
      affine.store %v, %om[%i] : memref<?xf64>
    }
    "enzymexla.polygeist_yield"() : () -> ()
  }) : (index, index, index, index, index, index) -> index
  llvm.return
}

// CHECK-LABEL: llvm.func @kern(
// CHECK-NOT: arith.select
// CHECK: enzymexla.xla_wrapper @rxla$raised
