// RUN: enzymexlamlir-opt --affine-cfg %s | FileCheck %s

// Raising the scf.if to affine.if composes an integer set over %v, whose
// non-index type needs an index cast inserted "after its defining op". That
// op is an llvm.invoke: a terminator. Inserting after it would place the cast
// past the end of the block, hiding the invoke's successors from CFG walks;
// region simplification then erased the (live) normal destination and left
// the invoke with a null successor. The cast belongs at the start of the
// normal destination instead.

llvm.func @get() -> i64
llvm.func @pers(i32) -> i32

llvm.func @invoke_operand(%p: !llvm.ptr, %x: f64) attributes {personality = @pers} {
  %m = "enzymexla.pointer2memref"(%p) : (!llvm.ptr) -> memref<16xf64>
  %v = llvm.invoke @get() to ^bb1 unwind ^bb2 : () -> i64
^bb1:
  %c4 = arith.constant 4 : i64
  %cond = arith.cmpi slt, %v, %c4 : i64
  scf.if %cond {
    %i = arith.index_cast %v : i64 to index
    memref.store %x, %m[%i] : memref<16xf64>
  }
  llvm.return
^bb2:
  %lp = llvm.landingpad cleanup : !llvm.struct<(ptr, i32)>
  llvm.return
}

// CHECK-LABEL: llvm.func @invoke_operand
// CHECK: %[[V:.+]] = llvm.invoke @get() to ^[[NORMAL:.+]] unwind
// CHECK: ^[[NORMAL]]:
// CHECK-NEXT: %[[IDX:.+]] = arith.index_cast %[[V]] : i64 to index
// CHECK: affine.store %{{.+}}, %{{.+}}[symbol(%[[IDX]])] : memref<16xf64>
