// RUN: enzymexlamlir-opt --polygeist-mem2reg --split-input-file %s | FileCheck %s

// A transfer writing the slot is stood for by a read of what it moved, which is
// worth the load it costs only where something goes on to ask for it. Here the
// one read of the destination comes before the copy and so is not replaceable,
// and the pass leaves without replacing anything -- the read of the source has
// to leave with it.
//
// Left behind it is a read of the source, which promoting the source then sees
// as a load nobody uses, sweeps, and counts as progress; promoting the
// destination writes it again the round after, and the two never settle.

llvm.func @use(!llvm.ptr {llvm.nocapture})
llvm.func @sink(i160)

llvm.func @unasked_read(%p: !llvm.ptr) {
  %c1 = llvm.mlir.constant(1 : i32) : i32
  %n = llvm.mlir.constant(20 : i64) : i64
  %src = llvm.alloca %c1 x !llvm.struct<"S", packed (ptr, i32, i32, i32, array<4 x i8>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %dst = llvm.alloca %c1 x !llvm.struct<"S", packed (ptr, i32, i32, i32, array<4 x i8>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  llvm.store %p, %src {alignment = 8 : i64} : !llvm.ptr, !llvm.ptr
  %v = llvm.load %dst {alignment = 8 : i64} : !llvm.ptr -> i160
  llvm.call @sink(%v) : (i160) -> ()
  "llvm.intr.memcpy"(%dst, %src, %n) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
  llvm.call @use(%dst) : (!llvm.ptr {llvm.nocapture}) -> ()
  llvm.return
}

// The read of the destination is the only load there is any use for, and the
// copy still stands.

// CHECK-LABEL: llvm.func @unasked_read
// CHECK:         %[[SRC:.+]] = llvm.alloca
// CHECK:         %[[DST:.+]] = llvm.alloca
// CHECK:         llvm.store %arg0, %[[SRC]]
// CHECK-NEXT:    %[[V:.+]] = llvm.load %[[DST]]
// CHECK-NEXT:    llvm.call @sink(%[[V]])
// CHECK-NEXT:    "llvm.intr.memcpy"(%[[DST]], %[[SRC]]
// CHECK-NEXT:    llvm.call @use(%[[DST]])
// CHECK-NEXT:    llvm.return
