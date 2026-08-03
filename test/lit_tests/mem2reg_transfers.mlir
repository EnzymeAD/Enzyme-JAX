// RUN: enzymexlamlir-opt %s -polygeist-mem2reg -split-input-file | FileCheck %s

// A memcpy reading the allocation does not stop the parts it does not name from
// being forwarded, and does not let the pointer escape.
llvm.func @memcpy_src(%dst: !llvm.ptr) -> i32 {
  %c1 = llvm.mlir.constant(1 : i32) : i32
  %len = llvm.mlir.constant(4 : i64) : i64
  %val = llvm.mlir.constant(42 : i32) : i32
  %mem = llvm.alloca %c1 x i32 : (i32) -> !llvm.ptr
  llvm.store %val, %mem : i32, !llvm.ptr
  "llvm.intr.memcpy"(%dst, %mem, %len) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
  %loaded = llvm.load %mem : !llvm.ptr -> i32
  llvm.return %loaded : i32
}

// CHECK-LABEL: llvm.func @memcpy_src(
// CHECK: %[[C42:.+]] = llvm.mlir.constant(42 : i32) : i32
// CHECK: "llvm.intr.memcpy"
// CHECK-NOT: llvm.load
// CHECK: llvm.return %[[C42]] : i32

// -----

// A memcpy writing the allocation overwrites it, so the store before it must
// not reach the load after it.
llvm.func @memcpy_dst(%src: !llvm.ptr) -> i32 {
  %c1 = llvm.mlir.constant(1 : i32) : i32
  %len = llvm.mlir.constant(4 : i64) : i64
  %val = llvm.mlir.constant(42 : i32) : i32
  %mem = llvm.alloca %c1 x i32 : (i32) -> !llvm.ptr
  llvm.store %val, %mem : i32, !llvm.ptr
  "llvm.intr.memcpy"(%mem, %src, %len) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
  %loaded = llvm.load %mem : !llvm.ptr -> i32
  llvm.return %loaded : i32
}

// CHECK-LABEL: llvm.func @memcpy_dst(
// CHECK: %[[LD:.+]] = llvm.load
// CHECK: llvm.return %[[LD]] : i32

// -----

// Likewise for a memset, and for a memmove reading it.
llvm.func @memset_dst(%other: !llvm.ptr) -> i32 {
  %c1 = llvm.mlir.constant(1 : i32) : i32
  %len = llvm.mlir.constant(4 : i64) : i64
  %byte = llvm.mlir.constant(0 : i8) : i8
  %val = llvm.mlir.constant(42 : i32) : i32
  %mem = llvm.alloca %c1 x i32 : (i32) -> !llvm.ptr
  llvm.store %val, %mem : i32, !llvm.ptr
  "llvm.intr.memset"(%mem, %byte, %len) <{isVolatile = false}> : (!llvm.ptr, i8, i64) -> ()
  %loaded = llvm.load %mem : !llvm.ptr -> i32
  llvm.return %loaded : i32
}

// CHECK-LABEL: llvm.func @memset_dst(
// CHECK: %[[LD:.+]] = llvm.load
// CHECK: llvm.return %[[LD]] : i32

// -----

llvm.func @memmove_src(%dst: !llvm.ptr) -> i32 {
  %c1 = llvm.mlir.constant(1 : i32) : i32
  %len = llvm.mlir.constant(4 : i64) : i64
  %val = llvm.mlir.constant(42 : i32) : i32
  %mem = llvm.alloca %c1 x i32 : (i32) -> !llvm.ptr
  llvm.store %val, %mem : i32, !llvm.ptr
  "llvm.intr.memmove"(%dst, %mem, %len) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
  %loaded = llvm.load %mem : !llvm.ptr -> i32
  llvm.return %loaded : i32
}

// CHECK-LABEL: llvm.func @memmove_src(
// CHECK: %[[C42:.+]] = llvm.mlir.constant(42 : i32) : i32
// CHECK: "llvm.intr.memmove"
// CHECK-NOT: llvm.load
// CHECK: llvm.return %[[C42]] : i32

// -----

// An allocation only ever written -- through a view, an offset of it, and a
// transfer into it -- is dead and goes away with all of them.
llvm.func @dead_alloca(%src: !llvm.ptr) {
  %c0 = arith.constant 0 : index
  %c1 = llvm.mlir.constant(1 : i32) : i32
  %c2 = llvm.mlir.constant(2 : i64) : i64
  %len = llvm.mlir.constant(4 : i64) : i64
  %byte = llvm.mlir.constant(0 : i8) : i8
  %val = llvm.mlir.constant(42 : i32) : i32
  %mem = llvm.alloca %c1 x !llvm.array<4 x i32> : (i32) -> !llvm.ptr
  llvm.intr.lifetime.start %mem : !llvm.ptr
  %gep = llvm.getelementptr %mem[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<4 x i32>
  llvm.store %val, %gep : i32, !llvm.ptr
  %view = "enzymexla.pointer2memref"(%mem) : (!llvm.ptr) -> memref<?xi32>
  memref.store %val, %view[%c0] : memref<?xi32>
  "llvm.intr.memcpy"(%mem, %src, %len) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
  "llvm.intr.memset"(%mem, %byte, %len) <{isVolatile = false}> : (!llvm.ptr, i8, i64) -> ()
  llvm.intr.lifetime.end %mem : !llvm.ptr
  llvm.return
}

// CHECK-LABEL: llvm.func @dead_alloca(
// CHECK-NOT: llvm.alloca
// CHECK-NOT: llvm.intr.mem
// CHECK: llvm.return

// -----

// The same allocation stays when something reads out of it.
llvm.func @live_alloca(%dst: !llvm.ptr) {
  %c1 = llvm.mlir.constant(1 : i32) : i32
  %len = llvm.mlir.constant(4 : i64) : i64
  %val = llvm.mlir.constant(42 : i32) : i32
  %mem = llvm.alloca %c1 x i32 : (i32) -> !llvm.ptr
  llvm.store %val, %mem : i32, !llvm.ptr
  "llvm.intr.memcpy"(%dst, %mem, %len) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
  llvm.return
}

// CHECK-LABEL: llvm.func @live_alloca(
// CHECK: llvm.alloca
// CHECK: "llvm.intr.memcpy"

// -----

// Storing the pointer itself hands it to whoever reads that memory, so nothing
// may be forwarded through it.
llvm.func @escaped_through_store(%out: !llvm.ptr) -> i32 {
  %c1 = llvm.mlir.constant(1 : i32) : i32
  %val = llvm.mlir.constant(42 : i32) : i32
  %mem = llvm.alloca %c1 x i32 : (i32) -> !llvm.ptr
  llvm.store %val, %mem : i32, !llvm.ptr
  llvm.store %mem, %out : !llvm.ptr, !llvm.ptr
  %loaded = llvm.load %mem : !llvm.ptr -> i32
  llvm.return %loaded : i32
}

// CHECK-LABEL: llvm.func @escaped_through_store(
// CHECK: %[[LD:.+]] = llvm.load
// CHECK: llvm.return %[[LD]] : i32
