// RUN: enzymexlamlir-opt %s --canonicalize-parallel="parallel=false" | FileCheck %s

module {
  llvm.func @memset_dst(%p: !llvm.ptr, %n: i32, %i: i64) {
    %null = llvm.mlir.zero : !llvm.ptr
    %c0 = arith.constant 0 : i32
    %c0_i8 = arith.constant 0 : i8
    %c24 = arith.constant 24 : i64
    %ok = arith.cmpi sgt, %n, %c0 : i32
    %s = arith.select %ok, %p, %null : !llvm.ptr
    %g = llvm.getelementptr %s[%i] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<8 x i8>
    %h = llvm.getelementptr %g[24] : (!llvm.ptr) -> !llvm.ptr, i8
    "llvm.intr.memset"(%h, %c0_i8, %c24) <{isVolatile = false}> : (!llvm.ptr, i8, i64) -> ()
    llvm.return
  }

  llvm.func @memcpy_src(%p: !llvm.ptr, %dst: !llvm.ptr, %n: i32) {
    %null = llvm.mlir.zero : !llvm.ptr
    %c0 = arith.constant 0 : i32
    %c16 = arith.constant 16 : i64
    %ok = arith.cmpi sgt, %n, %c0 : i32
    %s = arith.select %ok, %null, %p : !llvm.ptr
    "llvm.intr.memcpy"(%dst, %s, %c16) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    llvm.return
  }

  llvm.func @memmove_dst(%p: !llvm.ptr, %src: !llvm.ptr, %n: i32) {
    %null = llvm.mlir.zero : !llvm.ptr
    %c0 = arith.constant 0 : i32
    %c16 = arith.constant 16 : i64
    %ok = arith.cmpi sgt, %n, %c0 : i32
    %s = arith.select %ok, %p, %null : !llvm.ptr
    "llvm.intr.memmove"(%s, %src, %c16) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    llvm.return
  }

  llvm.func @memset_then_compared(%p: !llvm.ptr, %n: i32) -> i1 {
    %null = llvm.mlir.zero : !llvm.ptr
    %c0 = arith.constant 0 : i32
    %c0_i8 = arith.constant 0 : i8
    %c24 = arith.constant 24 : i64
    %ok = arith.cmpi sgt, %n, %c0 : i32
    %s = arith.select %ok, %p, %null : !llvm.ptr
    "llvm.intr.memset"(%s, %c0_i8, %c24) <{isVolatile = false}> : (!llvm.ptr, i8, i64) -> ()
    %isnull = llvm.icmp "eq" %s, %null : !llvm.ptr
    llvm.return %isnull : i1
  }
}

// CHECK-LABEL: llvm.func @memset_dst(
// CHECK-NOT: arith.select
// CHECK: %[[g:.+]] = llvm.getelementptr %arg0[%arg2]
// CHECK: %[[h:.+]] = llvm.getelementptr %[[g]][24]
// CHECK: "llvm.intr.memset"(%[[h]],

// CHECK-LABEL: llvm.func @memcpy_src(
// CHECK-NOT: arith.select
// CHECK: "llvm.intr.memcpy"(%arg1, %arg0,

// CHECK-LABEL: llvm.func @memmove_dst(
// CHECK-NOT: arith.select
// CHECK: "llvm.intr.memmove"(%arg0, %arg1,

// CHECK-LABEL: llvm.func @memset_then_compared(
// CHECK: arith.select
// CHECK: llvm.icmp
