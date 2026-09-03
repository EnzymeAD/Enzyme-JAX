// RUN: enzymexlamlir-opt %s --canonicalize-parallel="parallel=false" | FileCheck %s

module {
  llvm.func @stored_and_memset(%p: !llvm.ptr, %n: i32, %i: i64, %slot: !llvm.ptr) {
    %null = llvm.mlir.zero : !llvm.ptr
    %c0 = arith.constant 0 : i32
    %c0_i8 = arith.constant 0 : i8
    %c24 = arith.constant 24 : i64
    %ok = arith.cmpi sgt, %n, %c0 : i32
    %s = arith.select %ok, %p, %null : !llvm.ptr
    llvm.store %s, %slot : !llvm.ptr, !llvm.ptr
    %g = llvm.getelementptr %s[%i] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<8 x i8>
    %h = llvm.getelementptr %g[24] : (!llvm.ptr) -> !llvm.ptr, i8
    "llvm.intr.memset"(%h, %c0_i8, %c24) <{isVolatile = false}> : (!llvm.ptr, i8, i64) -> ()
    llvm.return
  }

  llvm.func @compared_and_loaded(%p: !llvm.ptr, %n: i32) -> i1 {
    %null = llvm.mlir.zero : !llvm.ptr
    %c0 = arith.constant 0 : i32
    %ok = arith.cmpi sgt, %n, %c0 : i32
    %s = arith.select %ok, %p, %null : !llvm.ptr
    %v = llvm.load %s : !llvm.ptr -> f64
    %m = "enzymexla.pointer2memref"(%s) : (!llvm.ptr) -> memref<?xf64>
    affine.store %v, %m[3] : memref<?xf64>
    %isnull = llvm.icmp "eq" %s, %null : !llvm.ptr
    llvm.return %isnull : i1
  }

  llvm.func @yielded_and_escaping(%p: !llvm.ptr, %n: i32, %c: i1, %sink: !llvm.ptr) -> f64 {
    %null = llvm.mlir.zero : !llvm.ptr
    %c0 = arith.constant 0 : i32
    %ok = arith.cmpi sgt, %n, %c0 : i32
    %s = arith.select %ok, %null, %p : !llvm.ptr
    %q = scf.if %c -> !llvm.ptr {
      %g = llvm.getelementptr %s[1] : (!llvm.ptr) -> !llvm.ptr, f64
      scf.yield %g : !llvm.ptr
    } else {
      scf.yield %sink : !llvm.ptr
    }
    %v = llvm.load %q : !llvm.ptr -> f64
    llvm.call @escape(%s) : (!llvm.ptr) -> ()
    llvm.return %v : f64
  }
  llvm.func @escape(!llvm.ptr)

  llvm.func @only_escapes(%p: !llvm.ptr, %n: i32, %sink: !llvm.ptr) {
    %null = llvm.mlir.zero : !llvm.ptr
    %c0 = arith.constant 0 : i32
    %ok = arith.cmpi sgt, %n, %c0 : i32
    %s = arith.select %ok, %p, %null : !llvm.ptr
    llvm.store %s, %sink : !llvm.ptr, !llvm.ptr
    %g = llvm.getelementptr %s[4] : (!llvm.ptr) -> !llvm.ptr, f64
    llvm.call @escape(%g) : (!llvm.ptr) -> ()
    llvm.return
  }
}

// The stored copy keeps the select; the memset's address chain takes the
// real pointer.
// CHECK-LABEL: llvm.func @stored_and_memset(
// CHECK: %[[sel:.+]] = arith.select %{{.*}}, %arg0, %{{.*}} : !llvm.ptr
// CHECK: llvm.store %[[sel]], %arg3
// CHECK: %[[g:.+]] = llvm.getelementptr %arg0[%arg2]
// CHECK: %[[h:.+]] = llvm.getelementptr %[[g]][24]
// CHECK: "llvm.intr.memset"(%[[h]],

// CHECK-LABEL: llvm.func @compared_and_loaded(
// CHECK: %[[sel:.+]] = arith.select %{{.*}}, %arg0, %{{.*}} : !llvm.ptr
// CHECK: llvm.load %arg0
// CHECK: "enzymexla.pointer2memref"(%arg0)
// CHECK: llvm.icmp "eq" %[[sel]],

// The yield carries the pointer to a load only, so the branch's address chain
// takes the real pointer; the call keeps the select.
// CHECK-LABEL: llvm.func @yielded_and_escaping(
// CHECK: %[[sel:.+]] = arith.select %{{.*}}, %{{.*}}, %arg0 : !llvm.ptr
// CHECK: %[[q:.+]] = scf.if %arg2 -> (!llvm.ptr) {
// CHECK-NEXT: %[[g:.+]] = llvm.getelementptr %arg0[1]
// CHECK-NEXT: scf.yield %[[g]] : !llvm.ptr
// CHECK: scf.yield %arg3 : !llvm.ptr
// CHECK: llvm.load %[[q]]
// CHECK: llvm.call @escape(%[[sel]])

// CHECK-LABEL: llvm.func @only_escapes(
// CHECK: %[[sel:.+]] = arith.select
// CHECK: llvm.store %[[sel]], %arg2
// CHECK: llvm.getelementptr %[[sel]][4]
