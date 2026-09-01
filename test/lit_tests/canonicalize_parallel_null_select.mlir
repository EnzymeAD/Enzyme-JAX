// RUN: enzymexlamlir-opt %s --canonicalize-parallel="parallel=false" | FileCheck %s

module {
  llvm.func @deref(%p: !llvm.ptr, %n: i32) -> f64 {
    %null = llvm.mlir.zero : !llvm.ptr
    %c0 = arith.constant 0 : i32
    %ok = arith.cmpi sgt, %n, %c0 : i32
    %s = arith.select %ok, %p, %null : !llvm.ptr
    %v = llvm.load %s : !llvm.ptr -> f64
    llvm.return %v : f64
  }

  llvm.func @through_gep_and_view(%p: !llvm.ptr, %n: i32, %x: f64) {
    %null = llvm.mlir.zero : !llvm.ptr
    %c0 = arith.constant 0 : i32
    %ok = arith.cmpi sgt, %n, %c0 : i32
    %s = arith.select %ok, %null, %p : !llvm.ptr
    %g = llvm.getelementptr %s[4] : (!llvm.ptr) -> !llvm.ptr, f64
    %m = "enzymexla.pointer2memref"(%g) : (!llvm.ptr) -> memref<?xf64>
    affine.store %x, %m[2] : memref<?xf64>
    llvm.return
  }

  llvm.func @yielded_from_if(%p: !llvm.ptr, %n: i32, %c: i1) -> f64 {
    %null = llvm.mlir.zero : !llvm.ptr
    %c0 = arith.constant 0 : i32
    %ok = arith.cmpi sgt, %n, %c0 : i32
    %s = arith.select %ok, %p, %null : !llvm.ptr
    %q = scf.if %c -> !llvm.ptr {
      scf.yield %s : !llvm.ptr
    } else {
      scf.yield %p : !llvm.ptr
    }
    %v = llvm.load %q : !llvm.ptr -> f64
    llvm.return %v : f64
  }

  llvm.func @yielded_from_if_then_compared(%p: !llvm.ptr, %n: i32, %c: i1) -> i1 {
    %null = llvm.mlir.zero : !llvm.ptr
    %c0 = arith.constant 0 : i32
    %ok = arith.cmpi sgt, %n, %c0 : i32
    %s = arith.select %ok, %p, %null : !llvm.ptr
    %q = scf.if %c -> !llvm.ptr {
      scf.yield %s : !llvm.ptr
    } else {
      scf.yield %p : !llvm.ptr
    }
    %isnull = llvm.icmp "eq" %q, %null : !llvm.ptr
    llvm.return %isnull : i1
  }

  llvm.func @compared(%p: !llvm.ptr, %n: i32) -> i1 {
    %null = llvm.mlir.zero : !llvm.ptr
    %c0 = arith.constant 0 : i32
    %ok = arith.cmpi sgt, %n, %c0 : i32
    %s = arith.select %ok, %p, %null : !llvm.ptr
    %v = llvm.load %s : !llvm.ptr -> f64
    %isnull = llvm.icmp "eq" %s, %null : !llvm.ptr
    llvm.return %isnull : i1
  }

  llvm.func @escapes(%p: !llvm.ptr, %n: i32, %sink: !llvm.ptr) {
    %null = llvm.mlir.zero : !llvm.ptr
    %c0 = arith.constant 0 : i32
    %ok = arith.cmpi sgt, %n, %c0 : i32
    %s = arith.select %ok, %p, %null : !llvm.ptr
    llvm.store %s, %sink : !llvm.ptr, !llvm.ptr
    llvm.return
  }
}

// CHECK-LABEL: llvm.func @deref(
// CHECK-NOT: arith.select
// CHECK: llvm.load %arg0

// The gep then folds into the view's index, which is a separate pattern.
// CHECK-LABEL: llvm.func @through_gep_and_view(
// CHECK-NOT: arith.select
// CHECK: "enzymexla.pointer2memref"(%arg0)

// The select is reached through the branch's result, which is only loaded.
// CHECK-LABEL: llvm.func @yielded_from_if(
// CHECK-NOT: arith.select

// Reached through the branch's result, which is compared: left alone.
// CHECK-LABEL: llvm.func @yielded_from_if_then_compared(
// CHECK: arith.select

// CHECK-LABEL: llvm.func @compared(
// CHECK: arith.select

// CHECK-LABEL: llvm.func @escapes(
// CHECK: arith.select
