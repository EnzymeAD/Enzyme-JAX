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

  llvm.func @atomics_and_views(%p: !llvm.ptr, %n: i32, %x: f64) {
    %null = llvm.mlir.zero : !llvm.ptr
    %c0 = arith.constant 0 : i32
    %idx = arith.constant 0 : index
    %ok = arith.cmpi sgt, %n, %c0 : i32
    %s = arith.select %ok, %p, %null : !llvm.ptr
    %m = "enzymexla.pointer2memref"(%s) : (!llvm.ptr) -> memref<?xf64>
    %r = "enzymexla.memref2pointer"(%m) : (memref<?xf64>) -> !llvm.ptr
    %m2 = "enzymexla.pointer2memref"(%r) : (!llvm.ptr) -> memref<?xf64>
    %a = enzyme.atomic_rmw addf %x, %m2[%idx] acq_rel : (f64, memref<?xf64>) -> f64
    %b = enzyme.affine_atomic_rmw addf %x, %m2, (affine_map<(d0) -> (d0)>)[%idx] acq_rel : (f64, memref<?xf64>) -> f64
    llvm.return
  }

  llvm.func @scf_if_null(%p: !llvm.ptr, %c: i1, %x: f64) {
    %null = llvm.mlir.zero : !llvm.ptr
    %q = scf.if %c -> !llvm.ptr {
      scf.yield %p : !llvm.ptr
    } else {
      scf.yield %null : !llvm.ptr
    }
    %m = "enzymexla.pointer2memref"(%q) : (!llvm.ptr) -> memref<?xf64>
    affine.store %x, %m[0] : memref<?xf64>
    %v = affine.load %m[1] : memref<?xf64>
    affine.store %v, %m[2] : memref<?xf64>
    llvm.return
  }

  func.func @affine_if_null(%p: !llvm.ptr, %n: index) -> f64 {
    %null = llvm.mlir.zero : !llvm.ptr
    %q = affine.if affine_set<()[s0] : (s0 - 1 >= 0)>()[%n] -> !llvm.ptr {
      affine.yield %p : !llvm.ptr
    } else {
      affine.yield %null : !llvm.ptr
    }
    %v = llvm.load %q : !llvm.ptr -> f64
    return %v : f64
  }

  llvm.func @kept_arm_defined_inside(%p: !llvm.ptr, %c: i1) -> f64 {
    %null = llvm.mlir.zero : !llvm.ptr
    %q = scf.if %c -> !llvm.ptr {
      %g = llvm.getelementptr %p[4] : (!llvm.ptr) -> !llvm.ptr, f64
      scf.yield %g : !llvm.ptr
    } else {
      scf.yield %null : !llvm.ptr
    }
    %v = llvm.load %q : !llvm.ptr -> f64
    llvm.return %v : f64
  }
}

// A view, a round trip back to a pointer, and the enzyme atomics all only
// reach the pointed-to memory.
// CHECK-LABEL: llvm.func @atomics_and_views(
// CHECK-NOT: arith.select

// A load with no other use sinks into the arms before this pattern is
// reached; a pointer viewed as a memref stays the branch's result.
// CHECK-LABEL: llvm.func @scf_if_null(
// CHECK-NOT: scf.if
// CHECK-NOT: llvm.mlir.zero
// CHECK: "enzymexla.pointer2memref"(%arg0)

// CHECK-LABEL: func.func @affine_if_null(
// CHECK-NOT: affine.if
// CHECK: llvm.load %arg0

// The kept arm is computed inside the branch, so it is not available where
// the result is used: left alone.
// CHECK-LABEL: llvm.func @kept_arm_defined_inside(
// CHECK: scf.if
