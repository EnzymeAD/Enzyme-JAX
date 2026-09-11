// RUN: enzymexlamlir-opt %s --canonicalize --split-input-file | FileCheck %s

// An if whose arms all yield the same pointer/memref conversion does the
// conversion once on the if result instead. Chains of such ifs (nested
// ternaries picking a slice) collapse to one conversion over a
// pointer-yielding if chain.

#set = affine_set<()[s0] : (s0 >= 1)>
func.func private @nested(%p1: !llvm.ptr<3>, %p2: !llvm.ptr<3>, %p3: !llvm.ptr<3>, %s: index, %i: index) -> f64 {
  %a = affine.if #set()[%s] -> memref<?xf64> {
    %v = "enzymexla.pointer2memref"(%p1) : (!llvm.ptr<3>) -> memref<?xf64>
    affine.yield %v : memref<?xf64>
  } else {
    %v = "enzymexla.pointer2memref"(%p2) : (!llvm.ptr<3>) -> memref<?xf64>
    affine.yield %v : memref<?xf64>
  }
  %b = affine.if #set()[%s] -> memref<?xf64> {
    %v = "enzymexla.pointer2memref"(%p3) : (!llvm.ptr<3>) -> memref<?xf64>
    affine.yield %v : memref<?xf64>
  } else {
    affine.yield %a : memref<?xf64>
  }
  %x = memref.load %b[%i] : memref<?xf64>
  return %x : f64
}

// CHECK:  func.func private @nested(%[[p1:.+]]: !llvm.ptr<3>, %[[p2:.+]]: !llvm.ptr<3>, %[[p3:.+]]: !llvm.ptr<3>, %[[s:.+]]: index, %[[i:.+]]: index) -> f64 {
// CHECK-NEXT:  %[[a:.+]] = affine.if #set()[%[[s]]] -> !llvm.ptr<3> {
// CHECK-NEXT:    affine.yield %[[p1]] : !llvm.ptr<3>
// CHECK-NEXT:  } else {
// CHECK-NEXT:    affine.yield %[[p2]] : !llvm.ptr<3>
// CHECK-NEXT:  }
// CHECK-NEXT:  %[[b:.+]] = affine.if #set()[%[[s]]] -> !llvm.ptr<3> {
// CHECK-NEXT:    affine.yield %[[p3]] : !llvm.ptr<3>
// CHECK-NEXT:  } else {
// CHECK-NEXT:    affine.yield %[[a]] : !llvm.ptr<3>
// CHECK-NEXT:  }
// CHECK-NEXT:  %[[v:.+]] = "enzymexla.pointer2memref"(%[[b]]) : (!llvm.ptr<3>) -> memref<?xf64>
// CHECK-NEXT:  %[[x:.+]] = memref.load %[[v]][%[[i]]] : memref<?xf64>
// CHECK-NEXT:  return %[[x]] : f64
// CHECK-NEXT:  }

// -----

// The memref2pointer direction hoists the same way; the pointer-free if is
// then a trivial select.

func.func private @m2p_arms(%m1: memref<8xf64>, %m2: memref<8xf64>, %c: i1) -> !llvm.ptr {
  %r = scf.if %c -> !llvm.ptr {
    %v = "enzymexla.memref2pointer"(%m1) : (memref<8xf64>) -> !llvm.ptr
    scf.yield %v : !llvm.ptr
  } else {
    %v = "enzymexla.memref2pointer"(%m2) : (memref<8xf64>) -> !llvm.ptr
    scf.yield %v : !llvm.ptr
  }
  return %r : !llvm.ptr
}

// CHECK:  func.func private @m2p_arms(%[[m1:.+]]: memref<8xf64>, %[[m2:.+]]: memref<8xf64>, %[[c:.+]]: i1) -> !llvm.ptr {
// CHECK-NEXT:  %[[sel:.+]] = arith.select %[[c]], %[[m1]], %[[m2]] : memref<8xf64>
// CHECK-NEXT:  %[[p:.+]] = "enzymexla.memref2pointer"(%[[sel]]) : (memref<8xf64>) -> !llvm.ptr
// CHECK-NEXT:  return %[[p]] : !llvm.ptr
// CHECK-NEXT:  }

// -----

// Mixed arms (one conversion, one raw view) stay as they are.

func.func private @mixed_arms(%p1: !llvm.ptr<3>, %m: memref<?xf64>, %c: i1) -> memref<?xf64> {
  %r = scf.if %c -> memref<?xf64> {
    %v = "enzymexla.pointer2memref"(%p1) : (!llvm.ptr<3>) -> memref<?xf64>
    scf.yield %v : memref<?xf64>
  } else {
    scf.yield %m : memref<?xf64>
  }
  return %r : memref<?xf64>
}

// CHECK:  func.func private @mixed_arms(%[[p1:.+]]: !llvm.ptr<3>, %[[m:.+]]: memref<?xf64>, %[[c:.+]]: i1) -> memref<?xf64> {
// CHECK-NEXT:  %[[r:.+]] = scf.if %[[c]] -> (memref<?xf64>) {
// CHECK-NEXT:    %[[v:.+]] = "enzymexla.pointer2memref"(%[[p1]]) : (!llvm.ptr<3>) -> memref<?xf64>
// CHECK-NEXT:    scf.yield %[[v]] : memref<?xf64>
// CHECK-NEXT:  } else {
// CHECK-NEXT:    scf.yield %[[m]] : memref<?xf64>
// CHECK-NEXT:  }
// CHECK-NEXT:  return %[[r]] : memref<?xf64>
// CHECK-NEXT:  }

// -----

// Conversions defined outside the if hoist too. An if whose arms only yield
// existing values becomes a select before the if pattern can see it, so the
// select twin carries the same rewrite.

func.func private @m2p_outside(%m1: memref<8xf64>, %m2: memref<8xf64>, %c: i1) -> !llvm.ptr {
  %p1 = "enzymexla.memref2pointer"(%m1) : (memref<8xf64>) -> !llvm.ptr
  %p2 = "enzymexla.memref2pointer"(%m2) : (memref<8xf64>) -> !llvm.ptr
  %r = scf.if %c -> !llvm.ptr {
    scf.yield %p1 : !llvm.ptr
  } else {
    scf.yield %p2 : !llvm.ptr
  }
  return %r : !llvm.ptr
}

// CHECK:  func.func private @m2p_outside(%[[m1:.+]]: memref<8xf64>, %[[m2:.+]]: memref<8xf64>, %[[c:.+]]: i1) -> !llvm.ptr {
// CHECK-NEXT:  %[[sel:.+]] = arith.select %[[c]], %[[m1]], %[[m2]] : memref<8xf64>
// CHECK-NEXT:  %[[p:.+]] = "enzymexla.memref2pointer"(%[[sel]]) : (memref<8xf64>) -> !llvm.ptr
// CHECK-NEXT:  return %[[p]] : !llvm.ptr
// CHECK-NEXT:  }

// -----

// One arm's conversion outside, the other inside: the if survives the
// trivial-select fold and the if pattern hoists it.

#set1 = affine_set<()[s0] : (s0 >= 1)>
func.func private @mixed_inside_outside(%q1: !llvm.ptr<3>, %q2: !llvm.ptr<3>, %s: index, %i: index) -> f64 {
  %v1 = "enzymexla.pointer2memref"(%q1) : (!llvm.ptr<3>) -> memref<?xf64>
  %a = affine.if #set1()[%s] -> memref<?xf64> {
    affine.yield %v1 : memref<?xf64>
  } else {
    %v2 = "enzymexla.pointer2memref"(%q2) : (!llvm.ptr<3>) -> memref<?xf64>
    affine.yield %v2 : memref<?xf64>
  }
  %x = memref.load %a[%i] : memref<?xf64>
  return %x : f64
}

// CHECK:  func.func private @mixed_inside_outside(%[[q1:.+]]: !llvm.ptr<3>, %[[q2:.+]]: !llvm.ptr<3>, %[[s:.+]]: index, %[[i:.+]]: index) -> f64 {
// CHECK-NEXT:  %[[a:.+]] = affine.if #set()[%[[s]]] -> !llvm.ptr<3> {
// CHECK-NEXT:    affine.yield %[[q1]] : !llvm.ptr<3>
// CHECK-NEXT:  } else {
// CHECK-NEXT:    affine.yield %[[q2]] : !llvm.ptr<3>
// CHECK-NEXT:  }
// CHECK-NEXT:  %[[v:.+]] = "enzymexla.pointer2memref"(%[[a]]) : (!llvm.ptr<3>) -> memref<?xf64>
// CHECK-NEXT:  %[[x:.+]] = memref.load %[[v]][%[[i]]] : memref<?xf64>
// CHECK-NEXT:  return %[[x]] : f64
// CHECK-NEXT:  }

// -----

// An outside conversion kept alive by another use stays; the hoisted one is
// materialized fresh.

func.func private @outside_still_used(%m1: memref<8xf64>, %m2: memref<8xf64>, %c: i1) -> (!llvm.ptr, !llvm.ptr) {
  %p1 = "enzymexla.memref2pointer"(%m1) : (memref<8xf64>) -> !llvm.ptr
  %p2 = "enzymexla.memref2pointer"(%m2) : (memref<8xf64>) -> !llvm.ptr
  %r = scf.if %c -> !llvm.ptr {
    scf.yield %p1 : !llvm.ptr
  } else {
    scf.yield %p2 : !llvm.ptr
  }
  return %r, %p1 : !llvm.ptr, !llvm.ptr
}

// CHECK:  func.func private @outside_still_used(%[[m1:.+]]: memref<8xf64>, %[[m2:.+]]: memref<8xf64>, %[[c:.+]]: i1) -> (!llvm.ptr, !llvm.ptr) {
// CHECK-NEXT:  %[[p1:.+]] = "enzymexla.memref2pointer"(%[[m1]]) : (memref<8xf64>) -> !llvm.ptr
// CHECK-NEXT:  %[[sel:.+]] = arith.select %[[c]], %[[m1]], %[[m2]] : memref<8xf64>
// CHECK-NEXT:  %[[p:.+]] = "enzymexla.memref2pointer"(%[[sel]]) : (memref<8xf64>) -> !llvm.ptr
// CHECK-NEXT:  return %[[p]], %[[p1]] : !llvm.ptr, !llvm.ptr
// CHECK-NEXT:  }
