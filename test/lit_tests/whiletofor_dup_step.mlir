// RUN: enzymexlamlir-opt %s --canonicalize-scf-for | FileCheck %s

// The rotated grid-stride reduction loop: the exit is a conjunction of the
// induction bound and a data-dependent in-bounds test, and clang emits the
// increment twice -- the condition passes one copy while the compare uses
// the other. The structurally identical add must not block the conversion.
func.func @gridstride(%N: i32, %ipt: i32, %bs: i32, %tid: i32, %woff: i32, %buf: memref<?xf64>) -> f64 {
  %cst = arith.constant 0.0 : f64
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %r:2 = scf.while (%acc = %cst, %idx = %c0_i32) : (f64, i32) -> (f64, i32) {
    %t0 = arith.addi %idx, %woff overflow<nsw> : i32
    %t1 = arith.muli %t0, %bs : i32
    %i = arith.addi %t1, %tid : i32
    %inb = arith.cmpi slt, %i, %N : i32
    %next = arith.addi %idx, %c1_i32 overflow<nsw, nuw> : i32
    %next2 = arith.addi %idx, %c1_i32 overflow<nsw, nuw> : i32
    %ne = arith.cmpi ne, %next2, %ipt : i32
    %acc2 = scf.if %inb -> (f64) {
      %ii = arith.index_cast %i : i32 to index
      %v = memref.load %buf[%ii] : memref<?xf64>
      %s = arith.addf %acc, %v : f64
      scf.yield %s : f64
    } else {
      scf.yield %acc : f64
    }
    %cond = arith.andi %inb, %ne : i1
    scf.condition(%cond) %acc2, %next : f64, i32
  } do {
  ^bb0(%a: f64, %i2: i32):
    scf.yield %a, %i2 : f64, i32
  }
  return %r#0 : f64
}

// CHECK-LABEL: func.func @gridstride(
// CHECK-SAME: %[[N:.+]]: i32, %[[IPT:.+]]: i32, %[[BS:.+]]: i32, %[[TID:.+]]: i32, %[[WOFF:.+]]: i32, %[[BUF:.+]]: memref<?xf64>
// CHECK-NEXT: %[[FALSE:.+]] = arith.constant false
// CHECK-NEXT: %[[CST:.+]] = arith.constant 0.000000e+00 : f64
// CHECK-NEXT: %[[C0:.+]] = arith.constant 0 : i32
// CHECK-NEXT: %[[C1:.+]] = arith.constant 1 : i32
// CHECK-NEXT: %[[PF:.+]] = ub.poison : f64
// CHECK-NEXT: %[[PI:.+]] = ub.poison : i32
// CHECK-NEXT: %[[TRUE:.+]] = arith.constant true
// CHECK-NEXT: %[[MAX:.+]] = arith.maxsi %[[IPT]], %[[C1]] : i32
// CHECK-NEXT: %[[UB:.+]] = arith.addi %[[MAX]], %[[C1]] : i32
// CHECK-NEXT: %[[FOR:.+]]:5 = scf.for %[[IV:.+]] = %[[C1]] to %[[UB]] step %[[C1]] iter_args(%[[ACC:.+]] = %[[CST]], %[[IDX:.+]] = %[[C0]], %[[SHF:.+]] = %[[PF]], %[[SHI:.+]] = %[[PI]], %[[LIVE:.+]] = %[[TRUE]]) -> (f64, i32, f64, i32, i1)  : i32 {
// CHECK-NEXT: %[[BODY:.+]]:3 = scf.if %[[LIVE]] -> (f64, i32, i1) {
// CHECK-NEXT: %[[T0:.+]] = arith.addi %[[IDX]], %[[WOFF]] overflow<nsw> : i32
// CHECK-NEXT: %[[T1:.+]] = arith.muli %[[T0]], %[[BS]] : i32
// CHECK-NEXT: %[[I:.+]] = arith.addi %[[T1]], %[[TID]] : i32
// CHECK-NEXT: %[[INB:.+]] = arith.cmpi slt, %[[I]], %[[N]] : i32
// CHECK-NEXT: %[[NEXT:.+]] = arith.addi %[[IDX]], %[[C1]] overflow<nsw, nuw> : i32
// CHECK-NEXT: %[[SUM:.+]] = scf.if %[[INB]] -> (f64) {
// CHECK-NEXT: %[[II:.+]] = arith.index_cast %[[I]] : i32 to index
// CHECK-NEXT: %[[V:.+]] = memref.load %[[BUF]][%[[II]]] : memref<?xf64>
// CHECK-NEXT: %[[ADD:.+]] = arith.addf %[[ACC]], %[[V]] : f64
// CHECK-NEXT: scf.yield %[[ADD]] : f64
// CHECK-NEXT: } else {
// CHECK-NEXT: scf.yield %[[ACC]] : f64
// CHECK-NEXT: }
// CHECK-NEXT: scf.yield %[[SUM]], %[[NEXT]], %[[INB]] : f64, i32, i1
// CHECK-NEXT: } else {
// CHECK-NEXT: scf.yield %[[SHF]], %[[SHI]], %[[FALSE]] : f64, i32, i1
// CHECK-NEXT: }
// CHECK-NEXT: %[[MORE:.+]] = arith.cmpi slt, %[[IV]], %[[IPT]] : i32
// CHECK-NEXT: %[[CONT:.+]] = arith.andi %[[MORE]], %[[BODY]]#2 : i1
// CHECK-NEXT: %[[RES:.+]]:2 = scf.if %[[CONT]] -> (f64, i32) {
// CHECK-NEXT: scf.yield %[[BODY]]#0, %[[BODY]]#1 : f64, i32
// CHECK-NEXT: } else {
// CHECK-NEXT: scf.yield %[[PF]], %[[PI]] : f64, i32
// CHECK-NEXT: }
// CHECK-NEXT: scf.yield %[[RES]]#0, %[[RES]]#1, %[[BODY]]#0, %[[BODY]]#1, %[[BODY]]#2 : f64, i32, f64, i32, i1
// CHECK-NEXT: }
// CHECK-NEXT: return %[[FOR]]#2 : f64
