// RUN: enzymexlamlir-opt %s --cse --canonicalize-scf-for | FileCheck %s

// The while-converted grid-stride loop carries an i1 "live" flag: true on
// entry, re-computed as `live ? f(counter) < N : false` with the counter
// advancing once per live iteration. Every earlier test held while the flag
// is true and f is nondecreasing in the counter, so the flag is just the
// previous iteration's test: `iv == lb || f(cnt0 + iv - lb - 1) < N`, a pure
// function of the induction variable, and the flag and counter fold away.
func.func @gridstride(%N: i32, %ipt: i32, %bsi: index, %tid: i32, %woff: i32, %buf: memref<?xf64>) -> f64 {
  %bs = arith.index_castui %bsi : index to i32
  %cst = arith.constant 0.0 : f64
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %r:2 = scf.while (%acc = %cst, %idx = %c0_i32) : (f64, i32) -> (f64, i32) {
    %t0 = arith.addi %idx, %woff overflow<nsw> : i32
    %t1 = arith.muli %t0, %bs : i32
    %i = arith.addi %t1, %tid : i32
    %inb = arith.cmpi slt, %i, %N : i32
    %next = arith.addi %idx, %c1_i32 overflow<nsw, nuw> : i32
    %ne = arith.cmpi ne, %next, %ipt : i32
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
// CHECK-SAME: %[[N:.+]]: i32, %[[IPT:.+]]: i32, %[[BSI:.+]]: index, %[[TID:.+]]: i32, %[[WOFF:.+]]: i32, %[[BUF:.+]]: memref<?xf64>
// CHECK-NEXT: %[[V0:.+]] = arith.constant false
// CHECK-NEXT: %[[V1:.+]] = arith.constant -1 : i32
// CHECK-NEXT: %[[V2:.+]] = ub.poison : i32
// CHECK-NEXT: %[[V3:.+]] = ub.poison : f64
// CHECK-NEXT: %[[V4:.+]] = arith.constant 1 : i32
// CHECK-NEXT: %[[V5:.+]] = arith.constant 0.000000e+00 : f64
// CHECK-NEXT: %[[V6:.+]] = arith.index_castui %[[BSI]] : index to i32
// CHECK-NEXT: %[[V7:.+]] = arith.maxsi %[[IPT]], %[[V4]] : i32
// CHECK-NEXT: %[[V8:.+]] = arith.addi %[[V7]], %[[V4]] : i32
// CHECK-NEXT: %[[V9:.+]]:3 = scf.for %[[V10:.+]] = %[[V4]] to %[[V8]] step %[[V4]] iter_args(%[[V11:.+]] = %[[V5]], %[[V12:.+]] = %[[V3]], %[[V13:.+]] = %[[V2]]) -> (f64, f64, i32)  : i32 {
// CHECK-NEXT: %[[V14:.+]] = arith.addi %[[V10]], %[[V1]] : i32
// CHECK-NEXT: %[[V15:.+]] = arith.addi %[[V14]], %[[V1]] : i32
// CHECK-NEXT: %[[V16:.+]] = arith.addi %[[V15]], %[[WOFF]] : i32
// CHECK-NEXT: %[[V17:.+]] = arith.muli %[[V16]], %[[V6]] : i32
// CHECK-NEXT: %[[V18:.+]] = arith.addi %[[V17]], %[[TID]] : i32
// CHECK-NEXT: %[[V19:.+]] = arith.cmpi slt, %[[V18]], %[[N]] : i32
// CHECK-NEXT: %[[V20:.+]] = arith.cmpi eq, %[[V10]], %[[V4]] : i32
// CHECK-NEXT: %[[V21:.+]] = arith.ori %[[V20]], %[[V19]] : i1
// CHECK-NEXT: %[[V22:.+]]:3 = scf.if %[[V21]] -> (f64, i32, i1) {
// CHECK-NEXT: %[[V23:.+]] = arith.addi %[[V14]], %[[WOFF]] overflow<nsw> : i32
// CHECK-NEXT: %[[V24:.+]] = arith.muli %[[V23]], %[[V6]] : i32
// CHECK-NEXT: %[[V25:.+]] = arith.addi %[[V24]], %[[TID]] : i32
// CHECK-NEXT: %[[V26:.+]] = arith.cmpi slt, %[[V25]], %[[N]] : i32
// CHECK-NEXT: %[[V27:.+]] = arith.addi %[[V14]], %[[V4]] overflow<nsw, nuw> : i32
// CHECK-NEXT: %[[V28:.+]] = scf.if %[[V26]] -> (f64) {
// CHECK-NEXT: %[[V29:.+]] = arith.index_cast %[[V25]] : i32 to index
// CHECK-NEXT: %[[V30:.+]] = memref.load %[[BUF]][%[[V29]]] : memref<?xf64>
// CHECK-NEXT: %[[V31:.+]] = arith.addf %[[V11]], %[[V30]] : f64
// CHECK-NEXT: scf.yield %[[V31]] : f64
// CHECK-NEXT: } else {
// CHECK-NEXT: scf.yield %[[V11]] : f64
// CHECK-NEXT: }
// CHECK-NEXT: scf.yield %[[V28]], %[[V27]], %[[V26]] : f64, i32, i1
// CHECK-NEXT: } else {
// CHECK-NEXT: scf.yield %[[V12]], %[[V13]], %[[V0]] : f64, i32, i1
// CHECK-NEXT: }
// CHECK-NEXT: %[[V32:.+]] = arith.cmpi slt, %[[V10]], %[[IPT]] : i32
// CHECK-NEXT: %[[V33:.+]] = arith.andi %[[V32]], %[[V22]]#2 : i1
// CHECK-NEXT: %[[V34:.+]] = scf.if %[[V33]] -> (f64) {
// CHECK-NEXT: scf.yield %[[V22]]#0 : f64
// CHECK-NEXT: } else {
// CHECK-NEXT: scf.yield %[[V3]] : f64
// CHECK-NEXT: }
// CHECK-NEXT: scf.yield %[[V34]], %[[V22]]#0, %[[V22]]#1 : f64, f64, i32
// CHECK-NEXT: }
// CHECK-NEXT: return %[[V9]]#1 : f64
// CHECK-NEXT: }

// -----

// Near-miss: the counter advances by two per iteration, not the `cnt + 1` the
// fold recognizes. The flag is not a pure function of the induction variable,
// so the pattern must not fire and the i1 "live" flag stays a carried value.
func.func @counter_step_two(%N: i32, %ipt: i32, %bsi: index, %tid: i32, %woff: i32, %buf: memref<?xf64>) -> f64 {
  %bs = arith.index_castui %bsi : index to i32
  %cst = arith.constant 0.0 : f64
  %c0_i32 = arith.constant 0 : i32
  %c2_i32 = arith.constant 2 : i32
  %r:2 = scf.while (%acc = %cst, %idx = %c0_i32) : (f64, i32) -> (f64, i32) {
    %t0 = arith.addi %idx, %woff overflow<nsw> : i32
    %t1 = arith.muli %t0, %bs : i32
    %i = arith.addi %t1, %tid : i32
    %inb = arith.cmpi slt, %i, %N : i32
    %next = arith.addi %idx, %c2_i32 overflow<nsw, nuw> : i32
    %ne = arith.cmpi ne, %next, %ipt : i32
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
// The flag stays a loop-carried i1 and the induction-variable predicate fold
// (an arith.ori of `iv == lb` with the test) never appears.
// CHECK-LABEL: func.func @counter_step_two(
// CHECK: scf.for {{.*}} -> (f64, i32, f64, i32, i1)
// CHECK-NOT: arith.ori
