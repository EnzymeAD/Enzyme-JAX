// RUN: enzymexlamlir-opt %s --canonicalize-scf-for | FileCheck %s

// The rotated (do-while) halving loop of a block tree reduction: the body,
// the shift, and the nonzero test all sit in the before region, with the
// shifted value forwarded as the carried state. WhileShiftToInduction only
// matches the canonical body-in-after form, so this shape converts here: a
// log-space scf.for over max(bitwidth - ctlz(start), 1) trips with
// i = start >> k, running the body once even for a zero start.
func.func @treereduce(%bs: i32, %tid: i32, %buf: memref<?xf64>) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %start = arith.shrui %bs, %c1_i32 : i32
  %r = scf.while (%i = %start) : (i32) -> i32 {
    %active = arith.cmpi ult, %tid, %i : i32
    scf.if %active {
      %pos = arith.addi %i, %tid overflow<nsw, nuw> : i32
      %pi = arith.index_cast %pos : i32 to index
      %ti = arith.index_cast %tid : i32 to index
      %a = memref.load %buf[%ti] : memref<?xf64>
      %b = memref.load %buf[%pi] : memref<?xf64>
      %s = arith.addf %a, %b : f64
      memref.store %s, %buf[%ti] : memref<?xf64>
    }
    %next = arith.shrui %i, %c1_i32 : i32
    %nz = arith.cmpi ne, %next, %c0_i32 : i32
    scf.condition(%nz) %next : i32
  } do {
  ^bb0(%a: i32):
    scf.yield %a : i32
  }
  return
}

// CHECK-LABEL: func.func @treereduce(
// CHECK-SAME: %[[BS:.+]]: i32, %[[TID:.+]]: i32, %[[BUF:.+]]: memref<?xf64>
// CHECK-NEXT: %[[V0:.+]] = arith.constant 1 : index
// CHECK-NEXT: %[[V1:.+]] = arith.constant 0 : index
// CHECK-NEXT: %[[V2:.+]] = arith.constant 32 : i32
// CHECK-NEXT: %[[V3:.+]] = arith.constant 1 : i32
// CHECK-NEXT: %[[V4:.+]] = arith.shrui %[[BS]], %[[V3]] : i32
// CHECK-NEXT: %[[V5:.+]] = math.ctlz %[[V4]] : i32
// CHECK-NEXT: %[[V6:.+]] = arith.subi %[[V2]], %[[V5]] : i32
// CHECK-NEXT: %[[V7:.+]] = arith.maxui %[[V6]], %[[V3]] : i32
// CHECK-NEXT: %[[V8:.+]] = arith.index_castui %[[V7]] : i32 to index
// CHECK-NEXT: scf.for %[[V9:.+]] = %[[V1]] to %[[V8]] step %[[V0]] {
// CHECK-NEXT: %[[V10:.+]] = arith.index_castui %[[V9]] : index to i32
// CHECK-NEXT: %[[V11:.+]] = arith.shrui %[[V4]], %[[V10]] : i32
// CHECK-NEXT: %[[V12:.+]] = arith.cmpi ult, %[[TID]], %[[V11]] : i32
// CHECK-NEXT: scf.if %[[V12]] {
// CHECK-NEXT: %[[V13:.+]] = arith.addi %[[V11]], %[[TID]] overflow<nsw, nuw> : i32
// CHECK-NEXT: %[[V14:.+]] = arith.index_cast %[[V13]] : i32 to index
// CHECK-NEXT: %[[V15:.+]] = arith.index_cast %[[TID]] : i32 to index
// CHECK-NEXT: %[[V16:.+]] = memref.load %[[BUF]][%[[V15]]] : memref<?xf64>
// CHECK-NEXT: %[[V17:.+]] = memref.load %[[BUF]][%[[V14]]] : memref<?xf64>
// CHECK-NEXT: %[[V18:.+]] = arith.addf %[[V16]], %[[V17]] : f64
// CHECK-NEXT: memref.store %[[V18]], %[[BUF]][%[[V15]]] : memref<?xf64>
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: return
// CHECK-NEXT: }

// The same rotated shape with a carried counter, where the condition forwards
// its operands in a different order than the before arguments and the after
// region's yield permutes them back: the shifted variable is condition
// operand 1 but before argument 0, so the before and after index spaces must
// be translated through the after yield.
func.func @treereduce_swap(%bs: i32) -> (i32, i32) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %start = arith.shrui %bs, %c1_i32 : i32
  %r:2 = scf.while (%i = %start, %count = %c0_i32) : (i32, i32) -> (i32, i32) {
    %c2 = arith.addi %count, %c1_i32 : i32
    %next = arith.shrui %i, %c1_i32 : i32
    %nz = arith.cmpi ne, %next, %c0_i32 : i32
    scf.condition(%nz) %c2, %next : i32, i32
  } do {
  ^bb0(%cc: i32, %a: i32):
    scf.yield %a, %cc : i32, i32
  }
  return %r#0, %r#1 : i32, i32
}

// CHECK-LABEL: func.func @treereduce_swap(
// CHECK-SAME: %[[SBS:.+]]: i32
// CHECK-NEXT: %[[S0:.+]] = arith.constant 0 : i32
// CHECK-NEXT: %[[S32:.+]] = arith.constant 32 : i32
// CHECK-NEXT: %[[S1:.+]] = arith.constant 1 : i32
// CHECK-NEXT: %[[SV0:.+]] = arith.shrui %[[SBS]], %[[S1]] : i32
// CHECK-NEXT: %[[SV1:.+]] = math.ctlz %[[SV0]] : i32
// CHECK-NEXT: %[[SV2:.+]] = arith.subi %[[S32]], %[[SV1]] : i32
// CHECK-NEXT: %[[SV3:.+]] = arith.maxui %[[SV2]], %[[S1]] : i32
// CHECK-NEXT: %[[SV4:.+]] = arith.index_castui %[[SV3]] : i32 to index
// CHECK-NEXT: %[[SV5:.+]] = arith.index_cast %[[SV4]] : index to i32
// CHECK-NEXT: return %[[SV5]], %[[S0]] : i32, i32
// CHECK-NEXT: }

// An index-typed halving loop: index has no fixed bit width, so the trip
// count is computed in i64 (which index_castui zero-extends into, keeping the
// bit length right for any narrower index lowering), the for's induction
// variable feeds the shift without a cast, and the exit value is a constant
// index zero.
func.func @treereduce_index(%bs: index, %buf: memref<?xf64>) -> index {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cf = arith.constant 1.000000e+00 : f64
  %start = arith.shrui %bs, %c1 : index
  %r = scf.while (%i = %start) : (index) -> index {
    memref.store %cf, %buf[%i] : memref<?xf64>
    %next = arith.shrui %i, %c1 : index
    %nz = arith.cmpi ne, %next, %c0 : index
    scf.condition(%nz) %next : index
  } do {
  ^bb0(%a: index):
    scf.yield %a : index
  }
  return %r : index
}

// CHECK-LABEL: func.func @treereduce_index(
// CHECK-SAME: %[[IBS:.+]]: index, %[[IBUF:.+]]: memref<?xf64>
// CHECK-NEXT: %[[I0:.+]] = arith.constant 0 : index
// CHECK-NEXT: %[[I1I64:.+]] = arith.constant 1 : i64
// CHECK-NEXT: %[[I64C:.+]] = arith.constant 64 : i64
// CHECK-NEXT: %[[I1:.+]] = arith.constant 1 : index
// CHECK-NEXT: %[[ICF:.+]] = arith.constant 1.000000e+00 : f64
// CHECK-NEXT: %[[IV0:.+]] = arith.shrui %[[IBS]], %[[I1]] : index
// CHECK-NEXT: %[[IV1:.+]] = arith.index_castui %[[IV0]] : index to i64
// CHECK-NEXT: %[[IV2:.+]] = math.ctlz %[[IV1]] : i64
// CHECK-NEXT: %[[IV3:.+]] = arith.subi %[[I64C]], %[[IV2]] : i64
// CHECK-NEXT: %[[IV4:.+]] = arith.maxui %[[IV3]], %[[I1I64]] : i64
// CHECK-NEXT: %[[IV5:.+]] = arith.index_castui %[[IV4]] : i64 to index
// CHECK-NEXT: scf.for %[[IK:.+]] = %[[I0]] to %[[IV5]] step %[[I1]] {
// CHECK-NEXT: %[[IV6:.+]] = arith.shrui %[[IV0]], %[[IK]] : index
// CHECK-NEXT: memref.store %[[ICF]], %[[IBUF]][%[[IV6]]] : memref<?xf64>
// CHECK-NEXT: }
// CHECK-NEXT: return %[[I0]] : index
// CHECK-NEXT: }
