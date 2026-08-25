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
