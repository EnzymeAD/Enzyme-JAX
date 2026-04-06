// RUN: enzymexlamlir-opt --canonicalize-scf-for %s | FileCheck %s

module {
  func.func @andi_while_to_for(%arg0: i32, %arg1: i64) -> (i32, i64) {
    %c2560 = arith.constant 2560 : i32
    %c0 = arith.constant 0 : i32
    %result:2 = scf.while (%iv = %c2560, %acc = %arg1) : (i32, i64) -> (i32, i64) {
      %diff = arith.subi %arg0, %c2560 : i32
      %cmp0 = arith.cmpi sle, %iv, %diff : i32
      %next = arith.addi %iv, %c2560 : i32
      %diff2 = arith.subi %arg0, %iv : i32
      %cmp1 = arith.cmpi sge, %diff2, %c2560 : i32
      %cond = arith.andi %cmp0, %cmp1 : i1
      scf.condition(%cond) %next, %acc : i32, i64
    } do {
    ^bb0(%iv: i32, %acc: i64):
      scf.yield %iv, %acc : i32, i64
    }
    return %result#0, %result#1 : i32, i64
  }
}

// CHECK-LABEL: func.func @andi_while_to_for
// CHECK-NOT:   scf.while
// CHECK-DAG:   %[[POISON_I64:.*]] = ub.poison : i64
// CHECK-DAG:   %[[FALSE:.*]] = arith.constant false
// CHECK-DAG:   %[[TRUE:.*]] = arith.constant true
// CHECK-DAG:   %[[POISON_I32:.*]] = ub.poison : i32
// CHECK-DAG:   %[[C1:.*]] = arith.constant 1 : i32
// CHECK-DAG:   %[[C2560:.*]] = arith.constant 2560 : i32
// CHECK-DAG:   %[[CM2560:.*]] = arith.constant -2560 : i32
// CHECK:       %[[BOUND0:.*]] = arith.addi %arg0, %[[CM2560]] : i32
// CHECK:       %[[BOUND:.*]] = arith.addi %[[BOUND0]], %[[C1]] : i32
// CHECK:       %[[FOR:.*]]:5 = scf.for %[[IV:.*]] = %[[C2560]] to %[[BOUND]] step %[[C2560]]
// CHECK-SAME:    iter_args(%[[A0:.*]] = %[[C2560]], %[[A1:.*]] = %arg1, %[[A2:.*]] = %[[POISON_I32]], %[[A3:.*]] = %arg1, %[[A4:.*]] = %[[TRUE]])
// CHECK-SAME:    -> (i32, i64, i32, i64, i1)  : i32 {
// CHECK:         %[[IF1:.*]]:3 = scf.if %[[A4]] -> (i32, i64, i1) {
// CHECK:           %[[NEXT:.*]] = arith.addi %[[A0]], %[[C2560]] : i32
// CHECK:           %[[DIFF:.*]] = arith.subi %arg0, %[[A0]] : i32
// CHECK:           %[[CMP:.*]] = arith.cmpi sge, %[[DIFF]], %[[C2560]] : i32
// CHECK:           scf.yield %[[NEXT]], %[[A1]], %[[CMP]] : i32, i64, i1
// CHECK:         } else {
// CHECK:           scf.yield %[[A2]], %[[A3]], %[[FALSE]] : i32, i64, i1
// CHECK:         }
// CHECK:         %[[CMP2:.*]] = arith.cmpi slt, %[[IV]], %[[BOUND]] : i32
// CHECK:         %[[ANDI:.*]] = arith.andi %[[CMP2]], %[[IF1]]#2 : i1
// CHECK:         %[[IF2:.*]]:2 = scf.if %[[ANDI]] -> (i32, i64) {
// CHECK:           scf.yield %[[IF1]]#0, %[[IF1]]#1 : i32, i64
// CHECK:         } else {
// CHECK:           scf.yield %[[POISON_I32]], %[[POISON_I64]] : i32, i64
// CHECK:         }
// CHECK:         scf.yield %[[IF2]]#0, %[[IF2]]#1, %[[IF1]]#0, %[[IF1]]#1, %[[IF1]]#2
// CHECK:       }
// CHECK:       return %[[FOR]]#2, %[[FOR]]#3 : i32, i64
