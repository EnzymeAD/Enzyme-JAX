// RUN: enzymexlamlir-opt --allow-unregistered-dialect --canonicalize-scf-for --split-input-file %s | FileCheck %s

func.func private @"reduced_example"() {
  %true = arith.constant true
  %c1_i64 = arith.constant 1 : i64
  %c30_i64 = arith.constant 30 : i64
  %c0_i64 = arith.constant 0 : i64
  %cst_58 = arith.constant 3.000000e-05 : f64

  %115:2 = scf.while (%arg8 = %c0_i64) : (i64) -> (i64, f64) {
    // Some computation that produces %278 and %280
    %278 = "test.def"(%arg8) : (i64) -> (f64)
    %280 = arith.addi %arg8, %c1_i64 : i64

    // The key condition we're interested in:
    %279 = arith.cmpf olt, %278, %cst_58 : f64
    %281 = arith.cmpi ult, %280, %c1_i64 : i64
    %282 = arith.cmpi sgt, %280, %c30_i64 : i64
    %283 = arith.ori %281, %282 : i1
    %284 = arith.xori %283, %true : i1
    %285 = arith.andi %284, %279 : i1

    scf.condition(%285) %280, %278 : i64, f64
  } do {
  ^bb0(%arg8: i64, %arg9: f64):
    scf.yield %arg8 : i64
  }

  return
}

// CHECK-LABEL: func @reduced_example
// CHECK-DAG:     %[[C1:.+]] = arith.constant 1 : i64
// CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : i64
// CHECK-DAG:     %[[C31:.+]] = arith.constant 31 : i64
// CHECK-DAG:     %[[FALSE:.+]] = arith.constant false
// CHECK-DAG:     %[[CST:.+]] = arith.constant 3.000000e-05 : f64
// CHECK-DAG:     %[[POISON_I64:.+]] = ub.poison : i64
// CHECK-DAG:     %[[POISON_F64:.+]] = ub.poison : f64
// CHECK-DAG:     %[[TRUE:.+]] = arith.constant true
// CHECK-DAG:     %[[C32:.+]] = arith.constant 32 : i64
// CHECK:         %[[FOR:.+]]:4 = scf.for %[[I:.+]] = %[[C1]] to %[[C32]] step %[[C1]] iter_args(
// CHECK-SAME:    %[[ARG1:.+]] = %[[C0]],
// CHECK-SAME:    %[[ARG2:.+]] = %[[POISON_I64]],
// CHECK-SAME:    %[[ARG3:.+]] = %[[POISON_F64]],
// CHECK-SAME:    %[[ARG4:.+]] = %[[TRUE]]) -> (i64, i64, f64, i1) : i64 {
// CHECK:           %[[IF1:.+]]:3 = scf.if %[[ARG4]] -> (i64, f64, i1) {
// CHECK:             %[[DEF:.+]] = "test.def"(%[[ARG1]])
// CHECK:             %[[ADD:.+]] = arith.addi %[[ARG1]], %[[C1]]
// CHECK:             %[[CMPF:.+]] = arith.cmpf olt, %[[DEF]], %[[CST]]
// CHECK:             %[[CMPI:.+]] = arith.cmpi ult, %[[ADD]], %[[C1]]
// CHECK:             %[[XOR:.+]] = arith.xori %[[CMPI]], %[[TRUE]]
// CHECK:             %[[AND:.+]] = arith.andi %[[CMPF]], %[[XOR]]
// CHECK:             scf.yield %[[ADD]], %[[DEF]], %[[AND]]
// CHECK:           } else {
// CHECK:             scf.yield %[[ARG2]], %[[ARG3]], %[[FALSE]]
// CHECK:           }
// CHECK:           %[[BOUND_CHECK:.+]] = arith.cmpi slt, %[[I]], %[[C31]]
// CHECK:           %[[BREAK_COND:.+]] = arith.andi %[[BOUND_CHECK]], %[[IF1]]#2
// CHECK:           %[[IF2:.+]] = scf.if %[[BREAK_COND]] -> (i64) {
// CHECK:             scf.yield %[[IF1]]#0
// CHECK:           } else {
// CHECK:             scf.yield %[[POISON_I64]]
// CHECK:           }
// CHECK:           scf.yield %[[IF2]], %[[IF1]]#0, %[[IF1]]#1, %[[IF1]]#2
// CHECK:         }

// ----

func.func private @"negative_step"() {
  %true = arith.constant true
  %c1_i64 = arith.constant 1 : i64
  %c30_i64 = arith.constant 30 : i64
  %c-1_i64 = arith.constant -1 : i64
  %cst_58 = arith.constant 3.000000e-05 : f64
  %c29_i64 = arith.constant 29 : i64

  %115:2 = scf.while (%arg8 = %c29_i64) : (i64) -> (i64, f64) {
    // Some computation that produces %278 and %280
    %278 = "test.def"(%arg8) : (i64) -> (f64)
    %280 = arith.addi %arg8, %c-1_i64 : i64

    // The key condition we're interested in:
    %279 = arith.cmpf olt, %278, %cst_58 : f64
    %281 = arith.cmpi ult, %280, %c1_i64 : i64
    %282 = arith.cmpi sgt, %280, %c30_i64 : i64
    %283 = arith.ori %281, %282 : i1
    %284 = arith.xori %283, %true : i1
    %285 = arith.andi %284, %279 : i1

    scf.condition(%285) %280, %278 : i64, f64
  } do {
  ^bb0(%arg8: i64, %arg9: f64):
    scf.yield %arg8 : i64
  }

  return
}

// CHECK-LABEL: func @negative_step
// CHECK-DAG:     %[[C1:.+]] = arith.constant 1 : i64
// CHECK-DAG:     %[[CM1:.+]] = arith.constant -1 : i64
// CHECK-DAG:     %[[C30:.+]] = arith.constant 30 : i64
// CHECK-DAG:     %[[FALSE:.+]] = arith.constant false
// CHECK-DAG:     %[[CST:.+]] = arith.constant 3.000000e-05 : f64
// CHECK-DAG:     %[[POISON_I64:.+]] = ub.poison : i64
// CHECK-DAG:     %[[POISON_F64:.+]] = ub.poison : f64
// CHECK-DAG:     %[[TRUE:.+]] = arith.constant true
// CHECK-DAG:     %[[C29:.+]] = arith.constant 29 : i64
// CHECK:         %[[FOR:.+]]:4 = scf.for %[[I:.+]] = %[[C1]] to %[[C30]] step %[[C1]] iter_args(
// CHECK-SAME:    %[[ARG1:.+]] = %[[C29]],
// CHECK-SAME:    %[[ARG2:.+]] = %[[POISON_I64]],
// CHECK-SAME:    %[[ARG3:.+]] = %[[POISON_F64]],
// CHECK-SAME:    %[[ARG4:.+]] = %[[TRUE]]) -> (i64, i64, f64, i1) : i64 {
// CHECK:           %[[IF1:.+]]:3 = scf.if %[[ARG4]] -> (i64, f64, i1) {
// CHECK:             %[[DEF:.+]] = "test.def"(%[[ARG1]])
// CHECK:             %[[ADD:.+]] = arith.addi %[[ARG1]], %[[CM1]]
// CHECK:             %[[CMPF:.+]] = arith.cmpf olt, %[[DEF]], %[[CST]]
// CHECK:             %[[CMPI:.+]] = arith.cmpi sgt, %[[ADD]], %[[C30]]
// CHECK:             %[[XOR:.+]] = arith.xori %[[CMPI]], %[[TRUE]]
// CHECK:             %[[AND:.+]] = arith.andi %[[CMPF]], %[[XOR]]
// CHECK:             scf.yield %[[ADD]], %[[DEF]], %[[AND]]
// CHECK:           } else {
// CHECK:             scf.yield %[[ARG2]], %[[ARG3]], %[[FALSE]]
// CHECK:           }
// CHECK:           %[[BOUND_CHECK:.+]] = arith.cmpi slt, %[[I]], %[[C29]]
// CHECK:           %[[BREAK_COND:.+]] = arith.andi %[[BOUND_CHECK]], %[[IF1]]#2
// CHECK:           %[[IF2:.+]] = scf.if %[[BREAK_COND]] -> (i64) {
// CHECK:             scf.yield %[[IF1]]#0
// CHECK:           } else {
// CHECK:             scf.yield %[[POISON_I64]]
// CHECK:           }
// CHECK:           scf.yield %[[IF2]], %[[IF1]]#0, %[[IF1]]#1, %[[IF1]]#2
// CHECK:         }
