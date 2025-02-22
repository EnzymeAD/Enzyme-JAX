// RUN: enzymexlamlir-opt --allow-unregistered-dialect --canonicalize-scf-for -split-input-file %s | FileCheck %s

module @if_trunc_2{
  func.func @if_trunc(%arg0: f64, %arg1: i64, %arg2: i64) -> (i1, i1) {
    %cst_8 = arith.constant 1.200000e+03 : f64
    %c0_i32 = arith.constant 0 : i32
    %c2_i32 = arith.constant 2 : i32
    %c1_i64 = arith.constant 1 : i64
    %0 = arith.cmpf une, %arg0, %cst_8 : f64
    %out:2 = scf.if %0 -> (i32, i32) {
      %1 = "if.keepalive"(%arg0, %cst_8) : (f64, f64) -> (f64)
      %2 = arith.addi %arg1, %c1_i64 : i64
      %3 = arith.cmpi slt, %2, %arg2 : i64
      %4 = arith.cmpi sge, %2, %arg2 : i64
      %5 = arith.extui %4 : i1 to i32
      %6 = arith.extui %3 : i1 to i32
      scf.yield %5, %6 : i32, i32
    } else {
      scf.yield %c0_i32, %c0_i32 : i32, i32
    }
    %8 = arith.trunci %out#0 : i32 to i1
    %9 = arith.trunci %out#1 : i32 to i1
    return %8, %9 : i1, i1
  }
}

// CHECK-LABEL:   func.func @if_trunc(
// CHECK-SAME:                        %[[VAL_0:.*]]: f64,
// CHECK-SAME:                        %[[VAL_1:.*]]: i64,
// CHECK-SAME:                        %[[VAL_2:.*]]: i64) -> (i1, i1) {
// CHECK-DAG:             %[[VAL_3:.*]] = arith.constant false
// CHECK-DAG:             %[[VAL_4:.*]] = arith.constant 1.200000e+03 : f64
// CHECK-DAG:             %[[VAL_5:.*]] = arith.constant 1 : i64
// CHECK:           %[[VAL_6:.*]] = arith.cmpf une, %[[VAL_0]], %[[VAL_4]] : f64
// CHECK:           %[[VAL_7:.*]]:2 = scf.if %[[VAL_6]] -> (i1, i1) {
// CHECK:             %[[VAL_8:.*]] = "if.keepalive"(%[[VAL_0]], %[[VAL_4]]) : (f64, f64) -> f64
// CHECK:             %[[VAL_9:.*]] = arith.addi %[[VAL_1]], %[[VAL_5]] : i64
// CHECK:             %[[VAL_10:.*]] = arith.cmpi slt, %[[VAL_9]], %[[VAL_2]] : i64
// CHECK:             %[[VAL_11:.*]] = arith.cmpi sge, %[[VAL_9]], %[[VAL_2]] : i64
// CHECK:             scf.yield %[[VAL_10]], %[[VAL_11]] : i1, i1
// CHECK:           } else {
// CHECK:             scf.yield %[[VAL_3]], %[[VAL_3]] : i1, i1
// CHECK:           }
// CHECK:           return %[[VAL_12:.*]]#1, %[[VAL_12]]#0 : i1, i1
// CHECK:         }