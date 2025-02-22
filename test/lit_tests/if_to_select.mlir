// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(func.func(canonicalize-loops))" --split-input-file | FileCheck %s

func.func @if_to_select(%arg0: i1, %arg1: i32, %arg2: i32) -> i32 {
  %result = scf.if %arg0 -> i32 {
    scf.yield %arg1 : i32
  } else {
    scf.yield %arg2 : i32
  }
  return %result : i32
}

// CHECK-LABEL: func @if_to_select
// CHECK-SAME: (%[[ARG0:.*]]: i1, %[[ARG1:.*]]: i32, %[[ARG2:.*]]: i32)
// CHECK: %[[SELECT:.*]] = arith.select %[[ARG0]], %[[ARG1]], %[[ARG2]] : i32
// CHECK: return %[[SELECT]] : i32

// -----

func.func @if_to_select_multi_result(
    %arg0: i1, %arg1: i64, %arg2: i32, 
    %arg3: i32, %arg4: i64, %arg5: i32, %arg6: i32) -> (i64, i32, i32) {
  %result:3 = scf.if %arg0 -> (i64, i32, i32) {
    scf.yield %arg1, %arg2, %arg3 : i64, i32, i32
  } else {
    scf.yield %arg4, %arg5, %arg6 : i64, i32, i32
  }
  return %result#0, %result#1, %result#2 : i64, i32, i32
}

// CHECK-LABEL: func @if_to_select_multi_result
// CHECK-SAME: (%[[ARG0:.*]]: i1, %[[ARG1:.*]]: i64, %[[ARG2:.*]]: i32, %[[ARG3:.*]]: i32, %[[ARG4:.*]]: i64, %[[ARG5:.*]]: i32, %[[ARG6:.*]]: i32)
// CHECK: %[[SELECT0:.*]] = arith.select %[[ARG0]], %[[ARG1]], %[[ARG4]] : i64
// CHECK: %[[SELECT1:.*]] = arith.select %[[ARG0]], %[[ARG2]], %[[ARG5]] : i32
// CHECK: %[[SELECT2:.*]] = arith.select %[[ARG0]], %[[ARG3]], %[[ARG6]] : i32
// CHECK: return %[[SELECT0]], %[[SELECT1]], %[[SELECT2]] : i64, i32, i32   

// -----

func.func @if_to_select_nested(%cond: i1, %val1: f64, %val2: f64, %const: f64, %const2: f64) -> f64 {
  %result = scf.if %cond -> (f64) {
    %cmp = arith.cmpf ogt, %val1, %const {fastmathFlags = #llvm.fastmath<none>} : f64
    %sel = arith.select %cmp, %val1, %val2 {fastmathFlags = #llvm.fastmath<none>} : f64
    scf.yield %sel : f64
  } else {
    %copy = math.copysign %val1, %const2 : f64
    scf.yield %copy : f64
  }
  return %result : f64
}

// CHECK-LABEL: func @if_to_select_nested
// CHECK-SAME: (%[[COND:.*]]: i1, %[[VAL1:.*]]: f64, %[[VAL2:.*]]: f64, %[[CONST:.*]]: f64, %[[CONST2:.*]]: f64)
// CHECK: %[[CMP:.*]] = arith.cmpf ogt, %[[VAL1]], %[[CONST]] {fastmathFlags = #llvm.fastmath<none>} : f64
// CHECK: %[[SEL1:.*]] = arith.select %[[CMP]], %[[VAL1]], %[[VAL2]] {fastmathFlags = #llvm.fastmath<none>} : f64
// CHECK: %[[COPY:.*]] = math.copysign %[[VAL1]], %[[CONST2]] : f64
// CHECK: %[[SEL2:.*]] = arith.select %[[COND]], %[[SEL1]], %[[COPY]] : f64
// CHECK: return %[[SEL2]] : f64   