// RUN: enzymexlamlir-opt -allow-unregistered-dialect --canonicalize-scf-for --split-input-file %s | FileCheck %s

module {
  func.func @w2f(%ub : i32) -> (i32, f32) {
    %cst = arith.constant 0.000000e+00 : f32
    %cst1 = arith.constant 1.000000e+00 : f32
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %true = arith.constant true
    %2:2 = scf.while (%arg10 = %c0_i32, %arg12 = %cst, %ac = %true) : (i32, f32, i1) -> (i32, f32) {
      %3 = arith.cmpi ult, %arg10, %ub : i32
      %a = arith.andi %3, %ac : i1
      scf.condition(%a) %arg10, %arg12 : i32, f32
    } do {
    ^bb0(%arg10: i32, %arg12: f32):
      %c = "test.something"() : () -> (i1)
      %3 = arith.addf %arg12, %cst1 : f32
      %p = arith.addi %arg10, %c1_i32 : i32
      scf.yield %p, %3, %c : i32, f32, i1
    }
    return %2#0, %2#1 : i32, f32
  }

  func.func @w2f_inner(%ub : i32) -> (i32, f32) {
    %cst = arith.constant 0.000000e+00 : f32
    %cst1 = arith.constant 1.000000e+00 : f32
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %true = arith.constant true
    %2:2 = scf.while (%arg10 = %c0_i32, %arg12 = %cst, %ac = %true) : (i32, f32, i1) -> (i32, f32) {
      %3 = arith.cmpi ult, %arg10, %ub : i32
      %a = arith.andi %3, %ac : i1
      scf.condition(%a) %arg10, %arg12 : i32, f32
    } do {
    ^bb0(%arg10: i32, %arg12: f32):
      %c = "test.something"() : () -> (i1)
      %r:2 = scf.if %c -> (i32, f32) {
        %3 = arith.addf %arg12, %cst1 : f32
        %p = arith.addi %arg10, %c1_i32 : i32
        scf.yield %p, %3 : i32, f32
      } else {
        scf.yield %arg10, %arg12 : i32, f32
      }
      scf.yield %r#0, %r#1, %c : i32, f32, i1
    }
    return %2#0, %2#1 : i32, f32
  }

  func.func @_Z17compute_tran_tempPfPS_iiiiiiii(%arg0: i8, %arg1: index, %arg2: i32, %arg3: i32, %arg4: i32) -> i32 {
    %c1_i8 = arith.constant 1 : i8
    %c0_i8 = arith.constant 0 : i8
    %true = arith.constant true
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %0:2 = scf.while (%arg5 = %c0_i32, %arg6 = %arg0, %arg7 = %true) : (i32, i8, i1) -> (i8, i32) {
      %1 = arith.cmpi slt, %arg5, %arg2 : i32
      %2 = arith.andi %1, %arg7 : i1
      scf.condition(%2) %arg6, %arg5 : i8, i32
    } do {
    ^bb0(%arg5: i8, %arg6: i32):
      %1 = arith.addi %arg6, %c1_i32 : i32
      %2 = arith.cmpi ne, %arg6, %arg4 : i32
      %3 = scf.if %2 -> (i32) {
        scf.yield %1 : i32
      } else {
        scf.yield %arg6 : i32
      }
      scf.yield %3, %c0_i8, %2 : i32, i8, i1
    }
    return %0#1 : i32
  }
}
// CHECK-LABEL:   func.func @w2f(
// CHECK-SAME:                   %[[ub:.*]]: i32) -> (i32, f32) {
// CHECK-DAG:           %[[undef_f32:.+]] = ub.poison : f32
// CHECK-DAG:           %[[undef_i32:.+]] = ub.poison : i32
// CHECK-DAG:           %[[true:.*]] = arith.constant true
// CHECK-DAG:           %[[false:.*]] = arith.constant false
// CHECK-DAG:           %[[cst0:.*]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:           %[[cst1:.*]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG:           %[[c0:.*]] = arith.constant 0 : i32
// CHECK-DAG:           %[[c1:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_7:.*]] = arith.cmpi ugt, %[[VAL_0]], %[[c0]] : i32
// CHECK:           %[[VAL_8:.*]]:2 = scf.if %[[VAL_7]] -> (i32, f32) {
// CHECK:             %[[VAL_9:.*]]:4 = scf.for %[[arg:.+]] = %[[c0]] to %[[ub]] step %[[c1]] iter_args(%[[VAL_12:.*]] = %[[cst0]], %[[VAL_13:.*]] = %[[true]], %[[VAL_11:.*]] = %[[undef_i32]], %[[idx:.*]] = %[[undef_f32]]) -> (f32, i1, i32, f32)  : i32 {
// CHECK:               %[[VAL_14:.*]]:2 = scf.if %[[VAL_13]] -> (i32, f32, i1) {
// CHECK:                 %[[VAL_15:.*]] = "test.something"() : () -> i1
// CHECK:                 %[[VAL_16:.*]] = arith.addf %[[VAL_12]], %[[cst1]] : f32
// CHECK:                 scf.yield %[[VAL_16]], %[[VAL_15]] : f32, i1
// CHECK:               } else {
// CHECK:                 scf.yield %[[VAL_12]], %[[false]] : f32, i1
// CHECK:               }
// CHECK:               scf.yield %[[VAL_18]]#1, %[[VAL_18]]#2, %[[arg]], %[[VAL_12]] : f32, i1, i32, f32
// CHECK:             }
// CHECK:             scf.yield %[[VAL_9]]#2, %[[VAL_9]]#3 : i32, f32
// CHECK:           } else {
// CHECK:             scf.yield %[[c0]], %[[cst0]] : i32, f32
// CHECK:           }
// CHECK:           return %[[VAL_20:.*]]#0, %[[VAL_20]]#1 : i32, f32
// CHECK:         }

// CHECK-LABEL:   func.func @w2f_inner(
// CHECK-SAME:                         %[[VAL_0:.*]]: i32) -> (i32, f32) {
// CHECK:           %[[VAL_1:.*]] = arith.constant true
// CHECK:           %[[VAL_2:.*]] = arith.constant false
// CHECK:           %[[VAL_3:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[VAL_4:.*]] = arith.constant 1.000000e+00 : f32
// CHECK:           %[[VAL_5:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_6:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_7:.*]] = arith.cmpi ugt, %[[VAL_0]], %[[VAL_5]] : i32
// CHECK:           %[[VAL_8:.*]]:2 = scf.if %[[VAL_7]] -> (i32, f32) {
// CHECK:             %[[VAL_9:.*]]:3 = scf.for %[[VAL_10:.*]] = %[[VAL_5]] to %[[VAL_0]] step %[[VAL_6]] iter_args(%[[VAL_11:.*]] = %[[VAL_5]], %[[VAL_12:.*]] = %[[VAL_3]], %[[VAL_13:.*]] = %[[VAL_1]]) -> (i32, f32, i1)  : i32 {
// CHECK:               %[[VAL_14:.*]]:3 = scf.if %[[VAL_13]] -> (i32, f32, i1) {
// CHECK:                 %[[VAL_15:.*]] = "test.something"() : () -> i1
// CHECK:                 %[[VAL_16:.*]]:2 = scf.if %[[VAL_15]] -> (i32, f32) {
// CHECK:                   %[[VAL_17:.*]] = arith.addf %[[VAL_12]], %[[VAL_4]] : f32
// CHECK:                   %[[VAL_18:.*]] = arith.addi %[[VAL_11]], %[[VAL_6]] : i32
// CHECK:                   scf.yield %[[VAL_18]], %[[VAL_17]] : i32, f32
// CHECK:                 } else {
// CHECK:                   scf.yield %[[VAL_11]], %[[VAL_12]] : i32, f32
// CHECK:                 }
// CHECK:                 scf.yield %[[VAL_16]]#0, %[[VAL_16]]#1, %[[VAL_15]] : i32, f32, i1
// CHECK:               } else {
// CHECK:                 scf.yield %[[VAL_11]], %[[VAL_12]], %[[VAL_2]] : i32, f32, i1
// CHECK:               }
// CHECK:               scf.yield %[[VAL_20:.*]]#0, %[[VAL_20]]#1, %[[VAL_20]]#2 : i32, f32, i1
// CHECK:             }
// CHECK:             scf.yield %[[VAL_21:.*]]#0, %[[VAL_21]]#1 : i32, f32
// CHECK:           } else {
// CHECK:             scf.yield %[[VAL_5]], %[[VAL_3]] : i32, f32
// CHECK:           }
// CHECK:           return %[[VAL_22:.*]]#0, %[[VAL_22]]#1 : i32, f32
// CHECK:         }

// CHECK-LABEL:   func.func @_Z17compute_tran_tempPfPS_iiiiiiii(
// CHECK-SAME: %[[VAL_0:.*]]: i8, %[[VAL_1:.*]]: index, %[[VAL_2:.*]]: i32, %[[VAL_3:.*]]: i32, %[[VAL_4:.*]]: i32) -> i32 {
// CHECK:           %[[VAL_5:.*]] = arith.constant true
// CHECK:           %[[VAL_6:.*]] = arith.constant false
// CHECK:           %[[VAL_7:.*]] = arith.constant 0 : i8
// CHECK:           %[[VAL_8:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_9:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_10:.*]] = arith.cmpi sgt, %[[VAL_2]], %[[VAL_9]] : i32
// CHECK:           %[[VAL_11:.*]] = scf.if %[[VAL_10]] -> (i32) {
// CHECK:             %[[VAL_12:.*]]:3 = scf.for %[[VAL_13:.*]] = %[[VAL_9]] to %[[VAL_2]] step %[[VAL_8]] iter_args(%[[VAL_14:.*]] = %[[VAL_0]], %[[VAL_15:.*]] = %[[VAL_9]], %[[VAL_16:.*]] = %[[VAL_5]]) -> (i8, i32, i1)  : i32 {
// CHECK:               %[[VAL_17:.*]]:3 = scf.if %[[VAL_16]] -> (i8, i32, i1) {
// CHECK:                 %[[VAL_18:.*]] = arith.addi %[[VAL_15]], %[[VAL_8]] : i32
// CHECK:                 %[[VAL_19:.*]] = arith.cmpi ne, %[[VAL_15]], %[[VAL_4]] : i32
// CHECK:                 %[[VAL_20:.*]] = scf.if %[[VAL_19]] -> (i32) {
// CHECK:                   scf.yield %[[VAL_18]] : i32
// CHECK:                 } else {
// CHECK:                   scf.yield %[[VAL_15]] : i32
// CHECK:                 }
// CHECK:                 scf.yield %[[VAL_7]], %[[VAL_20]], %[[VAL_19]] : i8, i32, i1
// CHECK:               } else {
// CHECK:                 scf.yield %[[VAL_14]], %[[VAL_15]], %[[VAL_6]] : i8, i32, i1
// CHECK:               }
// CHECK:               scf.yield %[[VAL_21:.*]]#0, %[[VAL_21]]#1, %[[VAL_21]]#2 : i8, i32, i1
// CHECK:             }
// CHECK:             scf.yield %[[VAL_22:.*]]#1 : i32
// CHECK:           } else {
// CHECK:             scf.yield %[[VAL_9]] : i32
// CHECK:           }
// CHECK:           return %[[VAL_11]] : i32
// CHECK:         }