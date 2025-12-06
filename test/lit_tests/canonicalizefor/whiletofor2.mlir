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

// CHECK-LABEL:   func.func @w2f(
// CHECK-SAME:                   %[[ub:.*]]: i32) -> (i32, f32) {
// CHECK-DAG:           %[[undef_f32:.+]] = ub.poison : f32
// CHECK-DAG:           %[[undef_i32:.+]] = ub.poison : i32
// CHECK-DAG:           %[[undef_i1:.+]] = ub.poison : i1
// CHECK-DAG:           %[[true:.*]] = arith.constant true
// CHECK-DAG:           %[[false:.*]] = arith.constant false
// CHECK-DAG:           %[[cst0:.*]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:           %[[cst1:.*]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG:           %[[c0:.*]] = arith.constant 0 : i32
// CHECK-DAG:           %[[c1:.*]] = arith.constant 1 : i32
// CHECK:           %[[FOR:.*]]:6 = scf.for %[[arg:.+]] = %[[c0]] to %[[ub]] step %[[c1]] iter_args(%[[VAL_1:.*]] = %[[c0]], %[[VAL_2:.*]] = %[[cst0]], %[[VAL_3:.*]] = %[[true]], %[[VAL_11:.*]] = %[[c0]], %[[VAL_12:.*]] = %[[cst0]], %[[VAL_13:.*]] = %[[true]])
// CHECK:             %[[IF1:.*]]:3 = scf.if %[[VAL_13]] -> (i32, f32, i1) {
// CHECK:               scf.yield %[[VAL_1]], %[[VAL_2]], %[[VAL_3]] : i32, f32, i1
// CHECK:             } else {
// CHECK:               scf.yield %[[VAL_11]], %[[VAL_12]], %[[false]] : i32, f32, i1
// CHECK:             }
// CHECK:             %[[CMP:.*]] = arith.cmpi slt, %[[arg]], %[[ub]]
// CHECK:             %[[AND:.*]] = arith.andi %[[CMP]], %[[IF1]]#2
// CHECK:             %[[IF2:.*]]:3 = scf.if %[[AND]] -> (i32, f32, i1) {
// CHECK:                 %[[VAL_15:.*]] = "test.something"() : () -> i1
// CHECK:                 %[[VAL_16:.*]] = arith.addf %[[IF1]]#1, %[[cst1]] : f32
// CHECK:                 %[[VAL_17:.*]] = arith.addi %[[IF1]]#0, %[[c1]] : i32
// CHECK:                 scf.yield %[[VAL_17]], %[[VAL_16]], %[[VAL_15]] : i32, f32, i1
// CHECK:               } else {
// CHECK:                 scf.yield %[[undef_i32]], %[[undef_f32]], %[[undef_i1]] : i32, f32, i1
// CHECK:               }
// CHECK:             scf.yield %[[IF2]]#0, %[[IF2]]#1, %[[IF2]]#2, %[[IF1]]#0, %[[IF1]]#1, %[[IF1]]#2 : i32, f32, i1, i32, f32, i1
// CHECK:           }
// CHECK:           return %[[FOR]]#3, %[[FOR]]#4 : i32, f32
// CHECK:         }

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

// CHECK-LABEL:   func.func @w2f_inner(
// CHECK-SAME:                         %[[VAL_0:.*]]: i32) -> (i32, f32) {
// CHECK-DAG:           %[[undef_f32:.+]] = ub.poison : f32
// CHECK-DAG:           %[[undef_i32:.+]] = ub.poison : i32
// CHECK-DAG:           %[[undef_i1:.+]] = ub.poison : i1
// CHECK-DAG:           %[[true:.*]] = arith.constant true
// CHECK-DAG:           %[[false:.*]] = arith.constant false
// CHECK-DAG:           %[[cst0:.*]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:           %[[cst1:.*]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG:           %[[c0:.*]] = arith.constant 0 : i32
// CHECK-DAG:           %[[c1:.*]] = arith.constant 1 : i32
// CHECK:           %[[FOR:.*]]:6 = scf.for %[[arg:.+]] = %[[c0]] to %[[VAL_0]] step %[[c1]] iter_args(%[[VAL_1:.*]] = %[[c0]], %[[VAL_2:.*]] = %[[cst0]], %[[VAL_3:.*]] = %[[true]], %[[VAL_11:.*]] = %[[c0]], %[[VAL_12:.*]] = %[[cst0]], %[[VAL_13:.*]] = %[[true]])
// CHECK:             %[[IF1:.*]]:3 = scf.if %[[VAL_13]] -> (i32, f32, i1) {
// CHECK:               scf.yield %[[VAL_1]], %[[VAL_2]], %[[VAL_3]] : i32, f32, i1
// CHECK:             } else {
// CHECK:               scf.yield %[[VAL_11]], %[[VAL_12]], %[[false]] : i32, f32, i1
// CHECK:             }
// CHECK:             %[[CMP:.*]] = arith.cmpi slt, %[[arg]], %[[ub]]
// CHECK:             %[[AND:.*]] = arith.andi %[[CMP]], %[[IF1]]#2
// CHECK:             %[[IF2:.*]]:3 = scf.if %[[AND]] -> (i32, f32, i1) {
// CHECK:                 %[[VAL_15:.*]] = "test.something"() : () -> i1
// CHECK:                 %[[IF_INNER:.*]]:2 = scf.if %[[VAL_15]] -> (i32, f32) {
// CHECK:                   %[[VAL_16:.*]] = arith.addf %[[IF1]]#1, %[[cst1]] : f32
// CHECK:                   %[[VAL_17:.*]] = arith.addi %[[IF1]]#0, %[[c1]] : i32
// CHECK:                   scf.yield %[[VAL_17]], %[[VAL_16]] : i32, f32
// CHECK:                 } else {
// CHECK:                   scf.yield %[[IF1]]#0, %[[IF1]]#1 : i32, f32
// CHECK:                 }
// CHECK:                 scf.yield %[[IF_INNER]]#0, %[[IF_INNER]]#1, %[[VAL_15]] : i32, f32, i1
// CHECK:               } else {
// CHECK:                 scf.yield %[[undef_i32]], %[[undef_f32]], %[[undef_i1]] : i32, f32, i1
// CHECK:               }
// CHECK:             scf.yield %[[IF2]]#0, %[[IF2]]#1, %[[IF2]]#2, %[[IF1]]#0, %[[IF1]]#1, %[[IF1]]#2 : i32, f32, i1, i32, f32, i1
// CHECK:           }
// CHECK:           return %[[FOR]]#3, %[[FOR]]#4 : i32, f32

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

// CHECK-LABEL:   func.func @_Z17compute_tran_tempPfPS_iiiiiiii(
// CHECK-SAME: %[[VAL_0:.*]]: i8, %[[VAL_1:.*]]: index, %[[ub:.*]]: i32, %[[VAL_3:.*]]: i32, %[[VAL_4:.*]]: i32) -> i32 {
// CHECK-DAG:           %[[true:.*]] = arith.constant true
// CHECK-DAG:           %[[false:.*]] = arith.constant false
// CHECK-DAG:           %[[c0_i8:.*]] = arith.constant 0 : i8
// CHECK-DAG:           %[[c1_i32:.*]] = arith.constant 1 : i32
// CHECK-DAG:           %[[c0_i32:.*]] = arith.constant 0 : i32
// CHECK-DAG:           %[[undef_i8:.*]] = ub.poison : i8
// CHECK-DAG:           %[[undef_i32:.+]] = ub.poison : i32
// CHECK-DAG:           %[[undef_i1:.+]] = ub.poison : i1
// CHECK:           %[[FOR:.*]]:6 = scf.for %[[arg:.*]] = %[[c0_i32]] to %[[ub]] step %[[c1_i32]] iter_args(%[[VAL_13:.*]] = %[[c0_i32]], %[[VAL_14:.*]] = %[[VAL_0]], %[[VAL_16:.*]] = %[[true]], %[[VAL_15:.*]] = %[[VAL_0]], %[[VAL_5:.*]] = %[[c0_i32]], %[[VAL_6:.*]] = %[[true]]) -> (i32, i8, i1, i8, i32, i1)  : i32 {
// CHECK:             %[[IF1:.*]]:3 = scf.if %[[VAL_6]] -> (i8, i32, i1) {
// CHECK:               scf.yield %[[VAL_14]], %[[VAL_13]], %[[VAL_16]] : i8, i32, i1
// CHECK:             } else {
// CHECK:               scf.yield %[[VAL_15]], %[[VAL_5]], %[[false]] : i8, i32, i1
// CHECK:             }
// CHECK:             %[[CMP:.*]] = arith.cmpi slt, %[[arg]], %[[ub]]
// CHECK:             %[[AND:.*]] = arith.andi %[[CMP]], %[[IF1]]#2
// CHECK:             %[[IF2:.*]]:3 = scf.if %[[AND]] -> (i32, i8, i1) {
// CHECK:               %[[ADD:.*]] = arith.addi %[[IF1]]#1, %[[c1_i32]] : i32
// CHECK:               %[[VAL_19:.*]] = arith.cmpi ne, %[[IF1]]#1, %[[VAL_4]] : i32
// CHECK:               %[[IF_INNER:.*]] = scf.if %[[VAL_19]] -> (i32) {
// CHECK:                 scf.yield %[[ADD]] : i32
// CHECK:               } else {
// CHECK:                 scf.yield %[[IF1]]#1 : i32
// CHECK:               }
// CHECK:               scf.yield %[[IF_INNER]], %[[c0_i8]], %[[VAL_19]] : i32, i8, i1
// CHECK:             } else {
// CHECK:               scf.yield %[[undef_i32]], %[[undef_i8]], %[[undef_i1]] : i32, i8, i1
// CHECK:             }
// CHECK:             scf.yield %[[IF2]]#0, %[[IF2]]#1, %[[IF2]]#2, %[[IF1]]#0, %[[IF1]]#1, %[[IF1]]#2 : i32, i8, i1, i8, i32, i1
// CHECK:           }
// CHECK:           return %[[FOR]]#4 : i32
// CHECK:         }

}
