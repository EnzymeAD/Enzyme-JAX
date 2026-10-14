// RUN: enzymexlamlir-opt %s  --enzyme-hlo-generate-td="patterns=speculate_if_binary_op_to_select;speculate_out_of_bounds_array_indexing;if_to_select" --transform-interpreter --enzyme-hlo-remove-transform -allow-unregistered-dialect | FileCheck %s

module {
  // Case 1A: Subtraction, true branch has %arg0 - %arg1, false branch has %arg0.
  // We expect:
  // %identity = constant 0.0
  // %result_if = if (pred) { yield %arg1 } else { yield %identity }
  // %res = sub %arg0, %result_if
  func.func @speculate_sub_right(%pred: tensor<i1>, %arg0: tensor<f64>, %arg1: tensor<f64>) -> tensor<f64> {
    %result = "stablehlo.if"(%pred) ({
      %sub = stablehlo.subtract %arg0, %arg1 : tensor<f64>
      stablehlo.return %sub : tensor<f64>
    }, {
      stablehlo.return %arg0 : tensor<f64>
    }) : (tensor<i1>) -> tensor<f64>
    return %result : tensor<f64>
  }

  // Case 1B: Addition, true branch has %arg0 + %arg1, false branch has %arg1.
  // We expect left-identity (0) used on true branch operand %arg0.
  // %identity = constant 0.0
  // %result_if = if (pred) { yield %arg0 } else { yield %identity }
  // %res = add %result_if, %arg1
  func.func @speculate_add_left(%pred: tensor<i1>, %arg0: tensor<f64>, %arg1: tensor<f64>) -> tensor<f64> {
    %result = "stablehlo.if"(%pred) ({
      %add = stablehlo.add %arg0, %arg1 : tensor<f64>
      stablehlo.return %add : tensor<f64>
    }, {
      stablehlo.return %arg1 : tensor<f64>
    }) : (tensor<i1>) -> tensor<f64>
    return %result : tensor<f64>
  }

  // Case 3: Dynamic slice / reshape chain speculation.
  func.func @speculate_array_indexing(
      %pred: tensor<i1>,
      %arg0: tensor<6x6xf64>,
      %arg1: tensor<i32>,
      %arg2: tensor<i64>,
      %cst_0: tensor<f64>) -> tensor<f64> {
    %cst_4 = stablehlo.constant dense<4> : tensor<i64>
    %cst_1 = stablehlo.constant dense<1> : tensor<i32>
    %result = "stablehlo.if"(%pred) ({
      %30 = stablehlo.add %arg2, %cst_4 : tensor<i64>
      %31 = stablehlo.convert %30 : (tensor<i64>) -> tensor<i32>
      %32 = stablehlo.subtract %31, %cst_1 : tensor<i32>
      %33 = stablehlo.dynamic_slice %arg0, %32, %arg1, sizes = [1, 1] : (tensor<6x6xf64>, tensor<i32>, tensor<i32>) -> tensor<1x1xf64>
      %34 = stablehlo.reshape %33 : (tensor<1x1xf64>) -> tensor<f64>
      stablehlo.return %34 : tensor<f64>
    }, {
      stablehlo.return %cst_0 : tensor<f64>
    }) : (tensor<i1>) -> tensor<f64>
    return %result : tensor<f64>
  }
}

// CHECK-LABEL: func.func @speculate_sub_right(
// CHECK-SAME:     %[[PRED:.*]]: tensor<i1>,
// CHECK-SAME:     %[[LHS:.*]]: tensor<f64>,
// CHECK-SAME:     %[[RHS:.*]]: tensor<f64>) -> tensor<f64> {
// CHECK-DAG:     %[[CST:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK:         %[[IF_RES:.*]] = stablehlo.select %[[PRED]], %[[RHS]], %[[CST]] : tensor<i1>, tensor<f64>
// CHECK:         %[[FINAL:.*]] = stablehlo.subtract %[[LHS]], %[[IF_RES]] : tensor<f64>
// CHECK:         return %[[FINAL]] : tensor<f64>

// CHECK-LABEL: func.func @speculate_add_left(
// CHECK-SAME:     %[[PRED:.*]]: tensor<i1>,
// CHECK-SAME:     %[[LHS:.*]]: tensor<f64>,
// CHECK-SAME:     %[[RHS:.*]]: tensor<f64>) -> tensor<f64> {
// CHECK-DAG:     %[[CST:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK:         %[[IF_RES:.*]] = stablehlo.select %[[PRED]], %[[LHS]], %[[CST]] : tensor<i1>, tensor<f64>
// CHECK:         %[[FINAL:.*]] = stablehlo.add %[[IF_RES]], %[[RHS]] : tensor<f64>
// CHECK:         return %[[FINAL]] : tensor<f64>

// CHECK-LABEL: func.func @speculate_array_indexing(
// CHECK-SAME:     %[[PRED:.*]]: tensor<i1>,
// CHECK-SAME:     %[[ARG0:.*]]: tensor<6x6xf64>,
// CHECK-SAME:     %[[ARG1:.*]]: tensor<i32>,
// CHECK-SAME:     %[[ARG2:.*]]: tensor<i64>,
// CHECK-SAME:     %[[CST0:.*]]: tensor<f64>) -> tensor<f64> {
// CHECK-DAG:     %[[CST4:.*]] = stablehlo.constant dense<4> : tensor<i64>
// CHECK-DAG:     %[[CST1:.*]] = stablehlo.constant dense<1> : tensor<i32>
// CHECK:         %[[ADD:.*]] = stablehlo.add %[[ARG2]], %[[CST4]] : tensor<i64>
// CHECK:         %[[CONV:.*]] = stablehlo.convert %[[ADD]] : (tensor<i64>) -> tensor<i32>
// CHECK:         %[[SUB:.*]] = stablehlo.subtract %[[CONV]], %[[CST1]] : tensor<i32>
// CHECK:         %[[SLICE:.*]] = stablehlo.dynamic_slice %[[ARG0]], %[[SUB]], %[[ARG1]], sizes = [1, 1] : (tensor<6x6xf64>, tensor<i32>, tensor<i32>) -> tensor<1x1xf64>
// CHECK:         %[[RESHAPE:.*]] = stablehlo.reshape %[[SLICE]] : (tensor<1x1xf64>) -> tensor<f64>
// CHECK:         %[[FINAL:.*]] = stablehlo.select %[[PRED]], %[[RESHAPE]], %[[CST0]] : tensor<i1>, tensor<f64>
// CHECK:         return %[[FINAL]] : tensor<f64>
