// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=select_select_neg_cond" --transform-interpreter --enzyme-hlo-remove-transform --split-input-file %s | FileCheck %s

// Case 1: select(cond, c, select(not(cond), a, b)) -> select(cond, c, a)
func.func @false_branch_neg_cond(%cond: tensor<i1>, %a: tensor<4xf32>, %b: tensor<4xf32>, %c: tensor<4xf32>) -> tensor<4xf32> {
  %not_cond = stablehlo.not %cond : tensor<i1>
  %inner = stablehlo.select %not_cond, %a, %b : tensor<i1>, tensor<4xf32>
  %result = stablehlo.select %cond, %c, %inner : tensor<i1>, tensor<4xf32>
  return %result : tensor<4xf32>
}
// CHECK-LABEL: func.func @false_branch_neg_cond
// CHECK: stablehlo.select %arg0, %arg3, %arg1 : tensor<i1>, tensor<4xf32>

// -----

// Case 2: select(cond, select(not(cond), a, b), c) -> select(cond, b, c)
func.func @true_branch_neg_cond(%cond: tensor<i1>, %a: tensor<4xf32>, %b: tensor<4xf32>, %c: tensor<4xf32>) -> tensor<4xf32> {
  %not_cond = stablehlo.not %cond : tensor<i1>
  %inner = stablehlo.select %not_cond, %a, %b : tensor<i1>, tensor<4xf32>
  %result = stablehlo.select %cond, %inner, %c : tensor<i1>, tensor<4xf32>
  return %result : tensor<4xf32>
}
// CHECK-LABEL: func.func @true_branch_neg_cond
// CHECK: stablehlo.select %arg0, %arg2, %arg3 : tensor<i1>, tensor<4xf32>

// -----

// Case 3: select(and(c1,c2), c, select(and(c1,not(c2)), a, b)) -> select(and(c1,c2), c, select(c1, a, b))
func.func @false_branch_and_neg_cond(%c1: tensor<i1>, %c2: tensor<i1>, %a: tensor<4xf32>, %b: tensor<4xf32>, %c: tensor<4xf32>) -> tensor<4xf32> {
  %and_c1c2 = stablehlo.and %c1, %c2 : tensor<i1>
  %not_c2 = stablehlo.not %c2 : tensor<i1>
  %and_c1_not_c2 = stablehlo.and %c1, %not_c2 : tensor<i1>
  %inner = stablehlo.select %and_c1_not_c2, %a, %b : tensor<i1>, tensor<4xf32>
  %result = stablehlo.select %and_c1c2, %c, %inner : tensor<i1>, tensor<4xf32>
  return %result : tensor<4xf32>
}
// CHECK-LABEL: func.func @false_branch_and_neg_cond
// CHECK-DAG: %[[AND:.*]] = stablehlo.and %arg0, %arg1
// CHECK-DAG: %[[INNER:.*]] = stablehlo.select %arg0, %arg2, %arg3 : tensor<i1>, tensor<4xf32>
// CHECK: stablehlo.select %[[AND]], %arg4, %[[INNER]] : tensor<i1>, tensor<4xf32>

// -----

// Case 4: select(and(c1,c2), select(and(c1,not(c2)), a, b), c) -> select(and(c1,c2), b, c)
func.func @true_branch_and_neg_cond(%c1: tensor<i1>, %c2: tensor<i1>, %a: tensor<4xf32>, %b: tensor<4xf32>, %c: tensor<4xf32>) -> tensor<4xf32> {
  %and_c1c2 = stablehlo.and %c1, %c2 : tensor<i1>
  %not_c2 = stablehlo.not %c2 : tensor<i1>
  %and_c1_not_c2 = stablehlo.and %c1, %not_c2 : tensor<i1>
  %inner = stablehlo.select %and_c1_not_c2, %a, %b : tensor<i1>, tensor<4xf32>
  %result = stablehlo.select %and_c1c2, %inner, %c : tensor<i1>, tensor<4xf32>
  return %result : tensor<4xf32>
}
// CHECK-LABEL: func.func @true_branch_and_neg_cond
// CHECK: %[[AND:.*]] = stablehlo.and %arg0, %arg1
// CHECK: stablehlo.select %[[AND]], %arg3, %arg4 : tensor<i1>, tensor<4xf32>
