// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt | FileCheck %s
// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td=patterns=compare_bool_const --transform-interpreter --enzyme-hlo-remove-transform | FileCheck %s

// Reproduction of: https://github.com/EnzymeAD/Enzyme-JAX/issues/1417
// "x != false and others"
//
// Comparisons of an i1 (boolean) tensor against a boolean constant should
// simplify for every comparison direction and either operand order.
//
// For i1, values are constrained to {0, 1}, so:
//   x != false  ->  x
//   x == false  ->  not(x)
//   x == true   ->  x
//   x != true   ->  not(x)

module {

  // The exact pattern reported in the issue:
  //   (x < cst) != false  ->  x < cst
  func.func @issue_ne_false(%arg0: tensor<8000x3xf32>) -> tensor<8000x3xi1> {
    %cst = stablehlo.constant dense<0.0> : tensor<8000x3xf32>
    %false = stablehlo.constant dense<false> : tensor<8000x3xi1>
    %lt = stablehlo.compare LT, %arg0, %cst, FLOAT : (tensor<8000x3xf32>, tensor<8000x3xf32>) -> tensor<8000x3xi1>
    %result = stablehlo.compare NE, %lt, %false, UNSIGNED : (tensor<8000x3xi1>, tensor<8000x3xi1>) -> tensor<8000x3xi1>
    return %result : tensor<8000x3xi1>
  }

  // x != false  ->  x
  func.func @ne_false(%arg0: tensor<4xi1>) -> tensor<4xi1> {
    %false = stablehlo.constant dense<false> : tensor<4xi1>
    %result = stablehlo.compare NE, %arg0, %false, UNSIGNED : (tensor<4xi1>, tensor<4xi1>) -> tensor<4xi1>
    return %result : tensor<4xi1>
  }

  // x == false  ->  not(x)
  func.func @eq_false(%arg0: tensor<4xi1>) -> tensor<4xi1> {
    %false = stablehlo.constant dense<false> : tensor<4xi1>
    %result = stablehlo.compare EQ, %arg0, %false, UNSIGNED : (tensor<4xi1>, tensor<4xi1>) -> tensor<4xi1>
    return %result : tensor<4xi1>
  }

  // x == true  ->  x
  func.func @eq_true(%arg0: tensor<4xi1>) -> tensor<4xi1> {
    %true = stablehlo.constant dense<true> : tensor<4xi1>
    %result = stablehlo.compare EQ, %arg0, %true, UNSIGNED : (tensor<4xi1>, tensor<4xi1>) -> tensor<4xi1>
    return %result : tensor<4xi1>
  }

  // x != true  ->  not(x)
  func.func @ne_true(%arg0: tensor<4xi1>) -> tensor<4xi1> {
    %true = stablehlo.constant dense<true> : tensor<4xi1>
    %result = stablehlo.compare NE, %arg0, %true, UNSIGNED : (tensor<4xi1>, tensor<4xi1>) -> tensor<4xi1>
    return %result : tensor<4xi1>
  }

  // x < false -> false
  func.func @lt_false(%arg0: tensor<4xi1>) -> tensor<4xi1> {
    %false = stablehlo.constant dense<false> : tensor<4xi1>
    %result = stablehlo.compare LT, %arg0, %false, UNSIGNED : (tensor<4xi1>, tensor<4xi1>) -> tensor<4xi1>
    return %result : tensor<4xi1>
  }

  // x < true -> not(x)
  func.func @lt_true(%arg0: tensor<4xi1>) -> tensor<4xi1> {
    %true = stablehlo.constant dense<true> : tensor<4xi1>
    %result = stablehlo.compare LT, %arg0, %true, UNSIGNED : (tensor<4xi1>, tensor<4xi1>) -> tensor<4xi1>
    return %result : tensor<4xi1>
  }

  // x <= false -> not(x)
  func.func @le_false(%arg0: tensor<4xi1>) -> tensor<4xi1> {
    %false = stablehlo.constant dense<false> : tensor<4xi1>
    %result = stablehlo.compare LE, %arg0, %false, UNSIGNED : (tensor<4xi1>, tensor<4xi1>) -> tensor<4xi1>
    return %result : tensor<4xi1>
  }

  // x <= true -> true
  func.func @le_true(%arg0: tensor<4xi1>) -> tensor<4xi1> {
    %true = stablehlo.constant dense<true> : tensor<4xi1>
    %result = stablehlo.compare LE, %arg0, %true, UNSIGNED : (tensor<4xi1>, tensor<4xi1>) -> tensor<4xi1>
    return %result : tensor<4xi1>
  }

  // x >= false -> true
  func.func @ge_false(%arg0: tensor<4xi1>) -> tensor<4xi1> {
    %false = stablehlo.constant dense<false> : tensor<4xi1>
    %result = stablehlo.compare GE, %arg0, %false, UNSIGNED : (tensor<4xi1>, tensor<4xi1>) -> tensor<4xi1>
    return %result : tensor<4xi1>
  }

  // x >= true -> x
  func.func @ge_true(%arg0: tensor<4xi1>) -> tensor<4xi1> {
    %true = stablehlo.constant dense<true> : tensor<4xi1>
    %result = stablehlo.compare GE, %arg0, %true, UNSIGNED : (tensor<4xi1>, tensor<4xi1>) -> tensor<4xi1>
    return %result : tensor<4xi1>
  }

  // x > false -> x
  func.func @gt_false(%arg0: tensor<4xi1>) -> tensor<4xi1> {
    %false = stablehlo.constant dense<false> : tensor<4xi1>
    %result = stablehlo.compare GT, %arg0, %false, UNSIGNED : (tensor<4xi1>, tensor<4xi1>) -> tensor<4xi1>
    return %result : tensor<4xi1>
  }

  // x > true -> false
  func.func @gt_true(%arg0: tensor<4xi1>) -> tensor<4xi1> {
    %true = stablehlo.constant dense<true> : tensor<4xi1>
    %result = stablehlo.compare GT, %arg0, %true, UNSIGNED : (tensor<4xi1>, tensor<4xi1>) -> tensor<4xi1>
    return %result : tensor<4xi1>
  }

  // false == x -> not(x)
  func.func @false_eq(%arg0: tensor<4xi1>) -> tensor<4xi1> {
    %false = stablehlo.constant dense<false> : tensor<4xi1>
    %result = stablehlo.compare EQ, %false, %arg0, UNSIGNED : (tensor<4xi1>, tensor<4xi1>) -> tensor<4xi1>
    return %result : tensor<4xi1>
  }

  // true == x -> x
  func.func @true_eq(%arg0: tensor<4xi1>) -> tensor<4xi1> {
    %true = stablehlo.constant dense<true> : tensor<4xi1>
    %result = stablehlo.compare EQ, %true, %arg0, UNSIGNED : (tensor<4xi1>, tensor<4xi1>) -> tensor<4xi1>
    return %result : tensor<4xi1>
  }

  // false != x -> x
  func.func @false_ne(%arg0: tensor<4xi1>) -> tensor<4xi1> {
    %false = stablehlo.constant dense<false> : tensor<4xi1>
    %result = stablehlo.compare NE, %false, %arg0, UNSIGNED : (tensor<4xi1>, tensor<4xi1>) -> tensor<4xi1>
    return %result : tensor<4xi1>
  }

  // true != x -> not(x)
  func.func @true_ne(%arg0: tensor<4xi1>) -> tensor<4xi1> {
    %true = stablehlo.constant dense<true> : tensor<4xi1>
    %result = stablehlo.compare NE, %true, %arg0, UNSIGNED : (tensor<4xi1>, tensor<4xi1>) -> tensor<4xi1>
    return %result : tensor<4xi1>
  }

  // false < x -> x
  func.func @false_lt(%arg0: tensor<4xi1>) -> tensor<4xi1> {
    %false = stablehlo.constant dense<false> : tensor<4xi1>
    %result = stablehlo.compare LT, %false, %arg0, UNSIGNED : (tensor<4xi1>, tensor<4xi1>) -> tensor<4xi1>
    return %result : tensor<4xi1>
  }

  // true < x -> false
  func.func @true_lt(%arg0: tensor<4xi1>) -> tensor<4xi1> {
    %true = stablehlo.constant dense<true> : tensor<4xi1>
    %result = stablehlo.compare LT, %true, %arg0, UNSIGNED : (tensor<4xi1>, tensor<4xi1>) -> tensor<4xi1>
    return %result : tensor<4xi1>
  }

  // false <= x -> true
  func.func @false_le(%arg0: tensor<4xi1>) -> tensor<4xi1> {
    %false = stablehlo.constant dense<false> : tensor<4xi1>
    %result = stablehlo.compare LE, %false, %arg0, UNSIGNED : (tensor<4xi1>, tensor<4xi1>) -> tensor<4xi1>
    return %result : tensor<4xi1>
  }

  // true <= x -> x
  func.func @true_le(%arg0: tensor<4xi1>) -> tensor<4xi1> {
    %true = stablehlo.constant dense<true> : tensor<4xi1>
    %result = stablehlo.compare LE, %true, %arg0, UNSIGNED : (tensor<4xi1>, tensor<4xi1>) -> tensor<4xi1>
    return %result : tensor<4xi1>
  }

  // false >= x -> not(x)
  func.func @false_ge(%arg0: tensor<4xi1>) -> tensor<4xi1> {
    %false = stablehlo.constant dense<false> : tensor<4xi1>
    %result = stablehlo.compare GE, %false, %arg0, UNSIGNED : (tensor<4xi1>, tensor<4xi1>) -> tensor<4xi1>
    return %result : tensor<4xi1>
  }

  // true >= x -> true
  func.func @true_ge(%arg0: tensor<4xi1>) -> tensor<4xi1> {
    %true = stablehlo.constant dense<true> : tensor<4xi1>
    %result = stablehlo.compare GE, %true, %arg0, UNSIGNED : (tensor<4xi1>, tensor<4xi1>) -> tensor<4xi1>
    return %result : tensor<4xi1>
  }

  // false > x -> false
  func.func @false_gt(%arg0: tensor<4xi1>) -> tensor<4xi1> {
    %false = stablehlo.constant dense<false> : tensor<4xi1>
    %result = stablehlo.compare GT, %false, %arg0, UNSIGNED : (tensor<4xi1>, tensor<4xi1>) -> tensor<4xi1>
    return %result : tensor<4xi1>
  }

  // true > x -> not(x)
  func.func @true_gt(%arg0: tensor<4xi1>) -> tensor<4xi1> {
    %true = stablehlo.constant dense<true> : tensor<4xi1>
    %result = stablehlo.compare GT, %true, %arg0, UNSIGNED : (tensor<4xi1>, tensor<4xi1>) -> tensor<4xi1>
    return %result : tensor<4xi1>
  }

}

// CHECK-LABEL: func.func @issue_ne_false
// CHECK:         %[[LT:.+]] = stablehlo.compare LT
// CHECK-NEXT:    return %[[LT]]

// CHECK-LABEL: func.func @ne_false
// CHECK-NEXT:    return %arg0

// CHECK-LABEL: func.func @eq_false
// CHECK-NEXT:    %[[NOT:.+]] = stablehlo.not %arg0
// CHECK-NEXT:    return %[[NOT]]

// CHECK-LABEL: func.func @eq_true
// CHECK-NEXT:    return %arg0

// CHECK-LABEL: func.func @ne_true
// CHECK-NEXT:    %[[NOT:.+]] = stablehlo.not %arg0
// CHECK-NEXT:    return %[[NOT]]

// CHECK-LABEL: func.func @lt_false(
// CHECK-NEXT:    %[[RESULT:.+]] = stablehlo.constant dense<false> : tensor<4xi1>
// CHECK-NEXT:    return %[[RESULT]] : tensor<4xi1>

// CHECK-LABEL: func.func @lt_true(
// CHECK-NEXT:    %[[RESULT:.+]] = stablehlo.not %arg0 : tensor<4xi1>
// CHECK-NEXT:    return %[[RESULT]] : tensor<4xi1>

// CHECK-LABEL: func.func @le_false(
// CHECK-NEXT:    %[[RESULT:.+]] = stablehlo.not %arg0 : tensor<4xi1>
// CHECK-NEXT:    return %[[RESULT]] : tensor<4xi1>

// CHECK-LABEL: func.func @le_true(
// CHECK-NEXT:    %[[RESULT:.+]] = stablehlo.constant dense<true> : tensor<4xi1>
// CHECK-NEXT:    return %[[RESULT]] : tensor<4xi1>

// CHECK-LABEL: func.func @ge_false(
// CHECK-NEXT:    %[[RESULT:.+]] = stablehlo.constant dense<true> : tensor<4xi1>
// CHECK-NEXT:    return %[[RESULT]] : tensor<4xi1>

// CHECK-LABEL: func.func @ge_true(
// CHECK-NEXT:    return %arg0 : tensor<4xi1>

// CHECK-LABEL: func.func @gt_false(
// CHECK-NEXT:    return %arg0 : tensor<4xi1>

// CHECK-LABEL: func.func @gt_true(
// CHECK-NEXT:    %[[RESULT:.+]] = stablehlo.constant dense<false> : tensor<4xi1>
// CHECK-NEXT:    return %[[RESULT]] : tensor<4xi1>

// CHECK-LABEL: func.func @false_eq(
// CHECK-NEXT:    %[[RESULT:.+]] = stablehlo.not %arg0 : tensor<4xi1>
// CHECK-NEXT:    return %[[RESULT]] : tensor<4xi1>

// CHECK-LABEL: func.func @true_eq(
// CHECK-NEXT:    return %arg0 : tensor<4xi1>

// CHECK-LABEL: func.func @false_ne(
// CHECK-NEXT:    return %arg0 : tensor<4xi1>

// CHECK-LABEL: func.func @true_ne(
// CHECK-NEXT:    %[[RESULT:.+]] = stablehlo.not %arg0 : tensor<4xi1>
// CHECK-NEXT:    return %[[RESULT]] : tensor<4xi1>

// CHECK-LABEL: func.func @false_lt(
// CHECK-NEXT:    return %arg0 : tensor<4xi1>

// CHECK-LABEL: func.func @true_lt(
// CHECK-NEXT:    %[[RESULT:.+]] = stablehlo.constant dense<false> : tensor<4xi1>
// CHECK-NEXT:    return %[[RESULT]] : tensor<4xi1>

// CHECK-LABEL: func.func @false_le(
// CHECK-NEXT:    %[[RESULT:.+]] = stablehlo.constant dense<true> : tensor<4xi1>
// CHECK-NEXT:    return %[[RESULT]] : tensor<4xi1>

// CHECK-LABEL: func.func @true_le(
// CHECK-NEXT:    return %arg0 : tensor<4xi1>

// CHECK-LABEL: func.func @false_ge(
// CHECK-NEXT:    %[[RESULT:.+]] = stablehlo.not %arg0 : tensor<4xi1>
// CHECK-NEXT:    return %[[RESULT]] : tensor<4xi1>

// CHECK-LABEL: func.func @true_ge(
// CHECK-NEXT:    %[[RESULT:.+]] = stablehlo.constant dense<true> : tensor<4xi1>
// CHECK-NEXT:    return %[[RESULT]] : tensor<4xi1>

// CHECK-LABEL: func.func @false_gt(
// CHECK-NEXT:    %[[RESULT:.+]] = stablehlo.constant dense<false> : tensor<4xi1>
// CHECK-NEXT:    return %[[RESULT]] : tensor<4xi1>

// CHECK-LABEL: func.func @true_gt(
// CHECK-NEXT:    %[[RESULT:.+]] = stablehlo.not %arg0 : tensor<4xi1>
// CHECK-NEXT:    return %[[RESULT]] : tensor<4xi1>
