// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt | FileCheck %s

// Reproduction of: https://github.com/EnzymeAD/Enzyme-JAX/issues/XXXX
// "x != false and others"
//
// Comparisons of an i1 (boolean) tensor against a boolean constant should
// simplify. The optimizer currently passes them through unchanged.
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
