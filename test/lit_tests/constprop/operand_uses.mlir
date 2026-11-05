// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=and_const_prop;add_const_prop;neg_const_prop" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s

// The BK1ELEMENT lane guards: each mask constant has one use, the and-results
// several. Each fold replaces a constant that dies.
func.func @lane_guards() -> (tensor<2x2x2xi1>, tensor<2x2x2xi1>, tensor<2x2x2xi1>) {
  %a = stablehlo.constant dense<[[[true, true], [true, true]], [[false, false], [false, false]]]> : tensor<2x2x2xi1>
  %b = stablehlo.constant dense<[[[true, false], [true, false]], [[true, false], [true, false]]]> : tensor<2x2x2xi1>
  %c = stablehlo.constant dense<[[[true, true], [false, false]], [[true, true], [false, false]]]> : tensor<2x2x2xi1>
  %ab = stablehlo.and %a, %b : tensor<2x2x2xi1>
  %abc = stablehlo.and %ab, %c : tensor<2x2x2xi1>
  return %ab, %abc, %ab : tensor<2x2x2xi1>, tensor<2x2x2xi1>, tensor<2x2x2xi1>
}

// CHECK-LABEL: func.func @lane_guards(
// CHECK-SAME: ) -> (tensor<2x2x2xi1>, tensor<2x2x2xi1>, tensor<2x2x2xi1>) {
// CHECK-NEXT: %[[V0:[^ ,:)]+]] = stablehlo.constant dense<{{\[\[}}[true, false], [true, false]], {{\[\[}}false, false], [false, false]]]> : tensor<2x2x2xi1>
// CHECK-NEXT: %[[V1:[^ ,:)]+]] = stablehlo.constant dense<{{\[\[}}[true, false], [false, false]], {{\[\[}}false, false], [false, false]]]> : tensor<2x2x2xi1>
// CHECK-NEXT: return %[[V0]], %[[V1]], %[[V0]] : tensor<2x2x2xi1>, tensor<2x2x2xi1>, tensor<2x2x2xi1>
// CHECK-NEXT: }

// Stays: both operands are non-splat and each has another use.
func.func @both_shared() -> (tensor<4xi32>, tensor<4xi32>, tensor<4xi32>) {
  %a = stablehlo.constant dense<[1, 2, 3, 4]> : tensor<4xi32>
  %b = stablehlo.constant dense<[5, 6, 7, 8]> : tensor<4xi32>
  %s = stablehlo.add %a, %b : tensor<4xi32>
  return %s, %a, %b : tensor<4xi32>, tensor<4xi32>, tensor<4xi32>
}

// CHECK-LABEL: func.func @both_shared(
// CHECK-SAME: ) -> (tensor<4xi32>, tensor<4xi32>, tensor<4xi32>) {
// CHECK-NEXT: %[[V0:[^ ,:)]+]] = stablehlo.constant dense<[1, 2, 3, 4]> : tensor<4xi32>
// CHECK-NEXT: %[[V1:[^ ,:)]+]] = stablehlo.constant dense<[5, 6, 7, 8]> : tensor<4xi32>
// CHECK-NEXT: %[[V2:[^ ,:)]+]] = stablehlo.add %[[V0]], %[[V1]] : tensor<4xi32>
// CHECK-NEXT: return %[[V2]], %[[V0]], %[[V1]] : tensor<4xi32>, tensor<4xi32>, tensor<4xi32>
// CHECK-NEXT: }

// Stays: the non-splat operand has another use, so folding would add a
// non-splat constant beside it.
func.func @splat_operand() -> (tensor<4xi32>, tensor<4xi32>, tensor<4xi32>) {
  %a = stablehlo.constant dense<[1, 2, 3, 4]> : tensor<4xi32>
  %b = stablehlo.constant dense<5> : tensor<4xi32>
  %s = stablehlo.add %a, %b : tensor<4xi32>
  return %s, %s, %a : tensor<4xi32>, tensor<4xi32>, tensor<4xi32>
}

// CHECK-LABEL: func.func @splat_operand(
// CHECK-SAME: ) -> (tensor<4xi32>, tensor<4xi32>, tensor<4xi32>) {
// CHECK-NEXT: %[[V0:[^ ,:)]+]] = stablehlo.constant dense<[1, 2, 3, 4]> : tensor<4xi32>
// CHECK-NEXT: %[[V1:[^ ,:)]+]] = stablehlo.constant dense<5> : tensor<4xi32>
// CHECK-NEXT: %[[V2:[^ ,:)]+]] = stablehlo.add %[[V0]], %[[V1]] : tensor<4xi32>
// CHECK-NEXT: return %[[V2]], %[[V2]], %[[V0]] : tensor<4xi32>, tensor<4xi32>, tensor<4xi32>
// CHECK-NEXT: }

// A unary op of a constant used only there: folds, its result used twice.
func.func @unary_dying() -> (tensor<3xf64>, tensor<3xf64>) {
  %a = stablehlo.constant dense<[1.0, 2.0, 3.0]> : tensor<3xf64>
  %n = stablehlo.negate %a : tensor<3xf64>
  return %n, %n : tensor<3xf64>, tensor<3xf64>
}

// CHECK-LABEL: func.func @unary_dying(
// CHECK-SAME: ) -> (tensor<3xf64>, tensor<3xf64>) {
// CHECK-NEXT: %[[V0:[^ ,:)]+]] = stablehlo.constant dense<[-1.000000e+00, -2.000000e+00, -3.000000e+00]> : tensor<3xf64>
// CHECK-NEXT: return %[[V0]], %[[V0]] : tensor<3xf64>, tensor<3xf64>
// CHECK-NEXT: }

// Stays: the unary op's constant has another use.
func.func @unary_shared() -> (tensor<3xf64>, tensor<3xf64>) {
  %a = stablehlo.constant dense<[1.0, 2.0, 3.0]> : tensor<3xf64>
  %n = stablehlo.negate %a : tensor<3xf64>
  return %n, %a : tensor<3xf64>, tensor<3xf64>
}

// CHECK-LABEL: func.func @unary_shared(
// CHECK-SAME: ) -> (tensor<3xf64>, tensor<3xf64>) {
// CHECK-NEXT: %[[V0:[^ ,:)]+]] = stablehlo.constant dense<[1.000000e+00, 2.000000e+00, 3.000000e+00]> : tensor<3xf64>
// CHECK-NEXT: %[[V1:[^ ,:)]+]] = stablehlo.negate %[[V0]] : tensor<3xf64>
// CHECK-NEXT: return %[[V1]], %[[V0]] : tensor<3xf64>, tensor<3xf64>
// CHECK-NEXT: }

// A splat and a non-splat operand used only here: folds.
func.func @splat_and_dying() -> (tensor<4xi32>, tensor<4xi32>) {
  %a = stablehlo.constant dense<[1, 2, 3, 4]> : tensor<4xi32>
  %b = stablehlo.constant dense<5> : tensor<4xi32>
  %s = stablehlo.add %a, %b : tensor<4xi32>
  return %s, %s : tensor<4xi32>, tensor<4xi32>
}

// CHECK-LABEL: func.func @splat_and_dying(
// CHECK-SAME: ) -> (tensor<4xi32>, tensor<4xi32>) {
// CHECK-NEXT: %[[V0:[^ ,:)]+]] = stablehlo.constant dense<[6, 7, 8, 9]> : tensor<4xi32>
// CHECK-NEXT: return %[[V0]], %[[V0]] : tensor<4xi32>, tensor<4xi32>
// CHECK-NEXT: }

// One non-splat constant in both operand positions, used nowhere else.
func.func @same_operand() -> tensor<4xi32> {
  %a = stablehlo.constant dense<[1, 2, 3, 4]> : tensor<4xi32>
  %s = stablehlo.add %a, %a : tensor<4xi32>
  return %s : tensor<4xi32>
}

// CHECK-LABEL: func.func @same_operand(
// CHECK-SAME: ) -> tensor<4xi32> {
// CHECK-NEXT: %[[V0:[^ ,:)]+]] = stablehlo.constant dense<[2, 4, 6, 8]> : tensor<4xi32>
// CHECK-NEXT: return %[[V0]] : tensor<4xi32>
// CHECK-NEXT: }
