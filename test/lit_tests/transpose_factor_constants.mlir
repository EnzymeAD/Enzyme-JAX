// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=broadcasting_elementwise_all_transpose_operands_simplify" --transform-interpreter --enzyme-hlo-remove-transform | FileCheck %s

// A scalar predicate does not give the result a layout. Both data operands
// are constants. Do not move a transpose between them indefinitely.
// CHECK-LABEL: func.func @constant_views(
// CHECK-SAME: %[[P:.*]]: tensor<i1>
// CHECK-NEXT: %[[A:.*]] = stablehlo.constant dense<2> : tensor<3x2xi32>
// CHECK-NEXT: %[[B:.*]] = stablehlo.constant dense<3> : tensor<2x3xi32>
// CHECK-NEXT: %[[TB:.*]] = stablehlo.transpose %[[B]], dims = [1, 0] : (tensor<2x3xi32>) -> tensor<3x2xi32>
// CHECK-NEXT: %[[S:.*]] = stablehlo.select %[[P]], %[[A]], %[[TB]] : tensor<i1>, tensor<3x2xi32>
// CHECK-NEXT: return %[[S]] : tensor<3x2xi32>
func.func @constant_views(%p: tensor<i1>) -> tensor<3x2xi32> {
  %a = stablehlo.constant dense<2> : tensor<3x2xi32>
  %b = stablehlo.constant dense<3> : tensor<2x3xi32>
  %tb = stablehlo.transpose %b, dims = [1, 0] : (tensor<2x3xi32>) -> tensor<3x2xi32>
  %selected = stablehlo.select %p, %a, %tb : tensor<i1>, tensor<3x2xi32>
  return %selected : tensor<3x2xi32>
}

// A data transpose still determines a layout when the other value is constant.
// CHECK-LABEL: func.func @data_view(
// CHECK-SAME: %[[P:.*]]: tensor<i1>, %[[X:.*]]: tensor<2x3xi32>
// CHECK-NEXT: %[[A:.*]] = stablehlo.constant dense<2> : tensor<3x2xi32>
// CHECK-NEXT: %[[TA:.*]] = stablehlo.transpose %[[A]], dims = [1, 0] : (tensor<3x2xi32>) -> tensor<2x3xi32>
// CHECK-NEXT: %[[S:.*]] = stablehlo.select %[[P]], %[[TA]], %[[X]] : tensor<i1>, tensor<2x3xi32>
// CHECK-NEXT: %[[T:.*]] = stablehlo.transpose %[[S]], dims = [1, 0] : (tensor<2x3xi32>) -> tensor<3x2xi32>
// CHECK-NEXT: return %[[T]] : tensor<3x2xi32>
func.func @data_view(%p: tensor<i1>, %x: tensor<2x3xi32>) -> tensor<3x2xi32> {
  %a = stablehlo.constant dense<2> : tensor<3x2xi32>
  %tx = stablehlo.transpose %x, dims = [1, 0] : (tensor<2x3xi32>) -> tensor<3x2xi32>
  %selected = stablehlo.select %p, %a, %tx : tensor<i1>, tensor<3x2xi32>
  return %selected : tensor<3x2xi32>
}
