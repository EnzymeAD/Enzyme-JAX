// RUN: enzymexlamlir-opt --pass-pipeline="any(enzyme-hlo-generate-td{patterns=reshape_iota},transform-interpreter,enzyme-hlo-remove-transform)" %s | FileCheck %s

// An iota along a unit dimension is all zeros. Dropping that dimension used
// to produce an iota along dimension 0 of the result, i.e. [0, 1, 2] for
// [0, 0, 0]. This came up as wrong pivot indices in a batched LU solve, where
// the per-batch slice of a batch-indexed iota is reshaped away.

// CHECK-LABEL: func.func @drop_trailing_unit_iota_dim
// CHECK:         %[[C:.+]] = stablehlo.constant dense<0> : tensor<3xi64>
// CHECK:         return %[[C]]
func.func @drop_trailing_unit_iota_dim() -> tensor<3xi64> {
  %0 = stablehlo.iota dim = 1 : tensor<3x1xi64>
  %1 = stablehlo.reshape %0 : (tensor<3x1xi64>) -> tensor<3xi64>
  return %1 : tensor<3xi64>
}

// CHECK-LABEL: func.func @drop_middle_unit_iota_dim
// CHECK:         %[[C:.+]] = stablehlo.constant dense<0> : tensor<3x4xi64>
// CHECK:         return %[[C]]
func.func @drop_middle_unit_iota_dim() -> tensor<3x4xi64> {
  %0 = stablehlo.iota dim = 1 : tensor<3x1x4xi64>
  %1 = stablehlo.reshape %0 : (tensor<3x1x4xi64>) -> tensor<3x4xi64>
  return %1 : tensor<3x4xi64>
}

// CHECK-LABEL: func.func @drop_leading_unit_iota_dim
// CHECK:         %[[C:.+]] = stablehlo.constant dense<0> : tensor<4xi64>
// CHECK:         return %[[C]]
func.func @drop_leading_unit_iota_dim() -> tensor<4xi64> {
  %0 = stablehlo.iota dim = 0 : tensor<1x4xi64>
  %1 = stablehlo.reshape %0 : (tensor<1x4xi64>) -> tensor<4xi64>
  return %1 : tensor<4xi64>
}

// CHECK-LABEL: func.func @unit_iota_dim_to_scalar
// CHECK:         %[[C:.+]] = stablehlo.constant dense<0> : tensor<i64>
// CHECK:         return %[[C]]
func.func @unit_iota_dim_to_scalar() -> tensor<i64> {
  %0 = stablehlo.iota dim = 0 : tensor<1xi64>
  %1 = stablehlo.reshape %0 : (tensor<1xi64>) -> tensor<i64>
  return %1 : tensor<i64>
}

// A non-unit iota dimension still maps to the matching result dimension.

// CHECK-LABEL: func.func @drop_other_unit_dim
// CHECK:         %[[I:.+]] = stablehlo.iota dim = 0 : tensor<3xi64>
// CHECK:         return %[[I]]
func.func @drop_other_unit_dim() -> tensor<3xi64> {
  %0 = stablehlo.iota dim = 0 : tensor<3x1xi64>
  %1 = stablehlo.reshape %0 : (tensor<3x1xi64>) -> tensor<3xi64>
  return %1 : tensor<3xi64>
}

// CHECK-LABEL: func.func @insert_unit_dim
// CHECK:         %[[I:.+]] = stablehlo.iota dim = 1 : tensor<1x4xi64>
// CHECK:         return %[[I]]
func.func @insert_unit_dim() -> tensor<1x4xi64> {
  %0 = stablehlo.iota dim = 0 : tensor<4xi64>
  %1 = stablehlo.reshape %0 : (tensor<4xi64>) -> tensor<1x4xi64>
  return %1 : tensor<1x4xi64>
}
