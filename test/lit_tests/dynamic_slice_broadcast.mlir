// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=dynamic_slice_broadcast" --transform-interpreter --enzyme-hlo-remove-transform --split-input-file | FileCheck %s

// CHECK-LABEL: func.func @exact
// CHECK-SAME: (%[[X:[^:]+]]: tensor<4x5xf64>, %[[I:[^:]+]]: tensor<i64>)
// CHECK-NEXT: %[[B:.+]] = stablehlo.broadcast_in_dim %[[X]], dims = [1, 2] : (tensor<4x5xf64>) -> tensor<1x4x5xf64>
// CHECK-NEXT: return %[[B]] : tensor<1x4x5xf64>
func.func @exact(%x: tensor<4x5xf64>, %i: tensor<i64>) -> tensor<1x4x5xf64> {
  %c0 = stablehlo.constant dense<0> : tensor<i64>
  %b = stablehlo.broadcast_in_dim %x, dims = [1, 2] : (tensor<4x5xf64>) -> tensor<14x4x5xf64>
  %s = stablehlo.dynamic_slice %b, %i, %c0, %c0, sizes = [1, 4, 5] : (tensor<14x4x5xf64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<1x4x5xf64>
  return %s : tensor<1x4x5xf64>
}

// -----

// CHECK-LABEL: func.func @partial
// CHECK: %[[B:.+]] = stablehlo.broadcast_in_dim %{{.+}}, dims = [1, 2] : (tensor<4x5xf64>) -> tensor<14x4x5xf64>
// CHECK-NEXT: %[[S:.+]] = stablehlo.dynamic_slice %[[B]], %{{.+}}, %{{.+}}, %{{.+}}, sizes = [1, 2, 5]
// CHECK-NEXT: return %[[S]]
func.func @partial(%x: tensor<4x5xf64>, %i: tensor<i64>, %j: tensor<i64>) -> tensor<1x2x5xf64> {
  %c0 = stablehlo.constant dense<0> : tensor<i64>
  %b = stablehlo.broadcast_in_dim %x, dims = [1, 2] : (tensor<4x5xf64>) -> tensor<14x4x5xf64>
  %s = stablehlo.dynamic_slice %b, %i, %j, %c0, sizes = [1, 2, 5] : (tensor<14x4x5xf64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<1x2x5xf64>
  return %s : tensor<1x2x5xf64>
}

// -----

// CHECK-LABEL: func.func @batched
// CHECK-SAME: (%[[X:[^:]+]]: tensor<1x5xf64>, %[[I:[^:]+]]: tensor<i64>, %[[J:[^:]+]]: tensor<i64>)
// CHECK-NEXT: %[[B:.+]] = stablehlo.broadcast_in_dim %[[X]], dims = [2, 3] : (tensor<1x5xf64>) -> tensor<1x2x3x5xf64>
// CHECK-NEXT: return %[[B]] : tensor<1x2x3x5xf64>
func.func @batched(%x: tensor<1x5xf64>, %i: tensor<i64>, %j: tensor<i64>) -> tensor<1x2x3x5xf64> {
  %c0 = stablehlo.constant dense<0> : tensor<i64>
  %b = stablehlo.broadcast_in_dim %x, dims = [2, 3] : (tensor<1x5xf64>) -> tensor<7x6x4x5xf64>
  %s = stablehlo.dynamic_slice %b, %i, %j, %c0, %c0, sizes = [1, 2, 3, 5] : (tensor<7x6x4x5xf64>, tensor<i64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<1x2x3x5xf64>
  return %s : tensor<1x2x3x5xf64>
}
