// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=transpose_like_broadcast_concat" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s

// A broadcast_in_dim that only permutes dimensions and inserts unit dimensions
// (the fusion of a transpose and a reshape) is pushed through a concatenate
// onto its operands, just like TransposeConcat does for a transpose.
func.func @permute_and_insert(%a: tensor<4x8xf64>, %b: tensor<4x8xf64>) -> tensor<1x8x8xf64> {
  %0 = stablehlo.concatenate %a, %b, dim = 0 : (tensor<4x8xf64>, tensor<4x8xf64>) -> tensor<8x8xf64>
  %1 = stablehlo.broadcast_in_dim %0, dims = [2, 1] : (tensor<8x8xf64>) -> tensor<1x8x8xf64>
  return %1 : tensor<1x8x8xf64>
}

// CHECK:       func.func @permute_and_insert(%[[ARG0:[a-z0-9_]+]]: tensor<4x8xf64>, %[[ARG1:[a-z0-9_]+]]: tensor<4x8xf64>) -> tensor<1x8x8xf64> {
// CHECK-NEXT:  %[[V0:[a-z0-9_]+]] = stablehlo.broadcast_in_dim %[[ARG0]], dims = [2, 1] : (tensor<4x8xf64>) -> tensor<1x8x4xf64>
// CHECK-NEXT:  %[[V1:[a-z0-9_]+]] = stablehlo.broadcast_in_dim %[[ARG1]], dims = [2, 1] : (tensor<4x8xf64>) -> tensor<1x8x4xf64>
// CHECK-NEXT:  %[[V2:[a-z0-9_]+]] = stablehlo.concatenate %[[V0]], %[[V1]], dim = 2 : (tensor<1x8x4xf64>, tensor<1x8x4xf64>) -> tensor<1x8x8xf64>
// CHECK-NEXT:  return %[[V2]] : tensor<1x8x8xf64>
// CHECK-NEXT:  }

// A broadcast that replicates data is not transpose-like and is left alone.
func.func @expanding(%a: tensor<4x8xf64>, %b: tensor<4x8xf64>) -> tensor<3x8x8xf64> {
  %0 = stablehlo.concatenate %a, %b, dim = 0 : (tensor<4x8xf64>, tensor<4x8xf64>) -> tensor<8x8xf64>
  %1 = stablehlo.broadcast_in_dim %0, dims = [2, 1] : (tensor<8x8xf64>) -> tensor<3x8x8xf64>
  return %1 : tensor<3x8x8xf64>
}

// CHECK:       func.func @expanding(%[[ARG0:[a-z0-9_]+]]: tensor<4x8xf64>, %[[ARG1:[a-z0-9_]+]]: tensor<4x8xf64>) -> tensor<3x8x8xf64> {
// CHECK-NEXT:  %[[V0:[a-z0-9_]+]] = stablehlo.concatenate %[[ARG0]], %[[ARG1]], dim = 0 : (tensor<4x8xf64>, tensor<4x8xf64>) -> tensor<8x8xf64>
// CHECK-NEXT:  %[[V1:[a-z0-9_]+]] = stablehlo.broadcast_in_dim %[[V0]], dims = [2, 1] : (tensor<8x8xf64>) -> tensor<3x8x8xf64>
// CHECK-NEXT:  return %[[V1]] : tensor<3x8x8xf64>
// CHECK-NEXT:  }

// A concatenate with other users is not duplicated.
func.func @multi_use(%a: tensor<4x8xf64>, %b: tensor<4x8xf64>) -> (tensor<1x8x8xf64>, tensor<8x8xf64>) {
  %0 = stablehlo.concatenate %a, %b, dim = 0 : (tensor<4x8xf64>, tensor<4x8xf64>) -> tensor<8x8xf64>
  %1 = stablehlo.broadcast_in_dim %0, dims = [2, 1] : (tensor<8x8xf64>) -> tensor<1x8x8xf64>
  return %1, %0 : tensor<1x8x8xf64>, tensor<8x8xf64>
}

// CHECK:       func.func @multi_use(%[[ARG0:[a-z0-9_]+]]: tensor<4x8xf64>, %[[ARG1:[a-z0-9_]+]]: tensor<4x8xf64>) -> (tensor<1x8x8xf64>, tensor<8x8xf64>) {
// CHECK-NEXT:  %[[V0:[a-z0-9_]+]] = stablehlo.concatenate %[[ARG0]], %[[ARG1]], dim = 0 : (tensor<4x8xf64>, tensor<4x8xf64>) -> tensor<8x8xf64>
// CHECK-NEXT:  %[[V1:[a-z0-9_]+]] = stablehlo.broadcast_in_dim %[[V0]], dims = [2, 1] : (tensor<8x8xf64>) -> tensor<1x8x8xf64>
// CHECK-NEXT:  return %[[V1]], %[[V0]] : tensor<1x8x8xf64>, tensor<8x8xf64>
// CHECK-NEXT:  }
