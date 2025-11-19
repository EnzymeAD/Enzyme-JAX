// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=dot_general_symmetric_simplify" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s

func.func @lhs_symmetric(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<2x2xf32>) -> tensor<2x2xf32>
  %1 = stablehlo.add %arg0, %0 : tensor<2x2xf32>
  // Contracting dim 0 of LHS (symmetric). Should switch to dim 1.
  %2 = stablehlo.dot_general %1, %arg1, contracting_dims = [0] x [0] : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  return %2 : tensor<2x2xf32>
}

// CHECK-LABEL: func.func @lhs_symmetric
// CHECK: %[[TRANS:.*]] = stablehlo.transpose %arg0, dims = [1, 0]
// CHECK: %[[SYM:.*]] = stablehlo.add %arg0, %[[TRANS]] {enzymexla.guaranteed_symmetric = true}
// CHECK: %[[RES:.*]] = stablehlo.dot_general %[[SYM]], %arg1, contracting_dims = [1] x [0]
// CHECK: return %[[RES]]

func.func @rhs_symmetric(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %0 = stablehlo.transpose %arg1, dims = [1, 0] : (tensor<2x2xf32>) -> tensor<2x2xf32>
  %1 = stablehlo.add %arg1, %0 : tensor<2x2xf32>
  // Contracting dim 0 of RHS (symmetric). Should switch to dim 1.
  %2 = stablehlo.dot_general %arg0, %1, contracting_dims = [1] x [0] : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  return %2 : tensor<2x2xf32>
}

// CHECK-LABEL: func.func @rhs_symmetric
// CHECK: %[[TRANS:.*]] = stablehlo.transpose %arg1, dims = [1, 0]
// CHECK: %[[SYM:.*]] = stablehlo.add %arg1, %[[TRANS]] {enzymexla.guaranteed_symmetric = true}
// CHECK: %[[RES:.*]] = stablehlo.dot_general %arg0, %[[SYM]], contracting_dims = [1] x [1]
// CHECK: return %[[RES]]

func.func @lhs_not_symmetric(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xf32> {
  // Contracting dim 0 of LHS (NOT symmetric). Should NOT switch.
  %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [0] x [0] : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  return %0 : tensor<2x2xf32>
}

// CHECK-LABEL: func.func @lhs_not_symmetric
// CHECK: %[[RES:.*]] = stablehlo.dot_general %arg0, %arg1, contracting_dims = [0] x [0]
// CHECK: return %[[RES]]

func.func @rhs_not_symmetric(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xf32> {
  // Contracting dim 0 of RHS (NOT symmetric). Should NOT switch.
  %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0] : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  return %0 : tensor<2x2xf32>
}

// CHECK-LABEL: func.func @rhs_not_symmetric
// CHECK: %[[RES:.*]] = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0]
// CHECK: return %[[RES]]
