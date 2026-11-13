// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt | FileCheck %s

// Exact #1863 reproducer: alpha * (A * x) + beta * (B * x) -> (alpha * A + beta * B) * x
// CHECK-LABEL: func.func @gesummv_shared_rhs(
// CHECK-DAG: %[[BA:.*]] = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
// CHECK-DAG: %[[SA:.*]] = stablehlo.multiply %[[BA]], %arg2 : tensor<64x64xf32>
// CHECK-DAG: %[[BB:.*]] = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
// CHECK-DAG: %[[SB:.*]] = stablehlo.multiply %[[BB]], %arg3 : tensor<64x64xf32>
// CHECK: %[[COMB:.*]] = stablehlo.add %[[SA]], %[[SB]] : tensor<64x64xf32>
// CHECK: %[[DOT:.*]] = stablehlo.dot_general %[[COMB]], %arg4, contracting_dims = [0] x [0] : (tensor<64x64xf32>, tensor<64xf32>) -> tensor<64xf32>
// CHECK-NOT: stablehlo.dot_general
// CHECK: return %[[DOT]] : tensor<64xf32>
func.func @gesummv_shared_rhs(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<64x64xf32>, %arg3: tensor<64x64xf32>, %arg4: tensor<64xf32>) -> tensor<64xf32> {
  %0 = stablehlo.dot_general %arg3, %arg4, contracting_dims = [0] x [0] : (tensor<64x64xf32>, tensor<64xf32>) -> tensor<64xf32>
  %1 = stablehlo.dot_general %arg2, %arg4, contracting_dims = [0] x [0] : (tensor<64x64xf32>, tensor<64xf32>) -> tensor<64xf32>
  %2 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<f32>) -> tensor<64xf32>
  %3 = stablehlo.multiply %2, %1 : tensor<64xf32>
  %4 = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<f32>) -> tensor<64xf32>
  %5 = stablehlo.multiply %4, %0 : tensor<64xf32>
  %6 = stablehlo.add %3, %5 : tensor<64xf32>
  return %6 : tensor<64xf32>
}

// Shared LHS with subtract and one-sided scalar: alpha * (A * x) - A * y -> A * (alpha * x - y)
// CHECK-LABEL: func.func @shared_lhs_sub_one_scaled(
// CHECK: %[[BX:.*]] = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<f32>) -> tensor<64xf32>
// CHECK: %[[SX:.*]] = stablehlo.multiply %[[BX]], %arg2 : tensor<64xf32>
// CHECK: %[[DIFF:.*]] = stablehlo.subtract %[[SX]], %arg3 : tensor<64xf32>
// CHECK: %[[DOT:.*]] = stablehlo.dot_general %arg1, %[[DIFF]], contracting_dims = [1] x [0] : (tensor<32x64xf32>, tensor<64xf32>) -> tensor<32xf32>
// CHECK-NOT: stablehlo.dot_general
// CHECK: return %[[DOT]] : tensor<32xf32>
func.func @shared_lhs_sub_one_scaled(%arg0: tensor<f32>, %arg1: tensor<32x64xf32>, %arg2: tensor<64xf32>, %arg3: tensor<64xf32>) -> tensor<32xf32> {
  %0 = stablehlo.dot_general %arg1, %arg2, contracting_dims = [1] x [0] : (tensor<32x64xf32>, tensor<64xf32>) -> tensor<32xf32>
  %1 = stablehlo.dot_general %arg1, %arg3, contracting_dims = [1] x [0] : (tensor<32x64xf32>, tensor<64xf32>) -> tensor<32xf32>
  %2 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<f32>) -> tensor<32xf32>
  %3 = stablehlo.multiply %2, %0 : tensor<32xf32>
  %4 = stablehlo.subtract %3, %1 : tensor<32xf32>
  return %4 : tensor<32xf32>
}

// Equal scalar on both sides: alpha * (A * x) + alpha * (B * x) -> alpha * ((A + B) * x)
// CHECK-LABEL: func.func @shared_rhs_same_scalar(
// CHECK: %[[B:.*]] = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<f32>) -> tensor<64xf32>
// CHECK: %[[SUM:.*]] = stablehlo.add %arg1, %arg2 : tensor<64x64xf32>
// CHECK: %[[DOT:.*]] = stablehlo.dot_general %[[SUM]], %arg3, contracting_dims = [1] x [0] : (tensor<64x64xf32>, tensor<64xf32>) -> tensor<64xf32>
// CHECK-NOT: stablehlo.dot_general
// CHECK: %[[OUT:.*]] = stablehlo.multiply %[[B]], %[[DOT]] : tensor<64xf32>
// CHECK: return %[[OUT]] : tensor<64xf32>
func.func @shared_rhs_same_scalar(%arg0: tensor<f32>, %arg1: tensor<64x64xf32>, %arg2: tensor<64x64xf32>, %arg3: tensor<64xf32>) -> tensor<64xf32> {
  %0 = stablehlo.dot_general %arg1, %arg3, contracting_dims = [1] x [0] : (tensor<64x64xf32>, tensor<64xf32>) -> tensor<64xf32>
  %1 = stablehlo.dot_general %arg2, %arg3, contracting_dims = [1] x [0] : (tensor<64x64xf32>, tensor<64xf32>) -> tensor<64xf32>
  %2 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<f32>) -> tensor<64xf32>
  %3 = stablehlo.multiply %2, %0 : tensor<64xf32>
  %4 = stablehlo.multiply %2, %1 : tensor<64xf32>
  %5 = stablehlo.add %3, %4 : tensor<64xf32>
  return %5 : tensor<64xf32>
}
