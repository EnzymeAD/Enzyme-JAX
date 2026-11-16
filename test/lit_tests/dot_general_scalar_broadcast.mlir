// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=dot_general_simplify" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s
// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=dot_general_simplify;reduce_mul_to_dot_general;dot_general_broadcast_in_dim;dot_general_broadcast_in_dim_sort_dims;dot_general_remove_batch_dimensions;broadcast_in_dim_op_canon" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s --check-prefix=GEMM

// sum_k alpha * (a[k] * b[k])
func.func @lhs_scalar(%alpha: tensor<f64>, %a: tensor<1000xf64>, %b: tensor<1000xf64>) -> tensor<f64> {
  %0 = stablehlo.multiply %a, %b : tensor<1000xf64>
  %1 = stablehlo.broadcast_in_dim %alpha, dims = [] : (tensor<f64>) -> tensor<1000xf64>
  %2 = stablehlo.dot_general %1, %0, contracting_dims = [0] x [0] : (tensor<1000xf64>, tensor<1000xf64>) -> tensor<f64>
  return %2 : tensor<f64>
}

// CHECK-LABEL: func.func @lhs_scalar
// CHECK-SAME:    (%[[ALPHA:.+]]: tensor<f64>, %[[A:.+]]: tensor<1000xf64>, %[[B:.+]]: tensor<1000xf64>)
// CHECK-NOT:     stablehlo.dot_general
// CHECK:         %[[AB:.+]] = stablehlo.multiply %[[A]], %[[B]] : tensor<1000xf64>
// CHECK-NEXT:    %[[SUM:.+]] = stablehlo.reduce(%[[AB]] init: %{{.+}}) applies stablehlo.add across dimensions = [0] : (tensor<1000xf64>, tensor<f64>) -> tensor<f64>
// CHECK-NEXT:    %[[SUMB:.+]] = stablehlo.broadcast_in_dim %[[SUM]], dims = [] : (tensor<f64>) -> tensor<f64>
// CHECK-NEXT:    %[[AL:.+]] = stablehlo.broadcast_in_dim %[[ALPHA]], dims = [] : (tensor<f64>) -> tensor<f64>
// CHECK-NEXT:    %[[R:.+]] = stablehlo.multiply %[[SUMB]], %[[AL]] : tensor<f64>
// CHECK-NEXT:    return %[[R]]

func.func @rhs_scalar(%s: tensor<f32>, %x: tensor<4x5x6xf32>) -> tensor<4x5x3xf32> {
  %0 = stablehlo.broadcast_in_dim %s, dims = [] : (tensor<f32>) -> tensor<4x6x3xf32>
  %1 = stablehlo.dot_general %x, %0, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<4x5x6xf32>, tensor<4x6x3xf32>) -> tensor<4x5x3xf32>
  return %1 : tensor<4x5x3xf32>
}

// CHECK-LABEL: func.func @rhs_scalar
// CHECK-SAME:    (%[[S:.+]]: tensor<f32>, %[[X:.+]]: tensor<4x5x6xf32>)
// CHECK-NOT:     stablehlo.dot_general
// CHECK:         %[[SUM:.+]] = stablehlo.reduce(%[[X]] init: %{{.+}}) applies stablehlo.add across dimensions = [2] : (tensor<4x5x6xf32>, tensor<f32>) -> tensor<4x5xf32>
// CHECK-NEXT:    %[[SUMB:.+]] = stablehlo.broadcast_in_dim %[[SUM]], dims = [0, 1] : (tensor<4x5xf32>) -> tensor<4x5x3xf32>
// CHECK-NEXT:    %[[SB:.+]] = stablehlo.broadcast_in_dim %[[S]], dims = [] : (tensor<f32>) -> tensor<4x5x3xf32>
// CHECK-NEXT:    %[[R:.+]] = stablehlo.multiply %[[SUMB]], %[[SB]] : tensor<4x5x3xf32>
// CHECK-NEXT:    return %[[R]]

// C[i,j] = sum_k alpha * (A[i,k] * B[j,k]), with the loops batched
func.func @gemm_nt(%alpha: tensor<f64>, %A: tensor<10x10xf64>, %B: tensor<10x10xf64>) -> tensor<10x10xf64> {
  %0 = stablehlo.broadcast_in_dim %alpha, dims = [] : (tensor<f64>) -> tensor<10xf64>
  %1 = stablehlo.broadcast_in_dim %A, dims = [0, 2] : (tensor<10x10xf64>) -> tensor<10x10x10xf64>
  %2 = stablehlo.broadcast_in_dim %B, dims = [1, 2] : (tensor<10x10xf64>) -> tensor<10x10x10xf64>
  %3 = stablehlo.multiply %1, %2 : tensor<10x10x10xf64>
  %4 = stablehlo.dot_general %0, %3, contracting_dims = [0] x [2] : (tensor<10xf64>, tensor<10x10x10xf64>) -> tensor<10x10xf64>
  return %4 : tensor<10x10xf64>
}

// GEMM-LABEL: func.func @gemm_nt
// GEMM-SAME:    (%[[ALPHA:.+]]: tensor<f64>, %[[A:.+]]: tensor<10x10xf64>, %[[B:.+]]: tensor<10x10xf64>)
// GEMM-NEXT:    %[[D:.+]] = stablehlo.dot_general %[[A]], %[[B]], contracting_dims = [1] x [1] : (tensor<10x10xf64>, tensor<10x10xf64>) -> tensor<10x10xf64>
// GEMM-NEXT:    %[[AL:.+]] = stablehlo.broadcast_in_dim %[[ALPHA]], dims = [] : (tensor<f64>) -> tensor<10x10xf64>
// GEMM-NEXT:    %[[R:.+]] = stablehlo.multiply %[[D]], %[[AL]] : tensor<10x10xf64>
// GEMM-NEXT:    return %[[R]]
