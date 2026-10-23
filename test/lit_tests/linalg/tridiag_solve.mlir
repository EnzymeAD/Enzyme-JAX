// RUN: enzymexlamlir-opt %s | FileCheck %s
// RUN: enzymexlamlir-opt --lower-enzymexla-linalg %s | FileCheck %s --check-prefix=LOWER

func.func @trisolve(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>, %arg3: tensor<64x32xf32>) -> tensor<64x32xf32> {
    %0 = enzymexla.linalg.tridiagonal_solve %arg0, %arg1, %arg2, %arg3: (tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x32xf32>) -> tensor<64x32xf32>
    return %0 : tensor<64x32xf32>
}

func.func @trisolve_batched(%arg0: tensor<4x64xf64>, %arg1: tensor<4x64xf64>, %arg2: tensor<4x64xf64>, %arg3: tensor<4x64x1xf64>) -> tensor<4x64x1xf64> {
    %0 = enzymexla.linalg.tridiagonal_solve %arg0, %arg1, %arg2, %arg3: (tensor<4x64xf64>, tensor<4x64xf64>, tensor<4x64xf64>, tensor<4x64x1xf64>) -> tensor<4x64x1xf64>
    return %0 : tensor<4x64x1xf64>
}

// CHECK: func.func @trisolve(%[[ARG0:[^:]+]]: tensor<64xf32>, %[[ARG1:[^:]+]]: tensor<64xf32>, %[[ARG2:[^:]+]]: tensor<64xf32>, %[[ARG3:[^:]+]]: tensor<64x32xf32>) -> tensor<64x32xf32> {
// CHECK-NEXT: %[[LINALG_0:.*]] = enzymexla.linalg.tridiagonal_solve %[[ARG0]], %[[ARG1]], %[[ARG2]], %[[ARG3]] : (tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x32xf32>) -> tensor<64x32xf32>
// CHECK-NEXT: return %[[LINALG_0]] : tensor<64x32xf32>
// CHECK-NEXT: }

// CHECK: func.func @trisolve_batched(%[[ARG0:[^:]+]]: tensor<4x64xf64>, %[[ARG1:[^:]+]]: tensor<4x64xf64>, %[[ARG2:[^:]+]]: tensor<4x64xf64>, %[[ARG3:[^:]+]]: tensor<4x64x1xf64>) -> tensor<4x64x1xf64> {
// CHECK-NEXT: %[[LINALG_0:.*]] = enzymexla.linalg.tridiagonal_solve %[[ARG0]], %[[ARG1]], %[[ARG2]], %[[ARG3]] : (tensor<4x64xf64>, tensor<4x64xf64>, tensor<4x64xf64>, tensor<4x64x1xf64>) -> tensor<4x64x1xf64>
// CHECK-NEXT: return %[[LINALG_0]] : tensor<4x64x1xf64>
// CHECK-NEXT: }

// LOWER: func.func @trisolve(%[[ARG0:[^:]+]]: tensor<64xf32>, %[[ARG1:[^:]+]]: tensor<64xf32>, %[[ARG2:[^:]+]]: tensor<64xf32>, %[[ARG3:[^:]+]]: tensor<64x32xf32>) -> tensor<64x32xf32> {
// LOWER-NEXT: %[[CONSTANT_0:.*]] = stablehlo.constant dense<0x7FC00000> : tensor<64x32xf32>
// LOWER-NEXT: %[[CONSTANT_1:.*]] = stablehlo.constant dense<0> : tensor<i32>
// LOWER-NEXT: %[[VAL_0:.*]], %[[LAPACK_0:.*]] = enzymexla.lapack.gtsv %[[ARG0]], %[[ARG1]], %[[ARG2]], %[[ARG3]] : (tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x32xf32>) -> (tensor<64x32xf32>, tensor<i32>)
// LOWER-NEXT: %[[COMPARE_0:.*]] = stablehlo.compare EQ, %[[LAPACK_0]], %[[CONSTANT_1]] : (tensor<i32>, tensor<i32>) -> tensor<i1>
// LOWER-NEXT: %[[BROADCAST_IN_DIM_0:.*]] = stablehlo.broadcast_in_dim %[[COMPARE_0]], dims = [] : (tensor<i1>) -> tensor<64x32xi1>
// LOWER-NEXT: %[[SELECT_0:.*]] = stablehlo.select %[[BROADCAST_IN_DIM_0]], %[[VAL_0]], %[[CONSTANT_0]] : tensor<64x32xi1>, tensor<64x32xf32>
// LOWER-NEXT: return %[[SELECT_0]] : tensor<64x32xf32>
// LOWER-NEXT: }

// LOWER: func.func @trisolve_batched(%[[ARG0:[^:]+]]: tensor<4x64xf64>, %[[ARG1:[^:]+]]: tensor<4x64xf64>, %[[ARG2:[^:]+]]: tensor<4x64xf64>, %[[ARG3:[^:]+]]: tensor<4x64x1xf64>) -> tensor<4x64x1xf64> {
// LOWER-NEXT: %[[CONSTANT_0:.*]] = stablehlo.constant dense<0x7FF8000000000000> : tensor<4x64x1xf64>
// LOWER-NEXT: %[[CONSTANT_1:.*]] = stablehlo.constant dense<0> : tensor<4xi32>
// LOWER-NEXT: %[[VAL_0:.*]], %[[LAPACK_0:.*]] = enzymexla.lapack.gtsv %[[ARG0]], %[[ARG1]], %[[ARG2]], %[[ARG3]] : (tensor<4x64xf64>, tensor<4x64xf64>, tensor<4x64xf64>, tensor<4x64x1xf64>) -> (tensor<4x64x1xf64>, tensor<4xi32>)
// LOWER-NEXT: %[[COMPARE_0:.*]] = stablehlo.compare EQ, %[[LAPACK_0]], %[[CONSTANT_1]] : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi1>
// LOWER-NEXT: %[[BROADCAST_IN_DIM_0:.*]] = stablehlo.broadcast_in_dim %[[COMPARE_0]], dims = [0] : (tensor<4xi1>) -> tensor<4x64x1xi1>
// LOWER-NEXT: %[[SELECT_0:.*]] = stablehlo.select %[[BROADCAST_IN_DIM_0]], %[[VAL_0]], %[[CONSTANT_0]] : tensor<4x64x1xi1>, tensor<4x64x1xf64>
// LOWER-NEXT: return %[[SELECT_0]] : tensor<4x64x1xf64>
// LOWER-NEXT: }
