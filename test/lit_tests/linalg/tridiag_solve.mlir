// RUN: enzymexlamlir-opt %s | FileCheck %s
// RUN: enzymexlamlir-opt --lower-enzymexla-linalg %s | FileCheck %s --check-prefix=LOWER

func.func @trisolve(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>, %arg3: tensor<64x32xf32>) -> (tensor<64x32xf32>, tensor<i32>) {
    %0:2 = enzymexla.linalg.tridiagonal_solve %arg0, %arg1, %arg2, %arg3: (tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x32xf32>) -> (tensor<64x32xf32>, tensor<i32>)
    return %0#0, %0#1 : tensor<64x32xf32>, tensor<i32>
}

func.func @trisolve_batched(%arg0: tensor<4x64xf64>, %arg1: tensor<4x64xf64>, %arg2: tensor<4x64xf64>, %arg3: tensor<4x64x1xf64>) -> (tensor<4x64x1xf64>, tensor<4xi64>) {
    %0:2 = enzymexla.linalg.tridiagonal_solve %arg0, %arg1, %arg2, %arg3: (tensor<4x64xf64>, tensor<4x64xf64>, tensor<4x64xf64>, tensor<4x64x1xf64>) -> (tensor<4x64x1xf64>, tensor<4xi64>)
    return %0#0, %0#1 : tensor<4x64x1xf64>, tensor<4xi64>
}

// CHECK: func.func @trisolve(%[[ARG0:[^:]+]]: tensor<64xf32>, %[[ARG1:[^:]+]]: tensor<64xf32>, %[[ARG2:[^:]+]]: tensor<64xf32>, %[[ARG3:[^:]+]]: tensor<64x32xf32>) -> (tensor<64x32xf32>, tensor<i32>) {
// CHECK-NEXT: %[[X:.*]], %[[INFO:.*]] = enzymexla.linalg.tridiagonal_solve %[[ARG0]], %[[ARG1]], %[[ARG2]], %[[ARG3]] : (tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x32xf32>) -> (tensor<64x32xf32>, tensor<i32>)
// CHECK-NEXT: return %[[X]], %[[INFO]] : tensor<64x32xf32>, tensor<i32>
// CHECK-NEXT: }

// CHECK: func.func @trisolve_batched(%[[ARG0:[^:]+]]: tensor<4x64xf64>, %[[ARG1:[^:]+]]: tensor<4x64xf64>, %[[ARG2:[^:]+]]: tensor<4x64xf64>, %[[ARG3:[^:]+]]: tensor<4x64x1xf64>) -> (tensor<4x64x1xf64>, tensor<4xi64>) {
// CHECK-NEXT: %[[X:.*]], %[[INFO:.*]] = enzymexla.linalg.tridiagonal_solve %[[ARG0]], %[[ARG1]], %[[ARG2]], %[[ARG3]] : (tensor<4x64xf64>, tensor<4x64xf64>, tensor<4x64xf64>, tensor<4x64x1xf64>) -> (tensor<4x64x1xf64>, tensor<4xi64>)
// CHECK-NEXT: return %[[X]], %[[INFO]] : tensor<4x64x1xf64>, tensor<4xi64>
// CHECK-NEXT: }

// LOWER: func.func @trisolve(%[[ARG0:[^:]+]]: tensor<64xf32>, %[[ARG1:[^:]+]]: tensor<64xf32>, %[[ARG2:[^:]+]]: tensor<64xf32>, %[[ARG3:[^:]+]]: tensor<64x32xf32>) -> (tensor<64x32xf32>, tensor<i32>) {
// LOWER-NEXT: %[[X:.*]], %[[INFO:.*]] = enzymexla.lapack.gtsv %[[ARG0]], %[[ARG1]], %[[ARG2]], %[[ARG3]] : (tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x32xf32>) -> (tensor<64x32xf32>, tensor<i32>)
// LOWER-NEXT: return %[[X]], %[[INFO]] : tensor<64x32xf32>, tensor<i32>
// LOWER-NEXT: }

// LOWER: func.func @trisolve_batched(%[[ARG0:[^:]+]]: tensor<4x64xf64>, %[[ARG1:[^:]+]]: tensor<4x64xf64>, %[[ARG2:[^:]+]]: tensor<4x64xf64>, %[[ARG3:[^:]+]]: tensor<4x64x1xf64>) -> (tensor<4x64x1xf64>, tensor<4xi64>) {
// LOWER-NEXT: %[[X:.*]], %[[INFO:.*]] = enzymexla.lapack.gtsv %[[ARG0]], %[[ARG1]], %[[ARG2]], %[[ARG3]] : (tensor<4x64xf64>, tensor<4x64xf64>, tensor<4x64xf64>, tensor<4x64x1xf64>) -> (tensor<4x64x1xf64>, tensor<4xi64>)
// LOWER-NEXT: return %[[X]], %[[INFO]] : tensor<4x64x1xf64>, tensor<4xi64>
// LOWER-NEXT: }
