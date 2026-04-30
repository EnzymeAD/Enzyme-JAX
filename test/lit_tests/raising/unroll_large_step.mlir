// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(raise-affine-to-stablehlo{err_if_not_fully_raised=true prefer_while_raising=false enable_lockstep_for=false})" | FileCheck %s

// CHECK-LABEL: func.func private @large_step_lb5_raised
// CHECK-SAME: %[[ARG1:.*]]: tensor<1xi32>
// CHECK: %[[C5:.*]] = stablehlo.constant dense<5> : tensor<i64>
// CHECK: %[[CONV5:.*]] = stablehlo.convert %[[C5]] : (tensor<i64>) -> tensor<i32>
// CHECK: return %[[ARG1]] : tensor<1xi32>

// CHECK-LABEL: func.func private @large_step_raised
// CHECK-SAME: %[[ARG0:.*]]: tensor<1xi32>
// CHECK: %[[C:.*]] = stablehlo.constant dense<0> : tensor<i64>
// CHECK: %[[CONV:.*]] = stablehlo.convert %[[C]] : (tensor<i64>) -> tensor<i32>
// CHECK: return %[[ARG0]] : tensor<1xi32>

func.func @large_step(%arg0: memref<1xi32>) {
  affine.for %arg2 = 0 to 19 step 32 {
    %0 = arith.index_cast %arg2 : index to i32
    affine.store %0, %arg0[0] : memref<1xi32>
  }
  return
}

func.func @large_step_lb5(%arg1: memref<1xi32>) {
  affine.for %arg2 = 5 to 20 step 32 {
    %0 = arith.index_cast %arg2 : index to i32
    affine.store %0, %arg1[0] : memref<1xi32>
  }
  return
}
