// RUN: enzymexlamlir-opt %s -raise-affine-to-stablehlo | FileCheck %s

func.func @test_if_0(%arg0: memref<10xf32>, %arg1: memref<i1>) {
  %cond = affine.load %arg1[] : memref<i1>
  scf.if %cond {
    %v = affine.load %arg0[0] : memref<10xf32>
    %v2 = arith.addf %v, %v : f32
    affine.store %v2, %arg0[0] : memref<10xf32>
  }
  return
}

// CHECK-LABEL:   func.func private @test_if_0_raised(
// CHECK-SAME:                                        %[[VAL_0:.*]]: tensor<10xf32>,
// CHECK-SAME:                                        %[[VAL_1:.*]]: tensor<i1>) -> (tensor<10xf32>, tensor<i1>) {
// CHECK:           %[[VAL_2:.*]] = stablehlo.reshape %[[VAL_1]] : (tensor<i1>) -> tensor<i1>
// CHECK:           %[[VAL_3:.*]] = stablehlo.slice %[[VAL_0]] [0:1] : (tensor<10xf32>) -> tensor<1xf32>
// CHECK:           %[[VAL_4:.*]] = stablehlo.reshape %[[VAL_3]] : (tensor<1xf32>) -> tensor<f32>
// CHECK:           %[[VAL_5:.*]] = arith.addf %[[VAL_4]], %[[VAL_4]] : tensor<f32>
// CHECK:           %[[VAL_6:.*]] = stablehlo.constant dense<0> : tensor<i64>
// CHECK:           %[[VAL_7:.*]] = stablehlo.broadcast_in_dim %[[VAL_5]], dims = [] : (tensor<f32>) -> tensor<1xf32>
// CHECK:           %[[VAL_8:.*]] = stablehlo.dynamic_slice %[[VAL_0]], %[[VAL_6]], sizes = [1] : (tensor<10xf32>, tensor<i64>) -> tensor<1xf32>
// CHECK:           %[[VAL_9:.*]] = stablehlo.reshape %[[VAL_7]] : (tensor<1xf32>) -> tensor<f32>
// CHECK:           %[[VAL_10:.*]] = stablehlo.reshape %[[VAL_8]] : (tensor<1xf32>) -> tensor<f32>
// CHECK:           %[[VAL_11:.*]] = stablehlo.select %[[VAL_2]], %[[VAL_9]], %[[VAL_10]] : tensor<i1>, tensor<f32>
// CHECK:           %[[VAL_12:.*]] = stablehlo.reshape %[[VAL_11]] : (tensor<f32>) -> tensor<1xf32>
// CHECK:           %[[VAL_13:.*]] = stablehlo.dynamic_update_slice %[[VAL_0]], %[[VAL_12]], %[[VAL_6]] : (tensor<10xf32>, tensor<1xf32>, tensor<i64>) -> tensor<10xf32>
// CHECK:           return %[[VAL_13]], %[[VAL_1]] : tensor<10xf32>, tensor<i1>
// CHECK:         }
