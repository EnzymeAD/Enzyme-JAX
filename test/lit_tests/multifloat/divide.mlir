// RUN: enzymexlamlir-opt %s --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=first" | FileCheck %s

func.func @divide(%arg0: tensor<2xf64>, %arg1: tensor<2xf64>) -> tensor<2xf64> {
  // CHECK-LABEL: @divide
  // CHECK: %[[X_HI:.*]] = stablehlo.slice %{{.*}} [0:1, 0:2] : (tensor<2x2xf32>) -> tensor<1x2xf32>
  // CHECK: %[[X_LO:.*]] = stablehlo.slice %{{.*}} [1:2, 0:2] : (tensor<2x2xf32>) -> tensor<1x2xf32>
  // CHECK: %[[Y_HI:.*]] = stablehlo.slice %{{.*}} [0:1, 0:2] : (tensor<2x2xf32>) -> tensor<1x2xf32>
  // CHECK: %[[Y_LO:.*]] = stablehlo.slice %{{.*}} [1:2, 0:2] : (tensor<2x2xf32>) -> tensor<1x2xf32>
  // CHECK: %[[Z_HI:.*]] = stablehlo.divide %[[X_HI]], %[[Y_HI]] : tensor<1x2xf32>
  // CHECK-DAG: %[[P:.*]] = stablehlo.multiply %[[Z_HI]], %[[Y_HI]] : tensor<1x2xf32>
  // CHECK-DAG: %[[SPLIT_CST:.*]] = stablehlo.constant dense<4.097000e+03> : tensor<1x2xf32>
  // CHECK-DAG: %[[C:.*]] = stablehlo.multiply %[[Z_HI]], %[[SPLIT_CST]] : tensor<1x2xf32>
  // CHECK-DAG: %[[A_BIG:.*]] = stablehlo.subtract %[[C]], %[[Z_HI]] : tensor<1x2xf32>
  // CHECK-DAG: %[[A_HI_SPLIT:.*]] = stablehlo.subtract %[[C]], %[[A_BIG]] : tensor<1x2xf32>
  // CHECK-DAG: %[[A_LO_SPLIT:.*]] = stablehlo.subtract %[[Z_HI]], %[[A_HI_SPLIT]] : tensor<1x2xf32>
  // CHECK: return
  
  %0 = stablehlo.divide %arg0, %arg1 : tensor<2xf64>
  return %0 : tensor<2xf64>
}
