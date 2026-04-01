// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32" %s | FileCheck %s

func.func @main(%arg0: tensor<2xf64>, %arg1: tensor<2xf64>) -> tensor<4xf64> {
  // CHECK-LABEL: @main
  // CHECK-DAG: %[[A:.*]] = stablehlo.concatenate %{{.*}}, %{{.*}}, dim = 0 : (tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<2x2xf32>
  // CHECK-DAG: %[[B:.*]] = stablehlo.concatenate %{{.*}}, %{{.*}}, dim = 0 : (tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<2x2xf32>
  // CHECK: %[[LHS_CONCAT:.*]] = stablehlo.concatenate %[[A]], %[[A]], dim = 1 : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x4xf32>
  // CHECK: %[[RHS_CONCAT:.*]] = stablehlo.concatenate %[[B]], %[[B]], dim = 1 : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x4xf32>
  // CHECK: %[[LHS_HI:.*]] = stablehlo.slice %[[LHS_CONCAT]] [0:1, 0:4] : (tensor<2x4xf32>) -> tensor<1x4xf32>
  // CHECK: %[[RHS_HI:.*]] = stablehlo.slice %[[RHS_CONCAT]] [0:1, 0:4] : (tensor<2x4xf32>) -> tensor<1x4xf32>
  // CHECK: stablehlo.add %[[LHS_HI]], %[[RHS_HI]] : tensor<1x4xf32>
  %0 = stablehlo.add %arg0, %arg1 : tensor<2xf64>
  %1 = stablehlo.add %arg0, %arg1 : tensor<2xf64>
  %2 = stablehlo.concatenate %0, %1, dim = 0 : (tensor<2xf64>, tensor<2xf64>) -> tensor<4xf64>
  return %2 : tensor<4xf64>
}
