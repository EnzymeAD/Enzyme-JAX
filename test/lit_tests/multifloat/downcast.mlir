// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 expansion-size=1" %s | FileCheck %s

func.func @add(%arg0: tensor<f64>, %arg1: tensor<f64>) -> tensor<f64> {
  %0 = stablehlo.add %arg0, %arg1 : tensor<f64>
  return %0 : tensor<f64>
}

// CHECK-LABEL: func.func @add
// CHECK-SAME: (%[[ARG0:.*]]: tensor<f64>, %[[ARG1:.*]]: tensor<f64>) -> tensor<f64> {
// CHECK-DAG: %[[C1:.*]] = builtin.unrealized_conversion_cast %[[ARG0]] : tensor<f64> to tensor<f32>
// CHECK-DAG: %[[C2:.*]] = builtin.unrealized_conversion_cast %[[ARG1]] : tensor<f64> to tensor<f32>
// CHECK: %[[RES:.*]] = stablehlo.add %[[C1]], %[[C2]] : tensor<f32>
// CHECK: %[[RES2:.*]] = builtin.unrealized_conversion_cast %[[RES]] : tensor<f32> to tensor<f64>
// CHECK: return %[[RES2]] : tensor<f64>
// CHECK: }
