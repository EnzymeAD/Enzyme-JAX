// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32" %s | FileCheck %s

func.func @test_add(%arg0: tensor<f64>, %arg1: tensor<f64>) -> tensor<f64> {
  %0 = stablehlo.add %arg0, %arg1 : tensor<f64>
  return %0 : tensor<f64>
}

// CHECK-LABEL: func.func @test_add
// CHECK-DAG: %[[C0:.*]] = stablehlo.convert %arg0 : (tensor<f64>) -> tensor<f32>
// CHECK-DAG: %[[C1:.*]] = stablehlo.convert %[[C0]] : (tensor<f32>) -> tensor<f64>
// CHECK-DAG: %[[C2:.*]] = stablehlo.subtract %arg0, %[[C1]] : tensor<f64>
// CHECK-DAG: %[[C3:.*]] = stablehlo.convert %[[C2]] : (tensor<f64>) -> tensor<f32>
// CHECK-DAG: %[[C4:.*]] = stablehlo.reshape %[[C0]] : (tensor<f32>) -> tensor<1xf32>
// CHECK-DAG: %[[C5:.*]] = stablehlo.reshape %[[C3]] : (tensor<f32>) -> tensor<1xf32>
// CHECK-DAG: %[[C6:.*]] = stablehlo.concatenate %[[C4]], %[[C5]], dim = 0 : (tensor<1xf32>, tensor<1xf32>) -> tensor<2xf32>
// CHECK: stablehlo.slice
// CHECK-NOT: stablehlo.reshape
// CHECK: stablehlo.slice
// CHECK-NOT: stablehlo.reshape
// CHECK: stablehlo.add
// CHECK: stablehlo.subtract
// CHECK: %[[CST:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK: %[[OUT:.*]] = stablehlo.reduce(%{{.*}} init: %[[CST]]) applies stablehlo.add across dimensions = [0] : (tensor<2xf64>, tensor<f64>) -> tensor<f64>
// CHECK: return %[[OUT]]
