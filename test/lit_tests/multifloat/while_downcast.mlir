// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 expansion-size=1 convert-signatures=true" --canonicalize %s | FileCheck %s

func.func @while(%arg0: tensor<f64>) -> tensor<f64> {
  %cst = stablehlo.constant dense<1.000000e+01> : tensor<f64>
  %0:2 = stablehlo.while(%iterArg0 = %arg0, %iterArg1 = %cst) : tensor<f64>, tensor<f64>
    cond {
      %1 = stablehlo.compare LT, %iterArg0, %iterArg1 : (tensor<f64>, tensor<f64>) -> tensor<i1>
      stablehlo.return %1 : tensor<i1>
    } do {
      %1 = stablehlo.add %iterArg0, %iterArg0 : tensor<f64>
      stablehlo.return %1, %iterArg1 : tensor<f64>, tensor<f64>
    }
  return %0#0 : tensor<f64>
}

// CHECK-LABEL: func.func @while
// CHECK-SAME: (%[[ARG0:.*]]: tensor<f32>) -> tensor<f32> {
// CHECK: %[[CST:.*]] = stablehlo.constant dense<1.000000e+01> : tensor<f32>
// CHECK: %[[RES:.*]]:2 = stablehlo.while(%[[ITER0:.*]] = %[[ARG0]], %[[ITER1:.*]] = %[[CST]]) : tensor<f32>, tensor<f32>
// CHECK: cond {
// CHECK:   %[[CMP:.*]] = stablehlo.compare LT, %[[ITER0]], %[[ITER1]] : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK:   stablehlo.return %[[CMP]] : tensor<i1>
// CHECK: } do {
// CHECK:   %[[ADD:.*]] = stablehlo.add %[[ITER0]], %[[ITER0]] : tensor<f32>
// CHECK:   stablehlo.return %[[ADD]], %[[ITER1]] : tensor<f32>, tensor<f32>
// CHECK: }
// CHECK: return %[[RES]]#0 : tensor<f32>
// CHECK: }
