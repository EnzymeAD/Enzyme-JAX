// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=first" %s | FileCheck %s

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
// CHECK: func @while(%arg0: tensor<f64>) -> tensor<f64> {
// CHECK: stablehlo.while
// CHECK: cond
// CHECK: stablehlo.compare
// CHECK: stablehlo.return
// CHECK: do
// CHECK: stablehlo.add
// CHECK: stablehlo.return
