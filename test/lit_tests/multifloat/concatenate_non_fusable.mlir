// RUN: enzymexlamlir-opt %s --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=first" | FileCheck %s

func.func @non_fusable_concat(%arg0: tensor<2xf64>, %arg1: tensor<2xf64>, %arg2: tensor<2xf64>) -> (tensor<4xf64>, tensor<2xf64>) {
  // CHECK-LABEL: @non_fusable_concat
  // CHECK: stablehlo.add
  // CHECK: stablehlo.concatenate
  // CHECK: return

  %0 = stablehlo.add %arg0, %arg1 : tensor<2xf64>
  %1 = stablehlo.add %arg0, %arg2 : tensor<2xf64>
  
  // %0 is used in concatenate AND returned!
  %2 = stablehlo.concatenate %0, %1, dim = 0 : (tensor<2xf64>, tensor<2xf64>) -> tensor<4xf64>
  
  return %2, %0 : tensor<4xf64>, tensor<2xf64>
}
