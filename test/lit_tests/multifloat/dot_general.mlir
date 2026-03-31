// RUN: enzymexlamlir-opt %s --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=first" | FileCheck %s --check-prefix=FIRST
// RUN: enzymexlamlir-opt %s --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=last" | FileCheck %s --check-prefix=LAST
// RUN: enzymexlamlir-opt %s --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=tuple" | FileCheck %s --check-prefix=TUPLE

func.func @dot_general(%arg0: tensor<2xf64>, %arg1: tensor<2xf64>) -> tensor<f64> {
  // FIRST-LABEL: @dot_general
  // LAST-LABEL: @dot_general
  // TUPLE-LABEL: @dot_general

  %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [0] x [0] : (tensor<2xf64>, tensor<2xf64>) -> tensor<f64>
  return %0 : tensor<f64>
}
