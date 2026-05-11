// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=first expansion-size=2 dot-general-to-reduce=false" %s | FileCheck %s --check-prefix=FIRST
// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=last expansion-size=2 dot-general-to-reduce=false" %s | FileCheck %s --check-prefix=LAST
// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=tuple expansion-size=2 dot-general-to-reduce=false convert-signatures=true" %s | FileCheck %s --check-prefix=TUPLE

func.func @power_test(%arg0: tensor<f64>, %arg1: tensor<f64>) -> tensor<f64> {
  // FIRST-LABEL: @power_test
  // FIRST-NOT: stablehlo.power {{.*}} : tensor<f64>
  
  // LAST-LABEL: @power_test
  // LAST-NOT: stablehlo.power {{.*}} : tensor<f64>

  // TUPLE-LABEL: @power_test
  // TUPLE-NOT: stablehlo.power {{.*}} : tensor<f64>

  %0 = stablehlo.power %arg0, %arg1 : tensor<f64>
  return %0 : tensor<f64>
}

func.func @power_neg_base(%arg0: tensor<f64>) -> tensor<f64> {
  // TUPLE-LABEL: @power_neg_base
  // TUPLE-NOT: stablehlo.power {{.*}} : tensor<f64>
  %cst = stablehlo.constant dense<-2.0> : tensor<f64>
  %0 = stablehlo.power %cst, %arg0 : tensor<f64>
  return %0 : tensor<f64>
}

func.func @power_const_exp(%arg0: tensor<f64>) -> tensor<f64> {
  // TUPLE-LABEL: @power_const_exp
  // TUPLE-NOT: stablehlo.power {{.*}} : tensor<f64>
  %cst = stablehlo.constant dense<0.1> : tensor<f64>
  %0 = stablehlo.power %arg0, %cst : tensor<f64>
  return %0 : tensor<f64>
}
