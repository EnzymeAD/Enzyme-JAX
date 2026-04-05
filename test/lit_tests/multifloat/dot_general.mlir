// RUN: enzymexlamlir-opt %s --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=first" | FileCheck %s --check-prefix=FIRST
// RUN: enzymexlamlir-opt %s --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=last" | FileCheck %s --check-prefix=LAST
// RUN: enzymexlamlir-opt %s --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=tuple" | FileCheck %s --check-prefix=TUPLE

func.func @dot_general(%arg0: tensor<2xf64>, %arg1: tensor<2xf64>) -> tensor<f64> {
  // FIRST-LABEL: @dot_general
  // FIRST: %[[LHS_HI:.*]] = stablehlo.reshape %{{.*}} : (tensor<1x2xf32>) -> tensor<2xf32>
  // FIRST: %[[LHS_LO:.*]] = stablehlo.reshape %{{.*}} : (tensor<1x2xf32>) -> tensor<2xf32>
  // FIRST: %[[RHS_HI:.*]] = stablehlo.reshape %{{.*}} : (tensor<1x2xf32>) -> tensor<2xf32>
  // FIRST: %[[RHS_LO:.*]] = stablehlo.reshape %{{.*}} : (tensor<1x2xf32>) -> tensor<2xf32>
  // FIRST: %[[HI_HI:.*]] = stablehlo.dot_general %[[LHS_HI]], %[[RHS_HI]], contracting_dims = [0] x [0] : (tensor<2xf32>, tensor<2xf32>) -> tensor<f32>
  // FIRST: %[[HI_LO:.*]] = stablehlo.dot_general %[[LHS_HI]], %[[RHS_LO]], contracting_dims = [0] x [0] : (tensor<2xf32>, tensor<2xf32>) -> tensor<f32>
  // FIRST: %[[LO_HI:.*]] = stablehlo.dot_general %[[LHS_LO]], %[[RHS_HI]], contracting_dims = [0] x [0] : (tensor<2xf32>, tensor<2xf32>) -> tensor<f32>
  // FIRST: %[[LO_LO:.*]] = stablehlo.dot_general %[[LHS_LO]], %[[RHS_LO]], contracting_dims = [0] x [0] : (tensor<2xf32>, tensor<2xf32>) -> tensor<f32>
  // FIRST-NOT: stablehlo.convert %{{.*}} : (tensor<1x2xf32>) -> tensor<1x2xf64>
  // LAST-LABEL: @dot_general
  // TUPLE-LABEL: @dot_general

  %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [0] x [0] : (tensor<2xf64>, tensor<2xf64>) -> tensor<f64>
  return %0 : tensor<f64>
}
