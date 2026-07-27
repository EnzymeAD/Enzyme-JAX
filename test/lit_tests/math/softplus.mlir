// RUN: enzymexlamlir-opt %s --lower-enzymexla-math | FileCheck %s --check-prefix=LOWER
// RUN: enzymexlamlir-opt %s --lower-enzymexla-math | stablehlo-translate - --interpret

func.func @apply_softplus(%arg0: tensor<7xf32>) -> tensor<7xf32> {
  %0 = enzymexla.math.softplus %arg0 : (tensor<7xf32>) -> tensor<7xf32>
  return %0 : tensor<7xf32>
}

func.func @apply_softplus_scalar(%arg0: tensor<f32>) -> tensor<f32> {
  %0 = enzymexla.math.softplus %arg0 : (tensor<f32>) -> tensor<f32>
  return %0 : tensor<f32>
}

func.func @main() {
  // Numerics test for stable softplus form on finite inputs.
  %input = stablehlo.constant dense<[-1000.0, -2.0, -1.0, 0.0, 1.0, 2.0, 1000.0]> : tensor<7xf32>
  %expected = stablehlo.constant dense<[0.0, 0.126928001, 0.313261688, 0.693147182, 1.31326175, 2.12692809, 1000.0]> : tensor<7xf32>

  %res = func.call @apply_softplus(%input) : (tensor<7xf32>) -> tensor<7xf32>
  check.expect_almost_eq %res, %expected : tensor<7xf32>

  // NaN propagation test: softplus(NaN) == NaN
  %zero = stablehlo.constant dense<0.0> : tensor<f32>
  %nan = stablehlo.divide %zero, %zero : tensor<f32>
  %nan_res = func.call @apply_softplus_scalar(%nan) : (tensor<f32>) -> tensor<f32>
  %is_nan = stablehlo.compare  NE, %nan_res, %nan_res,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
  check.expect_eq_const %is_nan, dense<true> : tensor<i1>

  return
}

// LOWER: func.func @apply_softplus(%arg0: tensor<7xf32>) -> tensor<7xf32> {
// LOWER-DAG:    %[[CST:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<7xf32>
// LOWER-DAG:    %[[CMP:.*]] = stablehlo.compare  NE, %arg0, %arg0,  FLOAT : (tensor<7xf32>, tensor<7xf32>) -> tensor<7xi1>
// LOWER-DAG:    %[[MAX:.*]] = stablehlo.maximum %arg0, %[[CST]] : tensor<7xf32>
// LOWER-DAG:    %[[ABS:.*]] = stablehlo.abs %arg0 : tensor<7xf32>
// LOWER-DAG:    %[[NEG:.*]] = stablehlo.negate %[[ABS]] : tensor<7xf32>
// LOWER-DAG:    %[[EXP:.*]] = stablehlo.exponential %[[NEG]] : tensor<7xf32>
// LOWER-DAG:    %[[LOG:.*]] = stablehlo.log_plus_one %[[EXP]] : tensor<7xf32>
// LOWER-DAG:    %[[ADD:.*]] = stablehlo.add %[[MAX]], %[[LOG]] : tensor<7xf32>
// LOWER-NEXT:    %[[RES:.*]] = stablehlo.select %[[CMP]], %arg0, %[[ADD]] : tensor<7xi1>, tensor<7xf32>
// LOWER-NEXT:    return %[[RES]] : tensor<7xf32>
// LOWER-NEXT:  }

// LOWER: func.func @apply_softplus_scalar(%arg0: tensor<f32>) -> tensor<f32> {
// LOWER-DAG:    %[[CST:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// LOWER-DAG:    %[[CMP:.*]] = stablehlo.compare  NE, %arg0, %arg0,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
// LOWER-DAG:    %[[MAX:.*]] = stablehlo.maximum %arg0, %[[CST]] : tensor<f32>
// LOWER-DAG:    %[[ABS:.*]] = stablehlo.abs %arg0 : tensor<f32>
// LOWER-DAG:    %[[NEG:.*]] = stablehlo.negate %[[ABS]] : tensor<f32>
// LOWER-DAG:    %[[EXP:.*]] = stablehlo.exponential %[[NEG]] : tensor<f32>
// LOWER-DAG:    %[[LOG:.*]] = stablehlo.log_plus_one %[[EXP]] : tensor<f32>
// LOWER-DAG:    %[[ADD:.*]] = stablehlo.add %[[MAX]], %[[LOG]] : tensor<f32>
// LOWER-NEXT:    %[[RES:.*]] = stablehlo.select %[[CMP]], %arg0, %[[ADD]] : tensor<i1>, tensor<f32>
// LOWER-NEXT:    return %[[RES]] : tensor<f32>
// LOWER-NEXT:  }
