// RUN: enzymexlamlir-opt %s --lower-enzymexla-ml | FileCheck %s --check-prefix=LOWER
// RUN: enzymexlamlir-opt %s --lower-enzymexla-ml | stablehlo-translate - --interpret

func.func @apply_softplus(%arg0: tensor<7xf32>) -> tensor<7xf32> {
  %0 = enzymexla.ml.softplus %arg0 : (tensor<7xf32>) -> tensor<7xf32>
  return %0 : tensor<7xf32>
}

func.func @apply_softplus_scalar(%arg0: tensor<f32>) -> tensor<f32> {
  %0 = enzymexla.ml.softplus %arg0 : (tensor<f32>) -> tensor<f32>
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
// LOWER-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<7xf32>
// LOWER-NEXT:    %0 = stablehlo.maximum %arg0, %cst : tensor<7xf32>
// LOWER-NEXT:    %1 = stablehlo.abs %arg0 : tensor<7xf32>
// LOWER-NEXT:    %2 = stablehlo.negate %1 : tensor<7xf32>
// LOWER-NEXT:    %3 = stablehlo.exponential %2 : tensor<7xf32>
// LOWER-NEXT:    %4 = stablehlo.log_plus_one %3 : tensor<7xf32>
// LOWER-NEXT:    %5 = stablehlo.add %0, %4 : tensor<7xf32>
// LOWER-NEXT:    %6 = stablehlo.compare  NE, %arg0, %arg0,  FLOAT : (tensor<7xf32>, tensor<7xf32>) -> tensor<7xi1>
// LOWER-NEXT:    %7 = stablehlo.select %6, %arg0, %5 : tensor<7xi1>, tensor<7xf32>
// LOWER-NEXT:    return %7 : tensor<7xf32>
// LOWER-NEXT:  }

// LOWER: func.func @apply_softplus_scalar(%arg0: tensor<f32>) -> tensor<f32> {
// LOWER-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// LOWER-NEXT:    %0 = stablehlo.maximum %arg0, %cst : tensor<f32>
// LOWER-NEXT:    %1 = stablehlo.abs %arg0 : tensor<f32>
// LOWER-NEXT:    %2 = stablehlo.negate %1 : tensor<f32>
// LOWER-NEXT:    %3 = stablehlo.exponential %2 : tensor<f32>
// LOWER-NEXT:    %4 = stablehlo.log_plus_one %3 : tensor<f32>
// LOWER-NEXT:    %5 = stablehlo.add %0, %4 : tensor<f32>
// LOWER-NEXT:    %6 = stablehlo.compare  NE, %arg0, %arg0,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
// LOWER-NEXT:    %7 = stablehlo.select %6, %arg0, %5 : tensor<i1>, tensor<f32>
// LOWER-NEXT:    return %7 : tensor<f32>
// LOWER-NEXT:  }
