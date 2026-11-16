// RUN: enzymexlamlir-opt %s --lower-enzymexla-math | FileCheck %s --check-prefix=LOWER
// RUN: enzymexlamlir-opt %s --lower-enzymexla-math | stablehlo-translate - --interpret

func.func @apply_elu(%arg0: tensor<6xf32>) -> tensor<6xf32> {
  %0 = enzymexla.math.elu %arg0, alpha = 1.500000e+00 : (tensor<6xf32>) -> tensor<6xf32>
  return %0 : tensor<6xf32>
}

func.func @apply_elu_scalar(%arg0: tensor<f32>) -> tensor<f32> {
  %0 = enzymexla.math.elu %arg0, alpha = 1.000000e+00 : (tensor<f32>) -> tensor<f32>
  return %0 : tensor<f32>
}

func.func @main() {
  %x = stablehlo.constant dense<[0.0, -1.0, 1.0, 2.0, 100.0, -5.0]> : tensor<6xf32>
  %expected = stablehlo.constant dense<[0.0, -0.948180854, 1.0, 2.0, 100.0, -1.48989308]> : tensor<6xf32>

  %res = func.call @apply_elu(%x) : (tensor<6xf32>) -> tensor<6xf32>
  check.expect_almost_eq %res, %expected : tensor<6xf32>

  // NaN propagation test: elu(NaN) == NaN
  %zero = stablehlo.constant dense<0.0> : tensor<f32>
  %nan = stablehlo.divide %zero, %zero : tensor<f32>
  %nan_res = func.call @apply_elu_scalar(%nan) : (tensor<f32>) -> tensor<f32>
  %is_nan = stablehlo.compare  NE, %nan_res, %nan_res,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
  check.expect_eq_const %is_nan, dense<true> : tensor<i1>

  return
}

// LOWER-LABEL: func.func @apply_elu(%arg0: tensor<6xf32>) -> tensor<6xf32> {
// LOWER-DAG:    %[[ALPHA:.*]] = stablehlo.constant dense<1.500000e+00> : tensor<6xf32>
// LOWER-DAG:    %[[ZERO:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<6xf32>
// LOWER-DAG:    %[[CMP:.*]] = stablehlo.compare  GT, %arg0, %[[ZERO]],  FLOAT : (tensor<6xf32>, tensor<6xf32>) -> tensor<6xi1>
// LOWER-DAG:    %[[MIN:.*]] = stablehlo.minimum %arg0, %[[ZERO]] : tensor<6xf32>
// LOWER-DAG:    %[[EXPM1:.*]] = stablehlo.exponential_minus_one %[[MIN]] : tensor<6xf32>
// LOWER-DAG:    %[[MUL:.*]] = stablehlo.multiply %[[ALPHA]], %[[EXPM1]] : tensor<6xf32>
// LOWER-NEXT:   %[[RES:.*]] = stablehlo.select %[[CMP]], %arg0, %[[MUL]] : tensor<6xi1>, tensor<6xf32>
// LOWER-NEXT:   return %[[RES]] : tensor<6xf32>
// LOWER-NEXT: }
