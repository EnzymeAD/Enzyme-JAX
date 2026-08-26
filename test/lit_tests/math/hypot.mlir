// RUN: enzymexlamlir-opt %s --lower-enzymexla-math | FileCheck %s --check-prefix=LOWER
// RUN: enzymexlamlir-opt %s --lower-enzymexla-math --chlo-legalize-to-stablehlo | stablehlo-translate - --interpret

func.func @apply_hypot_f32(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  %0 = enzymexla.math.hypot %arg0, %arg1 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

func.func @apply_hypot_f64(%arg0: tensor<4xf64>, %arg1: tensor<4xf64>) -> tensor<4xf64> {
  %0 = enzymexla.math.hypot %arg0, %arg1 : (tensor<4xf64>, tensor<4xf64>) -> tensor<4xf64>
  return %0 : tensor<4xf64>
}

func.func @main() {
  // f32 Tests
  // Elements:
  // [0]: Standard case: hypot(3.0, 4.0) = 5.0
  // [1]: Zero edge case: hypot(0.0, 0.0) = 0.0
  // [2]: Overflow boundary case: hypot(large, large) where large = floatmax/2 = 1.70141173e38.
  //      Expected: large * sqrt(2) = 2.40615945e38. Naive x^2 + y^2 would overflow to infinity.
  // [3]: Underflow boundary case: hypot(small, small) where small = 1.0e-30.
  //      Expected: small * sqrt(2) = 1.41421356e-30. Naive x^2 + y^2 would underflow to zero.
  %x_f32 = stablehlo.constant dense<[3.0, 0.0, 1.70141173e38, 1.0e-30]> : tensor<4xf32>
  %y_f32 = stablehlo.constant dense<[4.0, 0.0, 1.70141173e38, 1.0e-30]> : tensor<4xf32>
  %expected_f32 = stablehlo.constant dense<[5.0, 0.0, 2.40615945e38, 1.41421356e-30]> : tensor<4xf32>

  %res_f32 = func.call @apply_hypot_f32(%x_f32, %y_f32) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  check.expect_almost_eq %res_f32, %expected_f32 : tensor<4xf32>

  // f64 Tests
  // Elements:
  // [0]: Standard case: hypot(3.0, 4.0) = 5.0
  // [1]: Zero edge case: hypot(0.0, 0.0) = 0.0
  // [2]: Overflow boundary case: hypot(large, large) where large = floatmax/2 = 8.988465674311579e307.
  //      Expected: large * sqrt(2) = 1.2711610061536462e308. Naive x^2 + y^2 would overflow to infinity.
  // [3]: Underflow boundary case: hypot(small, small) where small = 1.0e-300.
  //      Expected: small * sqrt(2) = 1.414213562373095e-300. Naive x^2 + y^2 would underflow to zero.
  %x_f64 = stablehlo.constant dense<[3.0, 0.0, 8.988465674311579e307, 1.0e-300]> : tensor<4xf64>
  %y_f64 = stablehlo.constant dense<[4.0, 0.0, 8.988465674311579e307, 1.0e-300]> : tensor<4xf64>
  %expected_f64 = stablehlo.constant dense<[5.0, 0.0, 1.2711610061536462e308, 1.414213562373095e-300]> : tensor<4xf64>

  %res_f64 = func.call @apply_hypot_f64(%x_f64, %y_f64) : (tensor<4xf64>, tensor<4xf64>) -> tensor<4xf64>
  check.expect_almost_eq %res_f64, %expected_f64 : tensor<4xf64>

  return
}

// LOWER-LABEL: func.func @apply_hypot_f32(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
// LOWER-DAG:    %[[ZERO:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<4xf32>
// LOWER-DAG:    %[[ONE:.*]] = stablehlo.constant dense<1.000000e+00> : tensor<4xf32>
// LOWER:        %[[ABS0:.*]] = stablehlo.abs %arg0 : tensor<4xf32>
// LOWER:        %[[ABS1:.*]] = stablehlo.abs %arg1 : tensor<4xf32>
// LOWER:        %[[MAX:.*]] = stablehlo.maximum %[[ABS0]], %[[ABS1]] : tensor<4xf32>
// LOWER:        %[[COMP:.*]] = stablehlo.compare EQ, %[[MAX]], %[[ZERO]], FLOAT : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
// LOWER:        %[[ABS2:.*]] = stablehlo.abs %arg0 : tensor<4xf32>
// LOWER:        %[[ABS3:.*]] = stablehlo.abs %arg1 : tensor<4xf32>
// LOWER:        %[[MAX2:.*]] = stablehlo.maximum %[[ABS2]], %[[ABS3]] : tensor<4xf32>
// LOWER:        %[[ABS4:.*]] = stablehlo.abs %arg0 : tensor<4xf32>
// LOWER:        %[[ABS5:.*]] = stablehlo.abs %arg1 : tensor<4xf32>
// LOWER:        %[[MIN:.*]] = stablehlo.minimum %[[ABS4]], %[[ABS5]] : tensor<4xf32>
// LOWER:        %[[ABS6:.*]] = stablehlo.abs %arg0 : tensor<4xf32>
// LOWER:        %[[ABS7:.*]] = stablehlo.abs %arg1 : tensor<4xf32>
// LOWER:        %[[MAX3:.*]] = stablehlo.maximum %[[ABS6]], %[[ABS7]] : tensor<4xf32>
// LOWER:        %[[DIV:.*]] = stablehlo.divide %[[MIN]], %[[MAX3]] : tensor<4xf32>
// LOWER:        %[[DIV2:.*]] = stablehlo.divide %{{.*}}, %{{.*}} : tensor<4xf32>
// LOWER:        %[[SQ:.*]] = stablehlo.multiply %[[DIV]], %[[DIV2]] : tensor<4xf32>
// LOWER:        %[[ADD:.*]] = stablehlo.add %[[ONE]], %[[SQ]] : tensor<4xf32>
// LOWER:        %[[SQRT:.*]] = stablehlo.sqrt %[[ADD]] : tensor<4xf32>
// LOWER:        %[[PROD:.*]] = stablehlo.multiply %[[MAX2]], %[[SQRT]] : tensor<4xf32>
// LOWER:        %[[RES:.*]] = stablehlo.select %[[COMP]], %[[ZERO]], %[[PROD]] : tensor<4xi1>, tensor<4xf32>
// LOWER:        return %[[RES]] : tensor<4xf32>
// LOWER:      }

// LOWER-LABEL: func.func @apply_hypot_f64(%arg0: tensor<4xf64>, %arg1: tensor<4xf64>) -> tensor<4xf64> {
// LOWER-DAG:    %[[ZERO_64:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<4xf64>
// LOWER-DAG:    %[[ONE_64:.*]] = stablehlo.constant dense<1.000000e+00> : tensor<4xf64>
// LOWER:        %[[ABS0_64:.*]] = stablehlo.abs %arg0 : tensor<4xf64>
// LOWER:        %[[ABS1_64:.*]] = stablehlo.abs %arg1 : tensor<4xf64>
// LOWER:        %[[MAX_64:.*]] = stablehlo.maximum %[[ABS0_64]], %[[ABS1_64]] : tensor<4xf64>
// LOWER:        %[[COMP_64:.*]] = stablehlo.compare EQ, %[[MAX_64]], %[[ZERO_64]], FLOAT : (tensor<4xf64>, tensor<4xf64>) -> tensor<4xi1>
// LOWER:        %[[ABS2_64:.*]] = stablehlo.abs %arg0 : tensor<4xf64>
// LOWER:        %[[ABS3_64:.*]] = stablehlo.abs %arg1 : tensor<4xf64>
// LOWER:        %[[MAX2_64:.*]] = stablehlo.maximum %[[ABS2_64]], %[[ABS3_64]] : tensor<4xf64>
// LOWER:        %[[ABS4_64:.*]] = stablehlo.abs %arg0 : tensor<4xf64>
// LOWER:        %[[ABS5_64:.*]] = stablehlo.abs %arg1 : tensor<4xf64>
// LOWER:        %[[MIN_64:.*]] = stablehlo.minimum %[[ABS4_64]], %[[ABS5_64]] : tensor<4xf64>
// LOWER:        %[[ABS6_64:.*]] = stablehlo.abs %arg0 : tensor<4xf64>
// LOWER:        %[[ABS7_64:.*]] = stablehlo.abs %arg1 : tensor<4xf64>
// LOWER:        %[[MAX3_64:.*]] = stablehlo.maximum %[[ABS6_64]], %[[ABS7_64]] : tensor<4xf64>
// LOWER:        %[[DIV_64:.*]] = stablehlo.divide %[[MIN_64]], %[[MAX3_64]] : tensor<4xf64>
// LOWER:        %[[DIV2_64:.*]] = stablehlo.divide %{{.*}}, %{{.*}} : tensor<4xf64>
// LOWER:        %[[SQ_64:.*]] = stablehlo.multiply %[[DIV_64]], %[[DIV2_64]] : tensor<4xf64>
// LOWER:        %[[ADD_64:.*]] = stablehlo.add %[[ONE_64]], %[[SQ_64]] : tensor<4xf64>
// LOWER:        %[[SQRT_64:.*]] = stablehlo.sqrt %[[ADD_64]] : tensor<4xf64>
// LOWER:        %[[PROD_64:.*]] = stablehlo.multiply %[[MAX2_64]], %[[SQRT_64]] : tensor<4xf64>
// LOWER:        %[[RES_64:.*]] = stablehlo.select %[[COMP_64]], %[[ZERO_64]], %[[PROD_64]] : tensor<4xi1>, tensor<4xf64>
// LOWER:        return %[[RES_64]] : tensor<4xf64>
// LOWER:      }
