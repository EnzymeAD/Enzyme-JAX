// RUN: enzymexlamlir-opt %s --lower-enzymexla-math | FileCheck %s --check-prefix=LOWER
// RUN: enzymexlamlir-opt %s --lower-enzymexla-math --chlo-legalize-to-stablehlo | stablehlo-translate - --interpret

func.func @apply_sinc(%arg0: tensor<6xf32>) -> tensor<6xf32> {
  %0 = enzymexla.math.sinc %arg0 : (tensor<6xf32>) -> tensor<6xf32>
  return %0 : tensor<6xf32>
}

func.func @main() {
  %x_sinc = stablehlo.constant dense<[0.0, 0.5, 1.0, 1.5, 2.0, -0.5]> : tensor<6xf32>
  %expected_sinc = stablehlo.constant dense<[1.0, 0.636619772, 0.0, -0.21220659, 0.0, 0.636619772]> : tensor<6xf32>

  %res_sinc = func.call @apply_sinc(%x_sinc) : (tensor<6xf32>) -> tensor<6xf32>
  check.expect_almost_eq %res_sinc, %expected_sinc : tensor<6xf32>
  return
}

// LOWER-LABEL: func.func @apply_sinc(%arg0: tensor<6xf32>) -> tensor<6xf32> {
// LOWER-DAG:    %[[PI:.*]] = stablehlo.constant dense<3.14159274> : tensor<6xf32>
// LOWER-DAG:    %[[ONE:.*]] = stablehlo.constant dense<1.000000e+00> : tensor<6xf32>
// LOWER-DAG:    %[[ZERO:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<6xf32>
// LOWER:        %[[IS_ZERO:.*]] = stablehlo.compare EQ, %arg0, %[[ZERO]], FLOAT : (tensor<6xf32>, tensor<6xf32>) -> tensor<6xi1>
// LOWER:        %[[MUL:.*]] = stablehlo.multiply %[[PI]], %arg0 : tensor<6xf32>
// LOWER:        %[[SIN:.*]] = stablehlo.sine %[[MUL]] : tensor<6xf32>
// LOWER:        %[[MUL_DEN:.*]] = stablehlo.multiply %[[PI]], %arg0 : tensor<6xf32>
// LOWER:        %[[DIV:.*]] = stablehlo.divide %[[SIN]], %[[MUL_DEN]] : tensor<6xf32>
// LOWER:        %[[RES:.*]] = stablehlo.select %[[IS_ZERO]], %[[ONE]], %[[DIV]] : tensor<6xi1>, tensor<6xf32>
// LOWER:        return %[[RES]] : tensor<6xf32>
// LOWER:      }
