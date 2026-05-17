// RUN: enzymexlamlir-opt %s --lower-enzymexla-math | FileCheck %s --check-prefix=LOWER
// RUN: enzymexlamlir-opt %s --lower-enzymexla-math --chlo-legalize-to-stablehlo | stablehlo-translate - --interpret

func.func @apply_cosc(%arg0: tensor<6xf32>) -> tensor<6xf32> {
  %0 = enzymexla.math.cosc %arg0 : (tensor<6xf32>) -> tensor<6xf32>
  return %0 : tensor<6xf32>
}

func.func @main() {
  %x_cosc = stablehlo.constant dense<[0.0, 0.5, 1.0, 1.5, 2.0, -0.5]> : tensor<6xf32>
  %expected_cosc = stablehlo.constant dense<[0.0, -1.27323954, -1.0, 0.14147106, 0.5, 1.27323954]> : tensor<6xf32>

  %res_cosc = func.call @apply_cosc(%x_cosc) : (tensor<6xf32>) -> tensor<6xf32>
  check.expect_almost_eq %res_cosc, %expected_cosc : tensor<6xf32>
  return
}

// LOWER-LABEL: func.func @apply_cosc(%arg0: tensor<6xf32>) -> tensor<6xf32> {
// LOWER-DAG:    %[[PI:.*]] = stablehlo.constant dense<3.14159274> : tensor<6xf32>
// LOWER-DAG:    %[[ZERO:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<6xf32>
// LOWER:        %[[IS_ZERO:.*]] = stablehlo.compare EQ, %arg0, %[[ZERO]], FLOAT : (tensor<6xf32>, tensor<6xf32>) -> tensor<6xi1>
// LOWER:        %[[MUL:.*]] = stablehlo.multiply %[[PI]], %arg0 : tensor<6xf32>
// LOWER:        %[[COS:.*]] = stablehlo.cosine %[[MUL]] : tensor<6xf32>
// LOWER:        %[[DIV_COS:.*]] = stablehlo.divide %[[COS]], %arg0 : tensor<6xf32>
// LOWER:        %[[MUL2:.*]] = stablehlo.multiply %[[PI]], %arg0 : tensor<6xf32>
// LOWER:        %[[SIN:.*]] = stablehlo.sine %[[MUL2]] : tensor<6xf32>
// LOWER:        %[[MUL3:.*]] = stablehlo.multiply %[[PI]], %arg0 : tensor<6xf32>
// LOWER:        %[[DEN_SIN:.*]] = stablehlo.multiply %[[MUL3]], %arg0 : tensor<6xf32>
// LOWER:        %[[DIV_SIN:.*]] = stablehlo.divide %[[SIN]], %[[DEN_SIN]] : tensor<6xf32>
// LOWER:        %[[SUB:.*]] = stablehlo.subtract %[[DIV_COS]], %[[DIV_SIN]] : tensor<6xf32>
// LOWER:        %[[RES:.*]] = stablehlo.select %[[IS_ZERO]], %[[ZERO]], %[[SUB]] : tensor<6xi1>, tensor<6xf32>
// LOWER:        return %[[RES]] : tensor<6xf32>
// LOWER:      }
