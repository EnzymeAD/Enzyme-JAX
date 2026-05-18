// RUN: enzymexlamlir-opt %s --lower-enzymexla-math | FileCheck %s

func.func @apply_gelu(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    %0 = enzymexla.math.gelu %arg0, approximation = TANH : (tensor<4xf32>) -> tensor<4xf32>
    return %0 : tensor<4xf32>
}

// CHECK: func.func @apply_gelu(%arg0: tensor<4xf32>) -> tensor<4xf32> {
// CHECK-DAG:     %[[CST_HALF:.*]] = stablehlo.constant dense<5.000000e-01> : tensor<4xf32>
// CHECK-DAG:     %[[CST_ONE:.*]] = stablehlo.constant dense<1.000000e+00> : tensor<4xf32>
// CHECK-DAG:     %[[CST_SQRT2PI:.*]] = stablehlo.constant dense<0.797884583> : tensor<4xf32>
// CHECK-DAG:     %[[CST_COEFF:.*]] = stablehlo.constant dense<4.471500e-02> : tensor<4xf32>
// CHECK-DAG:     %[[X2:.*]] = stablehlo.multiply %arg0, %arg0 : tensor<4xf32>
// CHECK-DAG:     %[[X3:.*]] = stablehlo.multiply %arg0, %[[X2]] : tensor<4xf32>
// CHECK-DAG:     %[[POLY:.*]] = stablehlo.multiply %[[CST_COEFF]], %[[X3]] : tensor<4xf32>
// CHECK-DAG:     %[[XPP:.*]] = stablehlo.add %arg0, %[[POLY]] : tensor<4xf32>
// CHECK-DAG:     %[[SCALED:.*]] = stablehlo.multiply %[[CST_SQRT2PI]], %[[XPP]] : tensor<4xf32>
// CHECK-DAG:     %[[TANH:.*]] = stablehlo.tanh %[[SCALED]] : tensor<4xf32>
// CHECK-DAG:     %[[OPT:.*]] = stablehlo.add %[[CST_ONE]], %[[TANH]] : tensor<4xf32>
// CHECK-DAG:     %[[HOPT:.*]] = stablehlo.multiply %[[CST_HALF]], %[[OPT]] : tensor<4xf32>
// CHECK-NEXT:    %[[RES:.*]] = stablehlo.multiply %arg0, %[[HOPT]] : tensor<4xf32>
// CHECK-NEXT:    return %[[RES]] : tensor<4xf32>
// CHECK-NEXT: }

func.func @apply_gelu2(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    %0 = enzymexla.math.gelu %arg0, approximation = SIGMOID : (tensor<4xf32>) -> tensor<4xf32>
    return %0 : tensor<4xf32>
}

// CHECK: func.func @apply_gelu2(%arg0: tensor<4xf32>) -> tensor<4xf32> {
// CHECK-DAG:     %[[CST_SQRT8PI:.*]] = stablehlo.constant dense<1.59576917> : tensor<4xf32>
// CHECK-DAG:     %[[CST_ONE:.*]] = stablehlo.constant dense<1.000000e+00> : tensor<4xf32>
// CHECK-DAG:     %[[CST_COEFF:.*]] = stablehlo.constant dense<4.471500e-02> : tensor<4xf32>
// CHECK-DAG:     %[[X2:.*]] = stablehlo.multiply %arg0, %arg0 : tensor<4xf32>
// CHECK-DAG:     %[[POLY:.*]] = stablehlo.multiply %[[CST_COEFF]], %[[X2]] : tensor<4xf32>
// CHECK-DAG:     %[[XPP:.*]] = stablehlo.add %[[CST_ONE]], %[[POLY]] : tensor<4xf32>
// CHECK-DAG:     %[[INNER:.*]] = stablehlo.multiply %arg0, %[[XPP]] : tensor<4xf32>
// CHECK-DAG:     %[[SCALED:.*]] = stablehlo.multiply %[[CST_SQRT8PI]], %[[INNER]] : tensor<4xf32>
// CHECK-DAG:     %[[SIGMOID:.*]] = stablehlo.logistic %[[SCALED]] : tensor<4xf32>
// CHECK-NEXT:    %[[RES:.*]] = stablehlo.multiply %arg0, %[[SIGMOID]] : tensor<4xf32>
// CHECK-NEXT:    return %[[RES]] : tensor<4xf32>
// CHECK-NEXT: }

func.func @apply_gelu3(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    %0 = enzymexla.math.gelu %arg0, approximation = NONE : (tensor<4xf32>) -> tensor<4xf32>
    return %0 : tensor<4xf32>
}

// CHECK: func.func @apply_gelu3(%arg0: tensor<4xf32>) -> tensor<4xf32> {
// CHECK-DAG:     %[[CST_HALF:.*]] = stablehlo.constant dense<5.000000e-01> : tensor<4xf32>
// CHECK-DAG:     %[[CST_ONE:.*]] = stablehlo.constant dense<1.000000e+00> : tensor<4xf32>
// CHECK-DAG:     %[[CST_INV_SQRT2:.*]] = stablehlo.constant dense<0.707106769> : tensor<4xf32>
// CHECK-DAG:     %[[X_OVER_SQRT2:.*]] = stablehlo.multiply %arg0, %[[CST_INV_SQRT2]] : tensor<4xf32>
// CHECK-DAG:     %[[ERF:.*]] = chlo.erf %[[X_OVER_SQRT2]] : tensor<4xf32> -> tensor<4xf32>
// CHECK-DAG:     %[[ONE_PLUS_ERF:.*]] = stablehlo.add %[[CST_ONE]], %[[ERF]] : tensor<4xf32>
// CHECK-DAG:     %[[H_ONE_PLUS_ERF:.*]] = stablehlo.multiply %[[CST_HALF]], %[[ONE_PLUS_ERF]] : tensor<4xf32>
// CHECK-NEXT:    %[[RES:.*]] = stablehlo.multiply %arg0, %[[H_ONE_PLUS_ERF]] : tensor<4xf32>
// CHECK-NEXT:    return %[[RES]] : tensor<4xf32>
// CHECK-NEXT: }
