// RUN: enzymexlamlir-opt %s --lower-enzymexla-ml | FileCheck %s

func.func @apply_gelu(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    %0 = enzymexla.ml.gelu %arg0, approximation = TANH : (tensor<4xf32>) -> tensor<4xf32>
    return %0 : tensor<4xf32>
}

// CHECK: func.func @apply_gelu(%arg0: tensor<4xf32>) -> tensor<4xf32> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<5.000000e-01> : tensor<4xf32>
// CHECK-NEXT:     %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<4xf32>
// CHECK-NEXT:     %cst_1 = stablehlo.constant dense<0.797884583> : tensor<4xf32>
// CHECK-NEXT:     %cst_2 = stablehlo.constant dense<4.471500e-02> : tensor<4xf32>
// CHECK-NEXT:     %0 = stablehlo.multiply %arg0, %arg0 : tensor<4xf32>
// CHECK-NEXT:     %1 = stablehlo.multiply %arg0, %0 : tensor<4xf32>
// CHECK-NEXT:     %2 = stablehlo.multiply %cst_2, %1 : tensor<4xf32>
// CHECK-NEXT:     %3 = stablehlo.add %arg0, %2 : tensor<4xf32>
// CHECK-NEXT:     %4 = stablehlo.multiply %cst_1, %3 : tensor<4xf32>
// CHECK-NEXT:     %5 = stablehlo.tanh %4 : tensor<4xf32>
// CHECK-NEXT:     %6 = stablehlo.add %cst_0, %5 : tensor<4xf32>
// CHECK-NEXT:     %7 = stablehlo.multiply %cst, %6 : tensor<4xf32>
// CHECK-NEXT:     %8 = stablehlo.multiply %arg0, %7 : tensor<4xf32>
// CHECK-NEXT:     return %8 : tensor<4xf32>
// CHECK-NEXT: }

func.func @apply_gelu2(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    %0 = enzymexla.ml.gelu %arg0, approximation = SIGMOID : (tensor<4xf32>) -> tensor<4xf32>
    return %0 : tensor<4xf32>
}

// CHECK: func.func @apply_gelu2(%arg0: tensor<4xf32>) -> tensor<4xf32> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<1.59576917> : tensor<4xf32>
// CHECK-NEXT:     %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<4xf32>
// CHECK-NEXT:     %cst_1 = stablehlo.constant dense<4.471500e-02> : tensor<4xf32>
// CHECK-NEXT:     %0 = stablehlo.multiply %arg0, %arg0 : tensor<4xf32>
// CHECK-NEXT:     %1 = stablehlo.multiply %cst_1, %0 : tensor<4xf32>
// CHECK-NEXT:     %2 = stablehlo.add %cst_0, %1 : tensor<4xf32>
// CHECK-NEXT:     %3 = stablehlo.multiply %arg0, %2 : tensor<4xf32>
// CHECK-NEXT:     %4 = stablehlo.multiply %cst, %3 : tensor<4xf32>
// CHECK-NEXT:     %5 = stablehlo.logistic %4 : tensor<4xf32>
// CHECK-NEXT:     %6 = stablehlo.multiply %arg0, %5 : tensor<4xf32>
// CHECK-NEXT:     return %6 : tensor<4xf32>
// CHECK-NEXT: }

func.func @apply_gelu3(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    %0 = enzymexla.ml.gelu %arg0, approximation = NONE : (tensor<4xf32>) -> tensor<4xf32>
    return %0 : tensor<4xf32>
}

// CHECK: func.func @apply_gelu3(%arg0: tensor<4xf32>) -> tensor<4xf32> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<5.000000e-01> : tensor<4xf32>
// CHECK-NEXT:     %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<4xf32>
// CHECK-NEXT:     %cst_1 = stablehlo.constant dense<0.707106769> : tensor<4xf32>
// CHECK-NEXT:     %0 = stablehlo.multiply %arg0, %cst_1 : tensor<4xf32>
// CHECK-NEXT:     %1 = chlo.erf %0 : tensor<4xf32> -> tensor<4xf32>
// CHECK-NEXT:     %2 = stablehlo.add %cst_0, %1 : tensor<4xf32>
// CHECK-NEXT:     %3 = stablehlo.multiply %cst, %2 : tensor<4xf32>
// CHECK-NEXT:     %4 = stablehlo.multiply %arg0, %3 : tensor<4xf32>
// CHECK-NEXT:     return %4 : tensor<4xf32>
// CHECK-NEXT: }
