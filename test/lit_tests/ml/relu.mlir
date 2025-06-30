// RUN: enzymexlamlir-opt --lower-enzymexla-ml %s | FileCheck %s

func.func @main(%arg0: tensor<4x5x6x7xf32>) -> tensor<4x5x6x7xf32> {
    %0 = enzymexla.ml.relu %arg0 : (tensor<4x5x6x7xf32>) -> tensor<4x5x6x7xf32>
    return %0 : tensor<4x5x6x7xf32>
}

// CHECK: func.func @main(%arg0: tensor<4x5x6x7xf32>) -> tensor<4x5x6x7xf32> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<4x5x6x7xf32>
// CHECK-NEXT:     %0 = stablehlo.maximum %arg0, %cst : tensor<4x5x6x7xf32>
// CHECK-NEXT:     return %0 : tensor<4x5x6x7xf32>
// CHECK-NEXT: }
