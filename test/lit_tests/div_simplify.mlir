// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt | FileCheck %s

func.func @main1(%arg0: tensor<10x4xf32>) -> tensor<10x4xf32> {
    %cst = stablehlo.constant dense<2.000000e+00> : tensor<10x4xf32>
    %0 = stablehlo.divide %arg0, %cst : tensor<10x4xf32>
    return %0 : tensor<10x4xf32>
}

// CHECK: func.func @main1(%arg0: tensor<10x4xf32>) -> tensor<10x4xf32> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<5.000000e-01> : tensor<10x4xf32>
// CHECK-NEXT:     %0 = stablehlo.multiply %arg0, %cst : tensor<10x4xf32>
// CHECK-NEXT:     return %0 : tensor<10x4xf32>
// CHECK-NEXT: }

// shouldn't apply to integer division
func.func @main2(%arg0: tensor<10x4xi32>) -> tensor<10x4xi32> {
    %cst = stablehlo.constant dense<2> : tensor<10x4xi32>
    %0 = stablehlo.divide %arg0, %cst : tensor<10x4xi32>
    return %0 : tensor<10x4xi32>
}

// CHECK: func.func @main2(%arg0: tensor<10x4xi32>) -> tensor<10x4xi32> {
// CHECK-NEXT:     %c = stablehlo.constant dense<2> : tensor<10x4xi32>
// CHECK-NEXT:     %0 = stablehlo.divide %arg0, %c : tensor<10x4xi32>
// CHECK-NEXT:     return %0 : tensor<10x4xi32>
// CHECK-NEXT: }
