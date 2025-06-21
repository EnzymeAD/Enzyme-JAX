// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt | FileCheck %s

func.func @main(%arg0: tensor<10x4xf32>) -> tensor<10x4xf32> {
    %cst = stablehlo.constant dense<2.000000e+00> : tensor<10x4xf32>
    %0 = stablehlo.multiply %arg0, %cst : tensor<10x4xf32>
    %1 = stablehlo.negate %0 : tensor<10x4xf32>
    return %1 : tensor<10x4xf32>
}

// CHECK: func.func @main(%arg0: tensor<10x4xf32>) -> tensor<10x4xf32> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<-2.000000e+00> : tensor<10x4xf32>
// CHECK-NEXT:     %0 = stablehlo.multiply %arg0, %cst : tensor<10x4xf32>
// CHECK-NEXT:     return %0 : tensor<10x4xf32>
// CHECK-NEXT:   }

func.func @main2(%arg0: tensor<10x4xf32>) -> tensor<10x4xf32> {
    %cst = stablehlo.constant dense<2.000000e+00> : tensor<10x4xf32>
    %0 = stablehlo.multiply %cst, %arg0 : tensor<10x4xf32>
    %1 = stablehlo.negate %0 : tensor<10x4xf32>
    return %1 : tensor<10x4xf32>
}

// CHECK: func.func @main2(%arg0: tensor<10x4xf32>) -> tensor<10x4xf32> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<-2.000000e+00> : tensor<10x4xf32>
// CHECK-NEXT:     %0 = stablehlo.multiply %cst, %arg0 : tensor<10x4xf32>
// CHECK-NEXT:     return %0 : tensor<10x4xf32>
// CHECK-NEXT: }
