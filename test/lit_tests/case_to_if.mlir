// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

func.func @main(%arg0: tensor<4x3xf32>, %arg1: tensor<4x3xf32>, %arg2: tensor<f32>) -> (tensor<4x3xf32>) {
    %0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.compare LT, %arg2, %0 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %2 = stablehlo.convert %1 : (tensor<i1>) -> tensor<i32>
    %3 = "stablehlo.case"(%2) ({
        %4 = stablehlo.constant dense<1.000000e+00> : tensor<4x3xf32>
        %5 = stablehlo.add %arg0, %4 : tensor<4x3xf32>
        stablehlo.return %5 : tensor<4x3xf32>
    }, {
        %6 = stablehlo.constant dense<-1.000000e+00> : tensor<4x3xf32>
        %7 = stablehlo.add %arg1, %6 : tensor<4x3xf32>
        stablehlo.return %7 : tensor<4x3xf32>
    }) : (tensor<i32>) -> tensor<4x3xf32>
    return %3 : tensor<4x3xf32>
}

// CHECK: func.func @main(%arg0: tensor<4x3xf32>, %arg1: tensor<4x3xf32>, %arg2: tensor<f32>) -> tensor<4x3xf32> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<-1.000000e+00> : tensor<4x3xf32>
// CHECK-NEXT:     %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<4x3xf32>
// CHECK-NEXT:     %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:     %0 = stablehlo.compare  LT, %arg2, %cst_1 : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK-NEXT:     %1 = "stablehlo.if"(%0) ({
// CHECK-NEXT:       %2 = stablehlo.add %arg1, %cst : tensor<4x3xf32>
// CHECK-NEXT:       stablehlo.return %2 : tensor<4x3xf32>
// CHECK-NEXT:     }, {
// CHECK-NEXT:       %2 = stablehlo.add %arg0, %cst_0 : tensor<4x3xf32>
// CHECK-NEXT:       stablehlo.return %2 : tensor<4x3xf32>
// CHECK-NEXT:     }) : (tensor<i1>) -> tensor<4x3xf32>
// CHECK-NEXT:     return %1 : tensor<4x3xf32>
// CHECK-NEXT: }
