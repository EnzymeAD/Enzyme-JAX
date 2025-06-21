// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt | FileCheck %s

func.func @main1(%arg0: tensor<f64>) -> tensor<f64> {
    %0 = stablehlo.exponential %arg0 : tensor<f64>
    %1 = stablehlo.log %0 : tensor<f64>
    return %1 : tensor<f64>
}

// CHECK: func.func @main1(%arg0: tensor<f64>) -> tensor<f64> {
// CHECK-NEXT:     return %arg0 : tensor<f64>
// CHECK-NEXT: }

func.func @main2(%arg0: tensor<f64>) -> tensor<f64> {
    %0 = stablehlo.multiply %arg0, %arg0 : tensor<f64>
    %1 = stablehlo.multiply %0, %arg0 : tensor<f64>
    %2 = stablehlo.multiply %1, %arg0 : tensor<f64>
    %3 = stablehlo.multiply %2, %arg0 : tensor<f64>
    %4 = stablehlo.log %3 : tensor<f64>
    return %4 : tensor<f64>
}

// CHECK: func.func @main2(%arg0: tensor<f64>) -> tensor<f64> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<5.000000e+00> : tensor<f64>
// CHECK-NEXT:    %0 = stablehlo.log %arg0 : tensor<f64>
// CHECK-NEXT:    %1 = stablehlo.multiply %cst, %0 : tensor<f64>
// CHECK-NEXT:    return %1 : tensor<f64>
// CHECK-NEXT:  }
