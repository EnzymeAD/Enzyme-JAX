// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt | FileCheck %s

func.func @main1(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xf32> {
    %cst = stablehlo.constant dense<-1.000000e+00> : tensor<2x2xf32>
    %0 = stablehlo.multiply %arg0, %cst : tensor<2x2xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<2x2xf32>
    return %1 : tensor<2x2xf32>
}

// CHECK: func.func @main1(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xf32> {
// CHECK-NEXT:     %0 = stablehlo.subtract %arg1, %arg0 : tensor<2x2xf32>
// CHECK-NEXT:     return %0 : tensor<2x2xf32>
// CHECK-NEXT:  }

func.func @main2(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xf32> {
    %cst = stablehlo.constant dense<-1.000000e+00> : tensor<2x2xf32>
    %0 = stablehlo.multiply %arg0, %cst : tensor<2x2xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<2x2xf32>
    return %1 : tensor<2x2xf32>
}

// CHECK: func.func @main2(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xf32> {
// CHECK-NEXT:    %0 = stablehlo.subtract %arg1, %arg0 : tensor<2x2xf32>
// CHECK-NEXT:    return %0 : tensor<2x2xf32>
// CHECK-NEXT:  }
