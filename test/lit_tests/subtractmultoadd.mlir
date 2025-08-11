// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

func.func @main1(%arg0: tensor<64xf32>) -> tensor<64xf32> {
    %cst = stablehlo.constant dense<3.000000e+00> : tensor<64xf32>
    %0 = stablehlo.multiply %arg0, %cst : tensor<64xf32>
    %1 = stablehlo.subtract %arg0, %0 : tensor<64xf32>
    return %1 : tensor<64xf32>
}

// CHECK: func.func @main1(%arg0: tensor<64xf32>) -> tensor<64xf32> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<-3.000000e+00> : tensor<64xf32>
// CHECK-NEXT:     %0 = stablehlo.multiply %arg0, %cst : tensor<64xf32>
// CHECK-NEXT:     %1 = stablehlo.add %arg0, %0 : tensor<64xf32>
// CHECK-NEXT:     return %1 : tensor<64xf32>
// CHECK-NEXT: }

func.func @main2(%arg0: tensor<64xf32>) -> tensor<64xf32> {
    %cst = stablehlo.constant dense<3.000000e+00> : tensor<64xf32>
    %0 = stablehlo.multiply %cst, %arg0 : tensor<64xf32>
    %1 = stablehlo.subtract %arg0, %0 : tensor<64xf32>
    return %1 : tensor<64xf32>
}

// CHECK: func.func @main2(%arg0: tensor<64xf32>) -> tensor<64xf32> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<-3.000000e+00> : tensor<64xf32>
// CHECK-NEXT:     %0 = stablehlo.multiply %cst, %arg0 : tensor<64xf32>
// CHECK-NEXT:     %1 = stablehlo.add %arg0, %0 : tensor<64xf32>
// CHECK-NEXT:     return %1 : tensor<64xf32>
// CHECK-NEXT: }
