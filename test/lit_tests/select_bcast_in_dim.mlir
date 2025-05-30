// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

func.func @main1(%arg0: tensor<1xi1>, %arg1: tensor<2x2xi32>) -> tensor<2x2xi32> {
    %cst = stablehlo.constant dense<1> : tensor<2x2xi32>
    %0 = stablehlo.add %arg1, %cst : tensor<2x2xi32>
    %2 = stablehlo.broadcast_in_dim %arg0, dims = [0] : (tensor<1xi1>) -> tensor<2x2xi1>
    %3 = stablehlo.select %2, %0, %arg1 : tensor<2x2xi1>, tensor<2x2xi32>
    return %3 : tensor<2x2xi32>
}

// CHECK: func.func @main1(%arg0: tensor<1xi1>, %arg1: tensor<2x2xi32>) -> tensor<2x2xi32> {
// CHECK-NEXT:     %c = stablehlo.constant dense<1> : tensor<2x2xi32>
// CHECK-NEXT:     %0 = stablehlo.add %arg1, %c : tensor<2x2xi32>
// CHECK-NEXT:     %1 = stablehlo.reshape %arg0 : (tensor<1xi1>) -> tensor<i1>
// CHECK-NEXT:     %2 = stablehlo.select %1, %0, %arg1 : tensor<i1>, tensor<2x2xi32>
// CHECK-NEXT:     return %2 : tensor<2x2xi32>
// CHECK-NEXT: }

func.func @main2(%arg0: tensor<i1>, %arg1: tensor<2x2xi32>) -> tensor<2x2xi32> {
    %cst = stablehlo.constant dense<1> : tensor<2x2xi32>
    %0 = stablehlo.add %arg1, %cst : tensor<2x2xi32>
    %2 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<i1>) -> tensor<2x2xi1>
    %3 = stablehlo.select %2, %0, %arg1 : tensor<2x2xi1>, tensor<2x2xi32>
    return %3 : tensor<2x2xi32>
}

// CHECK: func.func @main2(%arg0: tensor<i1>, %arg1: tensor<2x2xi32>) -> tensor<2x2xi32> {
// CHECK-NEXT:     %c = stablehlo.constant dense<1> : tensor<2x2xi32>
// CHECK-NEXT:     %0 = stablehlo.add %arg1, %c : tensor<2x2xi32>
// CHECK-NEXT:     %1 = stablehlo.select %arg0, %0, %arg1 : tensor<i1>, tensor<2x2xi32>
// CHECK-NEXT:     return %1 : tensor<2x2xi32>
// CHECK-NEXT: }
