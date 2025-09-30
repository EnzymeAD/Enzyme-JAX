// RUN: enzymexlamlir-opt --auto-batching --inline --enzyme-hlo-opt %s | FileCheck %s

func.func @elem(%arg0: tensor<2xi64>) -> tensor<2xi64> {
    %c = stablehlo.constant dense<1> : tensor<i64>
    %c_0 = stablehlo.constant dense<3> : tensor<i64>
    %0 = stablehlo.slice %arg0 [0:1] : (tensor<2xi64>) -> tensor<1xi64>
    %1 = stablehlo.slice %arg0 [1:2] : (tensor<2xi64>) -> tensor<1xi64>
    %2 = stablehlo.reshape %0 : (tensor<1xi64>) -> tensor<i64>
    %3 = stablehlo.reshape %1 : (tensor<1xi64>) -> tensor<i64>
    %6 = stablehlo.add %2, %c : tensor<i64>
    %7 = stablehlo.add %3, %c : tensor<i64>
    %10 = stablehlo.reshape %6 : (tensor<i64>) -> tensor<1xi64>
    %11 = stablehlo.reshape %7 : (tensor<i64>) -> tensor<1xi64>
    %12 = stablehlo.concatenate %10, %11, dim = 0 : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
    return %12 : tensor<2xi64>
}


// CHECK: func.func @elem(%arg0: tensor<2xi64>) -> tensor<2xi64> {
// CHECK-NEXT:     %c = stablehlo.constant dense<1> : tensor<2xi64>
// CHECK-NEXT:     %0 = stablehlo.add %arg0, %c : tensor<2xi64>
// CHECK-NEXT:     return %0 : tensor<2xi64>
// CHECK-NEXT: }
