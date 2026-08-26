// RUN: enzymexlamlir-opt --convert-all-constants-to-splatted-constant %s | FileCheck %s

func.func @main() -> tensor<3x4xf32> {
    %c = stablehlo.constant dense<[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]> : tensor<2x4xf32>
    %c1 = stablehlo.constant dense<[[9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0]]> : tensor<2x4xf32>
    %0 = stablehlo.add %c, %c1 : tensor<2x4xf32>
    %1 = stablehlo.slice %0 [0:1, 0:4] : (tensor<2x4xf32>) -> tensor<1x4xf32>
    %c2 = stablehlo.constant dense<4.0> : tensor<3x4xf32>
    %2 = stablehlo.concatenate %0, %1, dim = 0 : (tensor<2x4xf32>, tensor<1x4xf32>) -> tensor<3x4xf32>
    %3 = stablehlo.add %2, %c2 : tensor<3x4xf32>
    return %3 : tensor<3x4xf32>
}

// CHECK: func.func @main() -> tensor<3x4xf32> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<4.000000e+00> : tensor<3x4xf32>
// CHECK-NEXT:     %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<2x4xf32>
// CHECK-NEXT:     %cst_1 = stablehlo.constant dense<9.000000e+00> : tensor<2x4xf32>
// CHECK-NEXT:     %0 = stablehlo.add %cst_0, %cst_1 : tensor<2x4xf32>
// CHECK-NEXT:     %1 = stablehlo.slice %0 [0:1, 0:4] : (tensor<2x4xf32>) -> tensor<1x4xf32>
// CHECK-NEXT:     %2 = stablehlo.concatenate %0, %1, dim = 0 : (tensor<2x4xf32>, tensor<1x4xf32>) -> tensor<3x4xf32>
// CHECK-NEXT:     %3 = stablehlo.add %2, %cst : tensor<3x4xf32>
// CHECK-NEXT:     return %3 : tensor<3x4xf32>
// CHECK-NEXT: }
