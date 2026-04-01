// RUN: enzymexlamlir-opt  --enzyme-hlo-generate-td="patterns=transpose_elementwise(0);transpose_simplify;transpose_transpose" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s

module {
  func.func @main(%arg0: tensor<4x3xf32>) -> tensor<4x3xf32> {
    %cst = stablehlo.constant dense<6.000000e+00> : tensor<3x4xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<3x4xf32>
    %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<4x3xf32>) -> tensor<3x4xf32>
    %1 = stablehlo.clamp %cst_0, %0, %cst : tensor<3x4xf32>
    %2 = stablehlo.transpose %1, dims = [1, 0] : (tensor<3x4xf32>) -> tensor<4x3xf32>
    return %2 : tensor<4x3xf32>
  }
}

// CHECK: func.func @main(%arg0: tensor<4x3xf32>) -> tensor<4x3xf32> {
// CHECK-NEXT:   %cst = stablehlo.constant dense<6.000000e+00> : tensor<4x3xf32>
// CHECK-NEXT:   %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<4x3xf32>
// CHECK-NEXT:   %0 = stablehlo.transpose %arg0, dims = [0, 1] : (tensor<4x3xf32>) -> tensor<4x3xf32>
// CHECK-NEXT:   %1 = stablehlo.clamp %cst_0, %0, %cst : tensor<4x3xf32>
// CHECK-NEXT:   return %1 : tensor<4x3xf32>
// CHECK-NEXT: }
