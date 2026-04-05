// RUN: enzymexlamlir-opt --pad-for-alignment --allow-unregistered-dialect %s | FileCheck %s

func.func @test_transpose(%arg0: tensor<4x760x1533xf32>) {
  %0 = stablehlo.transpose %arg0, dims = [0, 1, 2] : (tensor<4x760x1533xf32>) -> tensor<4x760x1533xf32>
  return
}

// CHECK: func.func @test_transpose(%arg0: tensor<4x760x1533xf32>) {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:     %0 = stablehlo.pad %arg0, %cst, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<4x768x1536xf32>
// CHECK-NEXT:     %1 = stablehlo.transpose %0, dims = [0, 1, 2] : (tensor<4x768x1536xf32>) -> tensor<4x768x1536xf32>
// CHECK-NEXT:     return
// CHECK-NEXT: }

func.func @test_transpose_ret(%arg0: tensor<4x760x1533xf32>) -> tensor<4x760x1533xf32> {
  %0 = stablehlo.transpose %arg0, dims = [0, 1, 2] : (tensor<4x760x1533xf32>) -> tensor<4x760x1533xf32>
  return %0 : tensor<4x760x1533xf32>
}

// CHECK: func.func @test_transpose_ret(%arg0: tensor<4x760x1533xf32>) -> tensor<4x760x1533xf32> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:     %0 = stablehlo.pad %arg0, %cst, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<4x768x1536xf32>
// CHECK-NEXT:     %1 = stablehlo.transpose %0, dims = [0, 1, 2] : (tensor<4x768x1536xf32>) -> tensor<4x768x1536xf32>
// CHECK-NEXT:     %2 = stablehlo.slice %1 [0:4, 0:760, 0:1533] : (tensor<4x768x1536xf32>) -> tensor<4x760x1533xf32>
// CHECK-NEXT:     return %2 : tensor<4x760x1533xf32>
// CHECK-NEXT: }
