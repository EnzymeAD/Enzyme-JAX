// RUN: enzymexlamlir-opt --pad-for-alignment --allow-unregistered-dialect %s | FileCheck %s

func.func @test_bcast(%arg0: tensor<4x760x1533xf32>) {
  %0 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1, 2] : (tensor<4x760x1533xf32>) -> tensor<4x760x1533x121xf32>
  return
}

// CHECK: func.func @test_bcast(%arg0: tensor<4x760x1533xf32>) {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:     %0 = stablehlo.pad %arg0, %cst, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<4x768x1536xf32>
// CHECK-NEXT:     %1 = stablehlo.broadcast_in_dim %0, dims = [0, 1, 2] : (tensor<4x768x1536xf32>) -> tensor<4x768x1536x128xf32>
// CHECK-NEXT:     %2 = stablehlo.slice %1 [0:4, 0:760, 0:1536, 0:128] : (tensor<4x768x1536x128xf32>) -> tensor<4x760x1536x128xf32>
// CHECK-NEXT:     return
// CHECK-NEXT: }

func.func @test_bcast_from_scalar(%arg0: tensor<1xf32>) -> tensor<1520x3056xf32> {
  %0 = stablehlo.broadcast_in_dim %arg0, dims = [0] : (tensor<1xf32>) -> tensor<1520x3056xf32>
  return %0 : tensor<1520x3056xf32>
}

// CHECK-NEXT: func.func @test_bcast_from_scalar(%arg0: tensor<1xf32>) -> tensor<1520x3056xf32> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:     %0 = stablehlo.pad %arg0, %cst, low = [0], high = [0], interior = [0] : (tensor<1xf32>, tensor<f32>) -> tensor<1xf32>
// CHECK-NEXT:     %1 = stablehlo.broadcast_in_dim %0, dims = [0] : (tensor<1xf32>) -> tensor<1536x3072xf32>
// CHECK-NEXT:     %2 = stablehlo.slice %1 [0:1520, 0:3056] : (tensor<1536x3072xf32>) -> tensor<1520x3056xf32>
// CHECK-NEXT:     return %2 : tensor<1520x3056xf32>
// CHECK-NEXT: }
