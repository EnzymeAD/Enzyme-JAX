// RUN: enzymexlamlir-opt --pad-for-alignment --allow-unregistered-dialect %s | FileCheck %s

func.func @test_slice_identity(%arg0: tensor<4x760x1533xf32>) {
  %0 = stablehlo.slice %arg0 [0:4, 0:760, 0:1533] : (tensor<4x760x1533xf32>) -> tensor<4x760x1533xf32>
  return
}

// CHECK: func.func @test_slice_identity(%arg0: tensor<4x760x1533xf32>) {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:     %0 = stablehlo.pad %arg0, %cst, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<4x768x1536xf32>
// CHECK-NEXT:     %1 = stablehlo.slice %0 [0:4, 0:768, 0:1536] : (tensor<4x768x1536xf32>) -> tensor<4x768x1536xf32>
// CHECK-NEXT:     return
// CHECK-NEXT: }

func.func @test_slice_identity_ret(%arg0: tensor<4x760x1533xf32>) -> tensor<4x760x1533xf32> {
  %0 = stablehlo.slice %arg0 [0:4, 0:760, 0:1533] : (tensor<4x760x1533xf32>) -> tensor<4x760x1533xf32>
  return %0 : tensor<4x760x1533xf32>
}

// CHECK: func.func @test_slice_identity_ret(%arg0: tensor<4x760x1533xf32>) -> tensor<4x760x1533xf32> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:     %0 = stablehlo.pad %arg0, %cst, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<4x768x1536xf32>
// CHECK-NEXT:     %1 = stablehlo.slice %0 [0:4, 0:768, 0:1536] : (tensor<4x768x1536xf32>) -> tensor<4x768x1536xf32>
// CHECK-NEXT:     %2 = stablehlo.slice %1 [0:4, 0:760, 0:1533] : (tensor<4x768x1536xf32>) -> tensor<4x760x1533xf32>
// CHECK-NEXT:     return %2 : tensor<4x760x1533xf32>
// CHECK-NEXT: }

func.func @test_slice_needs_pad_after(%arg0: tensor<4x760x1533xf32>) {
  %0 = stablehlo.slice %arg0 [0:4, 160:760, 0:1533] : (tensor<4x760x1533xf32>) -> tensor<4x600x1533xf32>
  return
}

// CHECK: func.func @test_slice_needs_pad_after(%arg0: tensor<4x760x1533xf32>) {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:     %0 = stablehlo.pad %arg0, %cst, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<4x768x1536xf32>
// CHECK-NEXT:     %1 = stablehlo.slice %0 [0:4, 160:768, 0:1536] : (tensor<4x768x1536xf32>) -> tensor<4x608x1536xf32>
// CHECK-NEXT:     %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:     %2 = stablehlo.pad %1, %cst_0, low = [0, 0, 0], high = [0, 32, 0], interior = [0, 0, 0] : (tensor<4x608x1536xf32>, tensor<f32>) -> tensor<4x640x1536xf32>
// CHECK-NEXT:     return
// CHECK-NEXT: }

func.func @test_slice_needs_pad_after_ret(%arg0: tensor<4x760x1533xf32>) -> tensor<4x600x1533xf32> {
  %0 = stablehlo.slice %arg0 [0:4, 160:760, 0:1533] : (tensor<4x760x1533xf32>) -> tensor<4x600x1533xf32>
  return %0 : tensor<4x600x1533xf32>
}

// CHECK: func.func @test_slice_needs_pad_after_ret(%arg0: tensor<4x760x1533xf32>) -> tensor<4x600x1533xf32> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:     %0 = stablehlo.pad %arg0, %cst, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<4x768x1536xf32>
// CHECK-NEXT:     %1 = stablehlo.slice %0 [0:4, 160:768, 0:1536] : (tensor<4x768x1536xf32>) -> tensor<4x608x1536xf32>
// CHECK-NEXT:     %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:     %2 = stablehlo.pad %1, %cst_0, low = [0, 0, 0], high = [0, 32, 0], interior = [0, 0, 0] : (tensor<4x608x1536xf32>, tensor<f32>) -> tensor<4x640x1536xf32>
// CHECK-NEXT:     %3 = stablehlo.slice %2 [0:4, 0:600, 0:1533] : (tensor<4x640x1536xf32>) -> tensor<4x600x1533xf32>
// CHECK-NEXT:     return %3 : tensor<4x600x1533xf32>
// CHECK-NEXT: }
