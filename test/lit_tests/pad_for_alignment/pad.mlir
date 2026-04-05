// RUN: enzymexlamlir-opt --pad-for-alignment --canonicalize --allow-unregistered-dialect %s | FileCheck %s

func.func @test_high(%arg0: tensor<4x760x1533xf32>) -> tensor<5x760x1534xf32> {
  %cst = stablehlo.constant dense<0.0> : tensor<f32>
  %0 = stablehlo.pad %arg0, %cst, low = [0, 0, 0], high = [1, 0, 1], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<5x760x1534xf32>
  return %0 : tensor<5x760x1534xf32>
}

// CHECK: func.func @test_high(%arg0: tensor<4x760x1533xf32>) -> tensor<5x760x1534xf32> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:     %0 = stablehlo.pad %arg0, %cst, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<4x768x1536xf32>
// CHECK-NEXT:     %1 = stablehlo.pad %0, %cst, low = [0, 0, 0], high = [1, 0, 0], interior = [0, 0, 0] : (tensor<4x768x1536xf32>, tensor<f32>) -> tensor<5x768x1536xf32>
// CHECK-NEXT:     %2 = stablehlo.slice %1 [0:5, 0:760, 0:1534] : (tensor<5x768x1536xf32>) -> tensor<5x760x1534xf32>
// CHECK-NEXT:     return %2 : tensor<5x760x1534xf32>
// CHECK-NEXT: }

func.func @test_high_padded_input(%arg0: tensor<4x768x1536xf32>) -> tensor<5x768x1537xf32> {
  %cst = stablehlo.constant dense<0.0> : tensor<f32>
  %0 = stablehlo.pad %arg0, %cst, low = [0, 0, 0], high = [1, 0, 1], interior = [0, 0, 0] : (tensor<4x768x1536xf32>, tensor<f32>) -> tensor<5x768x1537xf32>
  return %0 : tensor<5x768x1537xf32>
}

// CHECK: func.func @test_high_padded_input(%arg0: tensor<4x768x1536xf32>) -> tensor<5x768x1537xf32> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:     %0 = stablehlo.pad %arg0, %cst, low = [0, 0, 0], high = [1, 0, 128], interior = [0, 0, 0] : (tensor<4x768x1536xf32>, tensor<f32>) -> tensor<5x768x1664xf32>
// CHECK-NEXT:     %1 = stablehlo.slice %0 [0:5, 0:768, 0:1537] : (tensor<5x768x1664xf32>) -> tensor<5x768x1537xf32>
// CHECK-NEXT:     return %1 : tensor<5x768x1537xf32>
// CHECK-NEXT: }

func.func @test_low(%arg0: tensor<4x760x1533xf32>) -> tensor<5x760x1534xf32> {
  %cst = stablehlo.constant dense<0.0> : tensor<f32>
  %0 = stablehlo.pad %arg0, %cst, low = [1, 0, 1], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<5x760x1534xf32>
  return %0 : tensor<5x760x1534xf32>
}

// CHECK: func.func @test_low(%arg0: tensor<4x760x1533xf32>) -> tensor<5x760x1534xf32> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:     %0 = stablehlo.pad %arg0, %cst, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<4x768x1536xf32>
// CHECK-NEXT:     %1 = stablehlo.pad %0, %cst, low = [1, 0, 1], high = [0, 0, -1], interior = [0, 0, 0] : (tensor<4x768x1536xf32>, tensor<f32>) -> tensor<5x768x1536xf32>
// CHECK-NEXT:     %2 = stablehlo.slice %1 [0:5, 0:760, 0:1534] : (tensor<5x768x1536xf32>) -> tensor<5x760x1534xf32>
// CHECK-NEXT:     return %2 : tensor<5x760x1534xf32>
// CHECK-NEXT: }

func.func @test_low_padded_input(%arg0: tensor<4x768x1536xf32>) -> tensor<5x768x1537xf32> {
  %cst = stablehlo.constant dense<0.0> : tensor<f32>
  %0 = stablehlo.pad %arg0, %cst, low = [1, 0, 1], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<4x768x1536xf32>, tensor<f32>) -> tensor<5x768x1537xf32>
  return %0 : tensor<5x768x1537xf32>
}

// CHECK: func.func @test_low_padded_input(%arg0: tensor<4x768x1536xf32>) -> tensor<5x768x1537xf32> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:     %0 = stablehlo.pad %arg0, %cst, low = [1, 0, 1], high = [0, 0, 127], interior = [0, 0, 0] : (tensor<4x768x1536xf32>, tensor<f32>) -> tensor<5x768x1664xf32>
// CHECK-NEXT:     %1 = stablehlo.slice %0 [0:5, 0:768, 0:1537] : (tensor<5x768x1664xf32>) -> tensor<5x768x1537xf32>
// CHECK-NEXT:     return %1 : tensor<5x768x1537xf32>
// CHECK-NEXT: }

func.func @test_interior(%arg0: tensor<4x767x1536xf32>) -> tensor<7x1533x1536xf32> {
  %cst = stablehlo.constant dense<0.0> : tensor<f32>
  %0 = stablehlo.pad %arg0, %cst, low = [0, 0, 0], high = [0, 0, 0], interior = [1, 1, 0] : (tensor<4x767x1536xf32>, tensor<f32>) -> tensor<7x1533x1536xf32>
  return %0 : tensor<7x1533x1536xf32>
}

// CHECK: func.func @test_interior(%arg0: tensor<4x767x1536xf32>) -> tensor<7x1533x1536xf32> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:     %0 = stablehlo.pad %arg0, %cst, low = [0, 0, 0], high = [0, 1, 0], interior = [0, 0, 0] : (tensor<4x767x1536xf32>, tensor<f32>) -> tensor<4x768x1536xf32>
// CHECK-NEXT:     %1 = stablehlo.pad %0, %cst, low = [0, 0, 0], high = [0, 1, 0], interior = [1, 1, 0] : (tensor<4x768x1536xf32>, tensor<f32>) -> tensor<7x1536x1536xf32>
// CHECK-NEXT:     %2 = stablehlo.slice %1 [0:7, 0:1533, 0:1536] : (tensor<7x1536x1536xf32>) -> tensor<7x1533x1536xf32>
// CHECK-NEXT:     return %2 : tensor<7x1533x1536xf32>
// CHECK-NEXT: }

func.func @test_interior_padded_input(%arg0: tensor<4x768x1536xf32>) -> tensor<7x1535x1536xf32> {
  %cst = stablehlo.constant dense<0.0> : tensor<f32>
  %0 = stablehlo.pad %arg0, %cst, low = [0, 0, 0], high = [0, 0, 0], interior = [1, 1, 0] : (tensor<4x768x1536xf32>, tensor<f32>) -> tensor<7x1535x1536xf32>
  return %0 : tensor<7x1535x1536xf32>
}

// CHECK: func.func @test_interior_padded_input(%arg0: tensor<4x768x1536xf32>) -> tensor<7x1535x1536xf32> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:     %0 = stablehlo.pad %arg0, %cst, low = [0, 0, 0], high = [0, 1, 0], interior = [1, 1, 0] : (tensor<4x768x1536xf32>, tensor<f32>) -> tensor<7x1536x1536xf32>
// CHECK-NEXT:     %1 = stablehlo.slice %0 [0:7, 0:1535, 0:1536] : (tensor<7x1536x1536xf32>) -> tensor<7x1535x1536xf32>
// CHECK-NEXT:     return %1 : tensor<7x1535x1536xf32>
// CHECK-NEXT: }

func.func @test_low_high(%17: tensor<4x1519x3056xf32>) -> tensor<4x1521x3056xf32> {
  %cst = stablehlo.constant dense<0.0> : tensor<f32>
  %18 = stablehlo.pad %17, %cst, low = [0, 1, 0], high = [0, 1, 0], interior = [0, 0, 0] : (tensor<4x1519x3056xf32>, tensor<f32>) -> tensor<4x1521x3056xf32>
  return %18 : tensor<4x1521x3056xf32>
}

// CHECK: func.func @test_low_high(%arg0: tensor<4x1519x3056xf32>) -> tensor<4x1521x3056xf32> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:     %0 = stablehlo.pad %arg0, %cst, low = [0, 0, 0], high = [0, 17, 16], interior = [0, 0, 0] : (tensor<4x1519x3056xf32>, tensor<f32>) -> tensor<4x1536x3072xf32>
// CHECK-NEXT:     %1 = stablehlo.pad %0, %cst, low = [0, 1, 0], high = [0, -1, 0], interior = [0, 0, 0] : (tensor<4x1536x3072xf32>, tensor<f32>) -> tensor<4x1536x3072xf32>
// CHECK-NEXT:     %2 = stablehlo.slice %1 [0:4, 0:1521, 0:3056] : (tensor<4x1536x3072xf32>) -> tensor<4x1521x3056xf32>
// CHECK-NEXT:     return %2 : tensor<4x1521x3056xf32>
// CHECK-NEXT: }

func.func @test_no_result_padding_needed(%arg0: tensor<4x760x1533xf32>) -> tensor<6x768x1536xf32> {
  %cst = stablehlo.constant dense<0.0> : tensor<f32>
  %0 = stablehlo.pad %arg0, %cst, low = [1, 4, 0], high = [1, 4, 3], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<6x768x1536xf32>
  return %0 : tensor<6x768x1536xf32>
}

// CHECK: func.func @test_no_result_padding_needed(%arg0: tensor<4x760x1533xf32>) -> tensor<6x768x1536xf32> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:     %0 = stablehlo.pad %arg0, %cst, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<4x768x1536xf32>
// CHECK-NEXT:     %1 = stablehlo.pad %0, %cst, low = [1, 4, 0], high = [1, -4, 0], interior = [0, 0, 0] : (tensor<4x768x1536xf32>, tensor<f32>) -> tensor<6x768x1536xf32>
// CHECK-NEXT:     return %1 : tensor<6x768x1536xf32>
// CHECK-NEXT: }

func.func @test_low_all_zero_input_padding(%arg0: tensor<3x1520x3056xf32>) -> tensor<4x1520x3056xf32> {
  %cst = stablehlo.constant dense<0.0> : tensor<f32>
  %0 = stablehlo.pad %arg0, %cst, low = [1, 0, 0], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<3x1520x3056xf32>, tensor<f32>) -> tensor<4x1520x3056xf32>
  return %0 : tensor<4x1520x3056xf32>
}

// CHECK: func.func @test_low_all_zero_input_padding(%arg0: tensor<3x1520x3056xf32>) -> tensor<4x1520x3056xf32> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:     %0 = stablehlo.pad %arg0, %cst, low = [0, 0, 0], high = [0, 16, 16], interior = [0, 0, 0] : (tensor<3x1520x3056xf32>, tensor<f32>) -> tensor<3x1536x3072xf32>
// CHECK-NEXT:     %1 = stablehlo.pad %0, %cst, low = [1, 0, 0], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<3x1536x3072xf32>, tensor<f32>) -> tensor<4x1536x3072xf32>
// CHECK-NEXT:     %2 = stablehlo.slice %1 [0:4, 0:1520, 0:3056] : (tensor<4x1536x3072xf32>) -> tensor<4x1520x3056xf32>
// CHECK-NEXT:     return %2 : tensor<4x1520x3056xf32>
// CHECK-NEXT: }

