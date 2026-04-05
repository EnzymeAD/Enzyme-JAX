// RUN: enzymexlamlir-opt --pad-for-alignment --allow-unregistered-dialect %s | FileCheck %s

func.func @test_rotate_on_aligned_dim(%arg0: tensor<4x760x1533xf32>) -> (tensor<4x760x1533xf32>, tensor<4x760x1533xf32>, tensor<4x760x1533xf32>) {
  %0:3 = "enzymexla.multi_rotate"(%arg0) <{left_amount = 1 : i32, right_amount = 1 : i32, dimension = 0 : i32}> : (tensor<4x760x1533xf32>) -> (tensor<4x760x1533xf32>, tensor<4x760x1533xf32>, tensor<4x760x1533xf32>)
  return %0#0, %0#1, %0#2 : tensor<4x760x1533xf32>, tensor<4x760x1533xf32>, tensor<4x760x1533xf32>
}

// CHECK-LABEL: func.func @test_rotate_on_aligned_dim(%arg0: tensor<4x760x1533xf32>) -> (tensor<4x760x1533xf32>, tensor<4x760x1533xf32>, tensor<4x760x1533xf32>) {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:     %0 = stablehlo.pad %arg0, %cst, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<4x768x1536xf32>
// CHECK-NEXT:     %1:3 = "enzymexla.multi_rotate"(%0) <{dimension = 0 : i32, left_amount = 1 : i32, right_amount = 1 : i32}> : (tensor<4x768x1536xf32>) -> (tensor<4x768x1536xf32>, tensor<4x768x1536xf32>, tensor<4x768x1536xf32>)
// CHECK-DAG:      %[[slice0:.*]] = stablehlo.slice %1#0 [0:4, 0:760, 0:1533] : (tensor<4x768x1536xf32>) -> tensor<4x760x1533xf32>
// CHECK-DAG:      %[[slice1:.*]] = stablehlo.slice %1#1 [0:4, 0:760, 0:1533] : (tensor<4x768x1536xf32>) -> tensor<4x760x1533xf32>
// CHECK-DAG:      %[[slice2:.*]] = stablehlo.slice %1#2 [0:4, 0:760, 0:1533] : (tensor<4x768x1536xf32>) -> tensor<4x760x1533xf32>
// CHECK-NEXT:     return %[[slice0]], %[[slice1]], %[[slice2]] : tensor<4x760x1533xf32>, tensor<4x760x1533xf32>, tensor<4x760x1533xf32>
// CHECK-NEXT: }
