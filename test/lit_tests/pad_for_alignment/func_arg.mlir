// RUN: enzymexlamlir-opt --pad-for-alignment --canonicalize --allow-unregistered-dialect %s | FileCheck %s

func.func @test_arg(%arg0: tensor<4x760x1533xf32>) -> tensor<4x760x1533xf32> {
  return %arg0 : tensor<4x760x1533xf32>
}

// CHECK: func.func @test_arg(%arg0: tensor<4x760x1533xf32>) -> tensor<4x760x1533xf32> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:     %0 = stablehlo.pad %arg0, %cst, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<4x768x1536xf32>
// CHECK-NEXT:     %1 = stablehlo.slice %0 [0:4, 0:760, 0:1533] : (tensor<4x768x1536xf32>) -> tensor<4x760x1533xf32>
// CHECK-NEXT:     return %1 : tensor<4x760x1533xf32>
// CHECK-NEXT: }

func.func @test_multi_arg(%arg0: tensor<4x760x1533xf32>, %arg1: tensor<4x760x1533xf32>, %arg2: tensor<4x760x1533xf32>, %arg3: tensor<4x760x1533xf32>) -> (tensor<4x760x1533xf32>, tensor<4x760x1533xf32>, tensor<4x760x1533xf32>, tensor<4x760x1533xf32>) {
  return %arg0, %arg1, %arg2, %arg3 : tensor<4x760x1533xf32>, tensor<4x760x1533xf32>, tensor<4x760x1533xf32>, tensor<4x760x1533xf32>
}

// CHECK-LABEL: func.func @test_multi_arg
// CHECK-SAME:                           (%arg0: tensor<4x760x1533xf32>, %arg1: tensor<4x760x1533xf32>, %arg2: tensor<4x760x1533xf32>, %arg3: tensor<4x760x1533xf32>) -> (tensor<4x760x1533xf32>, tensor<4x760x1533xf32>, tensor<4x760x1533xf32>, tensor<4x760x1533xf32>) {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-DAG:     %[[pad0:.*]] = stablehlo.pad %arg0, %cst, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<4x768x1536xf32>
// CHECK-DAG:     %[[pad1:.*]] = stablehlo.pad %arg1, %cst, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<4x768x1536xf32>
// CHECK-DAG:     %[[pad2:.*]] = stablehlo.pad %arg2, %cst, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<4x768x1536xf32>
// CHECK-DAG:     %[[pad3:.*]] = stablehlo.pad %arg3, %cst, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<4x768x1536xf32>
// CHECK-DAG:     %[[slice0:.*]] = stablehlo.slice %[[pad0]] [0:4, 0:760, 0:1533] : (tensor<4x768x1536xf32>) -> tensor<4x760x1533xf32>
// CHECK-DAG:     %[[slice1:.*]] = stablehlo.slice %[[pad1]] [0:4, 0:760, 0:1533] : (tensor<4x768x1536xf32>) -> tensor<4x760x1533xf32>
// CHECK-DAG:     %[[slice2:.*]] = stablehlo.slice %[[pad2]] [0:4, 0:760, 0:1533] : (tensor<4x768x1536xf32>) -> tensor<4x760x1533xf32>
// CHECK-DAG:     %[[slice3:.*]] = stablehlo.slice %[[pad3]] [0:4, 0:760, 0:1533] : (tensor<4x768x1536xf32>) -> tensor<4x760x1533xf32>
// CHECK-NEXT:     return %[[slice0]], %[[slice1]], %[[slice2]], %[[slice3]] : tensor<4x760x1533xf32>, tensor<4x760x1533xf32>, tensor<4x760x1533xf32>, tensor<4x760x1533xf32>
// CHECK-NEXT: }
