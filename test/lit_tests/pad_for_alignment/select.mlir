// RUN: enzymexlamlir-opt --pad-for-alignment --canonicalize --allow-unregistered-dialect %s | FileCheck %s

func.func @test_select(%arg0: tensor<4x760x1533xi1>, %arg1: tensor<4x760x1533xf32>, %arg2: tensor<4x760x1533xf32>) -> tensor<4x760x1533xf32> {
  %0 = stablehlo.select %arg0, %arg1, %arg2 : (tensor<4x760x1533xi1>, tensor<4x760x1533xf32>, tensor<4x760x1533xf32>) -> tensor<4x760x1533xf32>
  return %0 : tensor<4x760x1533xf32>
}

// CHECK-LABEL: func.func @test_select(%arg0: tensor<4x760x1533xi1>, %arg1: tensor<4x760x1533xf32>, %arg2: tensor<4x760x1533xf32>) -> tensor<4x760x1533xf32> {
// CHECK-DAG:      %[[false:.*]] = stablehlo.constant dense<false> : tensor<i1>
// CHECK-DAG:      %[[zero:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-DAG:      %[[pad0:.*]] = stablehlo.pad %arg0, %[[false]], low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xi1>, tensor<i1>) -> tensor<4x768x1536xi1>
// CHECK-DAG:      %[[pad1:.*]] = stablehlo.pad %arg1, %[[zero]], low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<4x768x1536xf32>
// CHECK-DAG:      %[[pad2:.*]] = stablehlo.pad %arg2, %[[zero]], low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<4x768x1536xf32>
// CHECK-NEXT:     %3 = stablehlo.select %[[pad0]], %[[pad1]], %[[pad2]] : tensor<4x768x1536xi1>, tensor<4x768x1536xf32>
// CHECK-NEXT:     %4 = stablehlo.slice %3 [0:4, 0:760, 0:1533] : (tensor<4x768x1536xf32>) -> tensor<4x760x1533xf32>
// CHECK-NEXT:     return %4 : tensor<4x760x1533xf32>
// CHECK-NEXT: }
