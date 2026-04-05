// RUN: enzymexlamlir-opt --pad-for-alignment --canonicalize --allow-unregistered-dialect %s | FileCheck %s

func.func @test_dot_general(%arg0: tensor<4x760x1533xf32>, %arg1: tensor<4x1533x760xf32>) -> tensor<4x760x760xf32> {
  %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<4x760x1533xf32>, tensor<4x1533x760xf32>) -> tensor<4x760x760xf32>
  return %0 : tensor<4x760x760xf32>
}

// CHECK: func.func @test_dot_general(%arg0: tensor<4x760x1533xf32>, %arg1: tensor<4x1533x760xf32>) -> tensor<4x760x760xf32> {
// CHECK-NEXT: %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-DAG: %[[pad0:.*]] = stablehlo.pad %arg0, {{%cst[_0-9]*}}, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<4x768x1536xf32>
// CHECK-DAG: %[[pad1:.*]] = stablehlo.pad %arg1, {{%cst[_0-9]*}}, low = [0, 0, 0], high = [0, 3, 8], interior = [0, 0, 0] : (tensor<4x1533x760xf32>, tensor<f32>) -> tensor<4x1536x768xf32>
// CHECK-NEXT %2 = stablehlo.dot_general %[[pad0]], %[[pad1]], batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<4x768x1536xf32>, tensor<4x1536x768xf32>) -> tensor<4x768x768xf32>
// CHECK-NEXT %3 = stablehlo.slice %2 [0:4, 0:760, 0:760] : (tensor<4x768x768xf32>) -> tensor<4x760x760xf32>
// CHECK-NEXT return %3 : tensor<4x760x760xf32>
// CHECK-NEXT }
