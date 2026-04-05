// RUN: enzymexlamlir-opt --pad-for-alignment --canonicalize --allow-unregistered-dialect %s | FileCheck %s

func.func @test_if(%arg0: tensor<i1>, %arg1: tensor<4x760x1533xf32>, %arg2: tensor<4x760x1533xf32>) -> tensor<4x760x1533xf32> {
  %0 = "stablehlo.if"(%arg0) ({
    stablehlo.return %arg1 : tensor<4x760x1533xf32>
  }, {
    stablehlo.return %arg2 : tensor<4x760x1533xf32>
  }) : (tensor<i1>) -> tensor<4x760x1533xf32>
  return %0 : tensor<4x760x1533xf32>
}

// CHECK-LABEL: func.func @test_if(%arg0: tensor<i1>, %arg1: tensor<4x760x1533xf32>, %arg2: tensor<4x760x1533xf32>) -> tensor<4x760x1533xf32> {
// CHECK-NEXT:  %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-DAG:   %[[pad1:.*]] = stablehlo.pad %arg1, %cst, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<4x768x1536xf32>
// CHECK-DAG:   %[[pad2:.*]] = stablehlo.pad %arg2, %cst, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<4x768x1536xf32>
// CHECK-NEXT:  %2 = "stablehlo.if"(%arg0) ({
// CHECK-NEXT:    stablehlo.return %[[pad1]] : tensor<4x768x1536xf32>
// CHECK-NEXT:  }, {
// CHECK-NEXT:    stablehlo.return %[[pad2]] : tensor<4x768x1536xf32>
// CHECK-NEXT:  }) : (tensor<i1>) -> tensor<4x768x1536xf32>
// CHECK-NEXT:  %3 = stablehlo.slice %2 [0:4, 0:760, 0:1533] : (tensor<4x768x1536xf32>) -> tensor<4x760x1533xf32>
// CHECK-NEXT:  return %3 : tensor<4x760x1533xf32>
// CHECK-NEXT: }

func.func @test_if_no_pad(%arg0: tensor<i1>, %arg1: tensor<4x768x1536xf32>, %arg2: tensor<4x768x1536xf32>) -> tensor<4x768x1536xf32> {
  %0 = "stablehlo.if"(%arg0) ({
    stablehlo.return %arg1 : tensor<4x768x1536xf32>
  }, {
    stablehlo.return %arg2 : tensor<4x768x1536xf32>
  }) : (tensor<i1>) -> tensor<4x768x1536xf32>
  return %0 : tensor<4x768x1536xf32>
}

// CHECK-LABEL: func.func @test_if_no_pad(%arg0: tensor<i1>, %arg1: tensor<4x768x1536xf32>, %arg2: tensor<4x768x1536xf32>) -> tensor<4x768x1536xf32> {
// CHECK-NEXT:     %0 = "stablehlo.if"(%arg0) ({
// CHECK-NEXT:       stablehlo.return %arg1 : tensor<4x768x1536xf32>
// CHECK-NEXT:     }, {
// CHECK-NEXT:       stablehlo.return %arg2 : tensor<4x768x1536xf32>
// CHECK-NEXT:     }) : (tensor<i1>) -> tensor<4x768x1536xf32>
// CHECK-NEXT:     return %0 : tensor<4x768x1536xf32>
// CHECK-NEXT: }

func.func @test_if_mixed(%arg0: tensor<i1>, %arg1: tensor<4x760x1533xf32>, %arg2: tensor<4x760x1533xf32>, %x: tensor<4x768x1536xf32>) -> (tensor<4x760x1533xf32>, tensor<4x768x1536xf32>) {
  %0:2 = "stablehlo.if"(%arg0) ({
    stablehlo.return %arg1, %x : tensor<4x760x1533xf32>, tensor<4x768x1536xf32>
  }, {
    stablehlo.return %arg2, %x : tensor<4x760x1533xf32>, tensor<4x768x1536xf32>
  }) : (tensor<i1>) -> (tensor<4x760x1533xf32>, tensor<4x768x1536xf32>)
  return %0#0, %0#1 : tensor<4x760x1533xf32>, tensor<4x768x1536xf32>
}

// CHECK-LABEL: func.func @test_if_mixed(%arg0: tensor<i1>, %arg1: tensor<4x760x1533xf32>, %arg2: tensor<4x760x1533xf32>, %arg3: tensor<4x768x1536xf32>) -> (tensor<4x760x1533xf32>, tensor<4x768x1536xf32>) {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-DAG:      %[[pad1:.*]] = stablehlo.pad %arg1, %cst, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<4x768x1536xf32>
// CHECK-DAG:      %[[pad2:.*]] = stablehlo.pad %arg2, %cst, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<4x768x1536xf32>
// CHECK-NEXT:     %2:2 = "stablehlo.if"(%arg0) ({
// CHECK-NEXT:       stablehlo.return %[[pad1]], %arg3 : tensor<4x768x1536xf32>, tensor<4x768x1536xf32>
// CHECK-NEXT:     }, {
// CHECK-NEXT:       stablehlo.return %[[pad2]], %arg3 : tensor<4x768x1536xf32>, tensor<4x768x1536xf32>
// CHECK-NEXT:     }) : (tensor<i1>) -> (tensor<4x768x1536xf32>, tensor<4x768x1536xf32>)
// CHECK-NEXT:     %3 = stablehlo.slice %2#0 [0:4, 0:760, 0:1533] : (tensor<4x768x1536xf32>) -> tensor<4x760x1533xf32>
// CHECK-NEXT:     return %3, %2#1 : tensor<4x760x1533xf32>, tensor<4x768x1536xf32>
// CHECK-NEXT: }
