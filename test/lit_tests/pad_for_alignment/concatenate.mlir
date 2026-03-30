// RUN: enzymexlamlir-opt --pad-for-alignment --canonicalize --allow-unregistered-dialect -split-input-file %s | FileCheck --dump-input=always %s

func.func @test_concatenate_dim_0(%arg0: tensor<4x760x1533xf32>, %arg1: tensor<4x760x1533xf32>) -> tensor<8x760x1533xf32> {
  %0 = stablehlo.concatenate %arg0, %arg1, dim = 0 : (tensor<4x760x1533xf32>, tensor<4x760x1533xf32>) -> tensor<8x760x1533xf32>
  return %0 : tensor<8x760x1533xf32>
}

// CHECK-LABEL: func.func @test_concatenate_dim_0(%arg0: tensor<4x760x1533xf32>, %arg1: tensor<4x760x1533xf32>) -> tensor<8x760x1533xf32> {
// CHECK-NEXT: %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-DAG:  %[[pad0:.*]] = stablehlo.pad %arg0, %cst, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<4x768x1536xf32>
// CHECK-DAG:  %[[pad1:.*]] = stablehlo.pad %arg1, %cst, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<4x768x1536xf32>
// CHECK-NEXT: %2 = stablehlo.concatenate %[[pad0]], %[[pad1]], dim = 0 : (tensor<4x768x1536xf32>, tensor<4x768x1536xf32>) -> tensor<8x768x1536xf32>
// CHECK-NEXT: %3 = stablehlo.slice %2 [0:8, 0:760, 0:1533] : (tensor<8x768x1536xf32>) -> tensor<8x760x1533xf32>
// CHECK-NEXT: return %3 : tensor<8x760x1533xf32>
// CHECK-NEXT: }

func.func @test_concatenate_dim_1(%arg0: tensor<4x760x1533xf32>, %arg1: tensor<4x760x1533xf32>) -> tensor<4x1520x1533xf32> {
  %0 = stablehlo.concatenate %arg0, %arg1, dim = 1 : (tensor<4x760x1533xf32>, tensor<4x760x1533xf32>) -> tensor<4x1520x1533xf32>
  return %0 : tensor<4x1520x1533xf32>
}

// CHECK-LABEL: func.func @test_concatenate_dim_1(%arg0: tensor<4x760x1533xf32>, %arg1: tensor<4x760x1533xf32>) -> tensor<4x1520x1533xf32> {
// CHECK-NEXT: %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-DAG: %[[pad0:.*]] = stablehlo.pad %arg0, %cst, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<4x768x1536xf32>
// CHECK-DAG: %[[pad1:.*]] = stablehlo.pad %arg1, %cst, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<4x768x1536xf32>
// CHECK-DAG: %[[slice0:.*]] = stablehlo.slice %[[pad0]] [0:4, 0:760, 0:1536] : (tensor<4x768x1536xf32>) -> tensor<4x760x1536xf32>
// CHECK-DAG: %[[slice1:.*]] = stablehlo.slice %[[pad1]] [0:4, 0:760, 0:1536] : (tensor<4x768x1536xf32>) -> tensor<4x760x1536xf32>
// CHECK-NEXT: %4 = stablehlo.concatenate %[[slice0]], %[[slice1]], dim = 1 : (tensor<4x760x1536xf32>, tensor<4x760x1536xf32>) -> tensor<4x1520x1536xf32>
// CHECK-NEXT: %5 = stablehlo.pad %4, %cst, low = [0, 0, 0], high = [0, 16, 0], interior = [0, 0, 0] : (tensor<4x1520x1536xf32>, tensor<f32>) -> tensor<4x1536x1536xf32>
// CHECK-NEXT: %6 = stablehlo.slice %5 [0:4, 0:1520, 0:1533] : (tensor<4x1536x1536xf32>) -> tensor<4x1520x1533xf32>
// CHECK-NEXT: return %6 : tensor<4x1520x1533xf32>
// CHECK-NEXT: }

func.func @test_concatenate_dim_1_various_shapes(%arg0: tensor<4x1x3056xf32>, %arg1: tensor<4x1x3056xf32>, %arg2: tensor<4x1516x3056xf32>, %arg3: tensor<4x1x3056xf32>, %arg4: tensor<4x1x3056xf32>) -> tensor<4x1520x3056xf32> {
  %0 = stablehlo.concatenate %arg0, %arg1, %arg2, %arg3, %arg4, dim = 1 : (tensor<4x1x3056xf32>, tensor<4x1x3056xf32>, tensor<4x1516x3056xf32>, tensor<4x1x3056xf32>, tensor<4x1x3056xf32>) -> tensor<4x1520x3056xf32>
  return %0 : tensor<4x1520x3056xf32>
}

// CHECK-LABEL: func.func @test_concatenate_dim_1_various_shapes(%arg0: tensor<4x1x3056xf32>, %arg1: tensor<4x1x3056xf32>, %arg2: tensor<4x1516x3056xf32>, %arg3: tensor<4x1x3056xf32>, %arg4: tensor<4x1x3056xf32>) -> tensor<4x1520x3056xf32> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-DAG:      %[[pad0:.*]] = stablehlo.pad %arg0, %cst, low = [0, 0, 0], high = [0, 0, 16], interior = [0, 0, 0] : (tensor<4x1x3056xf32>, tensor<f32>) -> tensor<4x1x3072xf32>
// CHECK-DAG:      %[[pad1:.*]] = stablehlo.pad %arg1, %cst, low = [0, 0, 0], high = [0, 0, 16], interior = [0, 0, 0] : (tensor<4x1x3056xf32>, tensor<f32>) -> tensor<4x1x3072xf32>
// CHECK-DAG:      %[[pad2:.*]] = stablehlo.pad %arg2, %cst, low = [0, 0, 0], high = [0, 20, 16], interior = [0, 0, 0] : (tensor<4x1516x3056xf32>, tensor<f32>) -> tensor<4x1536x3072xf32>
// CHECK-DAG:      %[[pad3:.*]] = stablehlo.pad %arg3, %cst, low = [0, 0, 0], high = [0, 0, 16], interior = [0, 0, 0] : (tensor<4x1x3056xf32>, tensor<f32>) -> tensor<4x1x3072xf32>
// CHECK-DAG:      %[[pad4:.*]] = stablehlo.pad %arg4, %cst, low = [0, 0, 0], high = [0, 0, 16], interior = [0, 0, 0] : (tensor<4x1x3056xf32>, tensor<f32>) -> tensor<4x1x3072xf32>
// CHECK-NEXT:     %5 = stablehlo.slice %[[pad2]] [0:4, 0:1516, 0:3072] : (tensor<4x1536x3072xf32>) -> tensor<4x1516x3072xf32>
// CHECK-NEXT:     %6 = stablehlo.concatenate %[[pad0]], %[[pad1]], %5, %[[pad3]], %[[pad4]], dim = 1 : (tensor<4x1x3072xf32>, tensor<4x1x3072xf32>, tensor<4x1516x3072xf32>, tensor<4x1x3072xf32>, tensor<4x1x3072xf32>) -> tensor<4x1520x3072xf32>
// CHECK-NEXT:     %7 = stablehlo.pad %6, %cst, low = [0, 0, 0], high = [0, 16, 0], interior = [0, 0, 0] : (tensor<4x1520x3072xf32>, tensor<f32>) -> tensor<4x1536x3072xf32>
// CHECK-NEXT:     %8 = stablehlo.slice %7 [0:4, 0:1520, 0:3056] : (tensor<4x1536x3072xf32>) -> tensor<4x1520x3056xf32>
// CHECK-NEXT:     return %8 : tensor<4x1520x3056xf32>
// CHECK-NEXT: }
