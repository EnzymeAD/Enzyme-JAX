// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32" %s | FileCheck %s

func.func @test_combine_add(%arg0: tensor<1xf64>, %arg1: tensor<1xf64>, %arg2: tensor<1xf64>, %arg3: tensor<1xf64>) -> tensor<2xf64> {
  %0 = stablehlo.add %arg0, %arg1 : tensor<1xf64>
  %1 = stablehlo.add %arg2, %arg3 : tensor<1xf64>
  %2 = stablehlo.concatenate %0, %1, dim = 0 : (tensor<1xf64>, tensor<1xf64>) -> tensor<2xf64>
  return %2 : tensor<2xf64>
}

// CHECK-LABEL: func.func @test_combine_add
// CHECK: %[[C0:.*]] = stablehlo.convert %arg0 : (tensor<1xf64>) -> tensor<1xf32>
// CHECK: %[[C1:.*]] = stablehlo.convert %[[C0]] : (tensor<1xf32>) -> tensor<1xf64>
// CHECK: %[[SUB1:.*]] = stablehlo.subtract %arg0, %[[C1]] : tensor<1xf64>
// CHECK: %[[C2:.*]] = stablehlo.convert %[[SUB1]] : (tensor<1xf64>) -> tensor<1xf32>
// CHECK: %[[R1:.*]] = stablehlo.reshape %[[C0]] : (tensor<1xf32>) -> tensor<1x1xf32>
// CHECK: %[[R2:.*]] = stablehlo.reshape %[[C2]] : (tensor<1xf32>) -> tensor<1x1xf32>
// CHECK: %[[CAT1:.*]] = stablehlo.concatenate %[[R1]], %[[R2]], dim = 0 : (tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<2x1xf32>
// CHECK: %[[C3:.*]] = stablehlo.convert %arg1 : (tensor<1xf64>) -> tensor<1xf32>
// CHECK: %[[C4:.*]] = stablehlo.convert %[[C3]] : (tensor<1xf32>) -> tensor<1xf64>
// CHECK: %[[SUB2:.*]] = stablehlo.subtract %arg1, %[[C4]] : tensor<1xf64>
// CHECK: %[[C5:.*]] = stablehlo.convert %[[SUB2]] : (tensor<1xf64>) -> tensor<1xf32>
// CHECK: %[[R3:.*]] = stablehlo.reshape %[[C3]] : (tensor<1xf32>) -> tensor<1x1xf32>
// CHECK: %[[R4:.*]] = stablehlo.reshape %[[C5]] : (tensor<1xf32>) -> tensor<1x1xf32>
// CHECK: %[[CAT2:.*]] = stablehlo.concatenate %[[R3]], %[[R4]], dim = 0 : (tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<2x1xf32>

