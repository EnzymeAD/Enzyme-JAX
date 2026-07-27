// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 expansion-size=1 convert-signatures=true" --canonicalize %s | FileCheck %s

func.func @simple_add(%arg0: tensor<4xf64>, %arg1: tensor<4xf64>) -> tensor<4xf64> {
  %0 = stablehlo.add %arg0, %arg1 : tensor<4xf64>
  return %0 : tensor<4xf64>
}

// CHECK-LABEL: func.func @simple_add(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: %[[ADD:.*]] = stablehlo.add %arg0, %arg1 : tensor<4xf32>
// CHECK: return %[[ADD]] : tensor<4xf32>
