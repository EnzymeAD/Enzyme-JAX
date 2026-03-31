// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=first expansion-size=2" %s | FileCheck %s

func.func @rotate(%arg0: tensor<4x4xf64>) -> tensor<4x4xf64> {
  %0 = "enzymexla.rotate"(%arg0) <{amount = 1 : i32, dimension = 0 : i32}> : (tensor<4x4xf64>) -> tensor<4x4xf64>
  return %0 : tensor<4x4xf64>
}
// CHECK-LABEL: func.func @rotate
// CHECK: %[[CAST:.*]] = builtin.unrealized_conversion_cast %arg0 : tensor<4x4xf64> to tensor<2x4x4xf32>
// CHECK: %[[NEW_OP:.*]] = "enzymexla.rotate"(%[[CAST]]) <{amount = 1 : i32, dimension = 1 : i32}> : (tensor<2x4x4xf32>) -> tensor<2x4x4xf32>
// CHECK: %[[CAST_BACK:.*]] = builtin.unrealized_conversion_cast %[[NEW_OP]] : tensor<2x4x4xf32> to tensor<4x4xf64>
// CHECK: return %[[CAST_BACK]]
