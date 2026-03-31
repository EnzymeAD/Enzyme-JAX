// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=first" %s | FileCheck %s

// 2-operand concatenate on dim 0
func.func @concat_2op(%arg0: tensor<3x4xf64>, %arg1: tensor<1x4xf64>) -> tensor<4x4xf64> {
  %0 = stablehlo.concatenate %arg0, %arg1, dim = 0 : (tensor<3x4xf64>, tensor<1x4xf64>) -> tensor<4x4xf64>
  return %0 : tensor<4x4xf64>
}
// CHECK-LABEL: func.func @concat_2op
// CHECK: %[[A:.*]] = builtin.unrealized_conversion_cast %arg0 : tensor<3x4xf64> to tensor<2x3x4xf32>
// CHECK: %[[B:.*]] = builtin.unrealized_conversion_cast %arg1 : tensor<1x4xf64> to tensor<2x1x4xf32>
// CHECK: stablehlo.concatenate %[[A]], %[[B]], dim = 1 : (tensor<2x3x4xf32>, tensor<2x1x4xf32>) -> tensor<2x4x4xf32>

// 3-operand concatenate on dim 0 (halo assembly pattern)
func.func @concat_3op(%arg0: tensor<1x4xf64>, %arg1: tensor<2x4xf64>, %arg2: tensor<1x4xf64>) -> tensor<4x4xf64> {
  %0 = stablehlo.concatenate %arg0, %arg1, %arg2, dim = 0 : (tensor<1x4xf64>, tensor<2x4xf64>, tensor<1x4xf64>) -> tensor<4x4xf64>
  return %0 : tensor<4x4xf64>
}
// CHECK-LABEL: func.func @concat_3op
// CHECK: stablehlo.concatenate %{{.*}}, %{{.*}}, %{{.*}}, dim = 1 : (tensor<2x1x4xf32>, tensor<2x2x4xf32>, tensor<2x1x4xf32>) -> tensor<2x4x4xf32>

// 5-operand concatenate on dim 1 (stencil reconstruction pattern)
func.func @concat_5op(%a: tensor<4x1x8xf64>, %b: tensor<4x1x8xf64>, %c: tensor<4x3x8xf64>, %d: tensor<4x1x8xf64>, %e: tensor<4x1x8xf64>) -> tensor<4x7x8xf64> {
  %0 = stablehlo.concatenate %a, %b, %c, %d, %e, dim = 1 : (tensor<4x1x8xf64>, tensor<4x1x8xf64>, tensor<4x3x8xf64>, tensor<4x1x8xf64>, tensor<4x1x8xf64>) -> tensor<4x7x8xf64>
  return %0 : tensor<4x7x8xf64>
}
// CHECK-LABEL: func.func @concat_5op
// CHECK: stablehlo.concatenate %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, dim = 2 : (tensor<2x4x1x8xf32>, tensor<2x4x1x8xf32>, tensor<2x4x3x8xf32>, tensor<2x4x1x8xf32>, tensor<2x4x1x8xf32>) -> tensor<2x4x7x8xf32>

// No f64 operations should remain inside any function body
// CHECK-NOT: stablehlo.concatenate {{.*}}f64
