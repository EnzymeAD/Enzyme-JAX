// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=first" %s | FileCheck %s

// 2-operand concatenate on dim 0
func.func @concat_2op(%arg0: tensor<3x4xf64>, %arg1: tensor<1x4xf64>) -> tensor<4x4xf64> {
  %0 = stablehlo.concatenate %arg0, %arg1, dim = 0 : (tensor<3x4xf64>, tensor<1x4xf64>) -> tensor<4x4xf64>
  return %0 : tensor<4x4xf64>
}
// CHECK-LABEL: func.func @concat_2op
// CHECK: %[[C0:.*]] = stablehlo.convert %arg0 : (tensor<3x4xf64>) -> tensor<3x4xf32>
// CHECK: %[[C1:.*]] = stablehlo.convert %[[C0]] : (tensor<3x4xf32>) -> tensor<3x4xf64>
// CHECK: %[[SUB1:.*]] = stablehlo.subtract %arg0, %[[C1]] : tensor<3x4xf64>
// CHECK: %[[C2:.*]] = stablehlo.convert %[[SUB1]] : (tensor<3x4xf64>) -> tensor<3x4xf32>
// CHECK: %[[R1:.*]] = stablehlo.reshape %[[C0]] : (tensor<3x4xf32>) -> tensor<1x3x4xf32>
// CHECK: %[[R2:.*]] = stablehlo.reshape %[[C2]] : (tensor<3x4xf32>) -> tensor<1x3x4xf32>
// CHECK: %[[CAT1:.*]] = stablehlo.concatenate %[[R1]], %[[R2]], dim = 0 : (tensor<1x3x4xf32>, tensor<1x3x4xf32>) -> tensor<2x3x4xf32>
// CHECK: %[[C3:.*]] = stablehlo.convert %arg1 : (tensor<1x4xf64>) -> tensor<1x4xf32>
// CHECK: %[[C4:.*]] = stablehlo.convert %[[C3]] : (tensor<1x4xf32>) -> tensor<1x4xf64>
// CHECK: %[[SUB2:.*]] = stablehlo.subtract %arg1, %[[C4]] : tensor<1x4xf64>
// CHECK: %[[C5:.*]] = stablehlo.convert %[[SUB2]] : (tensor<1x4xf64>) -> tensor<1x4xf32>
// CHECK: %[[R3:.*]] = stablehlo.reshape %[[C3]] : (tensor<1x4xf32>) -> tensor<1x1x4xf32>
// CHECK: %[[R4:.*]] = stablehlo.reshape %[[C5]] : (tensor<1x4xf32>) -> tensor<1x1x4xf32>
// CHECK: %[[CAT2:.*]] = stablehlo.concatenate %[[R3]], %[[R4]], dim = 0 : (tensor<1x1x4xf32>, tensor<1x1x4xf32>) -> tensor<2x1x4xf32>
// CHECK: %[[CAT3:.*]] = stablehlo.concatenate %[[CAT1]], %[[CAT2]], dim = 1 : (tensor<2x3x4xf32>, tensor<2x1x4xf32>) -> tensor<2x4x4xf32>
// CHECK: %[[C6:.*]] = stablehlo.convert %[[CAT3]] : (tensor<2x4x4xf32>) -> tensor<2x4x4xf64>
// CHECK: %[[ZERO:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK: %[[RES:.*]] = stablehlo.reduce(%[[C6]] init: %[[ZERO]]) applies stablehlo.add across dimensions = [0] : (tensor<2x4x4xf64>, tensor<f64>) -> tensor<4x4xf64>
// CHECK: return %[[RES]] : tensor<4x4xf64>
// 3-operand concatenate on dim 0 (halo assembly pattern)
func.func @concat_3op(%arg0: tensor<1x4xf64>, %arg1: tensor<2x4xf64>, %arg2: tensor<1x4xf64>) -> tensor<4x4xf64> {
  %0 = stablehlo.concatenate %arg0, %arg1, %arg2, dim = 0 : (tensor<1x4xf64>, tensor<2x4xf64>, tensor<1x4xf64>) -> tensor<4x4xf64>
  return %0 : tensor<4x4xf64>
}
// CHECK-LABEL: func.func @concat_3op
// CHECK: %[[C0:.*]] = stablehlo.convert %arg0 : (tensor<1x4xf64>) -> tensor<1x4xf32>
// CHECK: %[[C1:.*]] = stablehlo.convert %[[C0]] : (tensor<1x4xf32>) -> tensor<1x4xf64>
// CHECK: %[[SUB1:.*]] = stablehlo.subtract %arg0, %[[C1]] : tensor<1x4xf64>
// CHECK: %[[C2:.*]] = stablehlo.convert %[[SUB1]] : (tensor<1x4xf64>) -> tensor<1x4xf32>
// CHECK: %[[R1:.*]] = stablehlo.reshape %[[C0]] : (tensor<1x4xf32>) -> tensor<1x1x4xf32>
// CHECK: %[[R2:.*]] = stablehlo.reshape %[[C2]] : (tensor<1x4xf32>) -> tensor<1x1x4xf32>
// CHECK: %[[CAT1:.*]] = stablehlo.concatenate %[[R1]], %[[R2]], dim = 0 : (tensor<1x1x4xf32>, tensor<1x1x4xf32>) -> tensor<2x1x4xf32>
// CHECK: %[[C3:.*]] = stablehlo.convert %arg1 : (tensor<2x4xf64>) -> tensor<2x4xf32>
// CHECK: %[[C4:.*]] = stablehlo.convert %[[C3]] : (tensor<2x4xf32>) -> tensor<2x4xf64>
// CHECK: %[[SUB2:.*]] = stablehlo.subtract %arg1, %[[C4]] : tensor<2x4xf64>
// CHECK: %[[C5:.*]] = stablehlo.convert %[[SUB2]] : (tensor<2x4xf64>) -> tensor<2x4xf32>
// CHECK: %[[R3:.*]] = stablehlo.reshape %[[C3]] : (tensor<2x4xf32>) -> tensor<1x2x4xf32>
// CHECK: %[[R4:.*]] = stablehlo.reshape %[[C5]] : (tensor<2x4xf32>) -> tensor<1x2x4xf32>
// CHECK: %[[CAT2:.*]] = stablehlo.concatenate %[[R3]], %[[R4]], dim = 0 : (tensor<1x2x4xf32>, tensor<1x2x4xf32>) -> tensor<2x2x4xf32>
// CHECK: %[[C7:.*]] = stablehlo.convert %arg2 : (tensor<1x4xf64>) -> tensor<1x4xf32>
// CHECK: %[[C8:.*]] = stablehlo.convert %[[C7]] : (tensor<1x4xf32>) -> tensor<1x4xf64>
// CHECK: %[[SUB3:.*]] = stablehlo.subtract %arg2, %[[C8]] : tensor<1x4xf64>
// CHECK: %[[C9:.*]] = stablehlo.convert %[[SUB3]] : (tensor<1x4xf64>) -> tensor<1x4xf32>
// CHECK: %[[R5:.*]] = stablehlo.reshape %[[C7]] : (tensor<1x4xf32>) -> tensor<1x1x4xf32>
// CHECK: %[[R6:.*]] = stablehlo.reshape %[[C9]] : (tensor<1x4xf32>) -> tensor<1x1x4xf32>
// CHECK: %[[CAT4:.*]] = stablehlo.concatenate %[[R5]], %[[R6]], dim = 0 : (tensor<1x1x4xf32>, tensor<1x1x4xf32>) -> tensor<2x1x4xf32>
// CHECK: %[[CAT5:.*]] = stablehlo.concatenate %[[CAT1]], %[[CAT2]], %[[CAT4]], dim = 1 : (tensor<2x1x4xf32>, tensor<2x2x4xf32>, tensor<2x1x4xf32>) -> tensor<2x4x4xf32>
// CHECK: %[[C10:.*]] = stablehlo.convert %[[CAT5]] : (tensor<2x4x4xf32>) -> tensor<2x4x4xf64>
// CHECK: %[[ZERO:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK: %[[RES:.*]] = stablehlo.reduce(%[[C10]] init: %[[ZERO]]) applies stablehlo.add across dimensions = [0] : (tensor<2x4x4xf64>, tensor<f64>) -> tensor<4x4xf64>
// CHECK: return %[[RES]] : tensor<4x4xf64>

// 5-operand concatenate on dim 1 (stencil reconstruction pattern)
func.func @concat_5op(%a: tensor<4x1x8xf64>, %b: tensor<4x1x8xf64>, %c: tensor<4x3x8xf64>, %d: tensor<4x1x8xf64>, %e: tensor<4x1x8xf64>) -> tensor<4x7x8xf64> {
  %0 = stablehlo.concatenate %a, %b, %c, %d, %e, dim = 1 : (tensor<4x1x8xf64>, tensor<4x1x8xf64>, tensor<4x3x8xf64>, tensor<4x1x8xf64>, tensor<4x1x8xf64>) -> tensor<4x7x8xf64>
  return %0 : tensor<4x7x8xf64>
}
// CHECK-LABEL: func.func @concat_5op
// CHECK: %[[C_A:.*]] = stablehlo.convert %arg0 : (tensor<4x1x8xf64>) -> tensor<4x1x8xf32>
// CHECK: %[[C_A_F64:.*]] = stablehlo.convert %[[C_A]] : (tensor<4x1x8xf32>) -> tensor<4x1x8xf64>
// CHECK: %[[SUB_A:.*]] = stablehlo.subtract %arg0, %[[C_A_F64]] : tensor<4x1x8xf64>
// CHECK: %[[C_A_LO:.*]] = stablehlo.convert %[[SUB_A]] : (tensor<4x1x8xf64>) -> tensor<4x1x8xf32>
// CHECK: %[[R_A_HI:.*]] = stablehlo.reshape %[[C_A]] : (tensor<4x1x8xf32>) -> tensor<1x4x1x8xf32>
// CHECK: %[[R_A_LO:.*]] = stablehlo.reshape %[[C_A_LO]] : (tensor<4x1x8xf32>) -> tensor<1x4x1x8xf32>
// CHECK: %[[CAT_A:.*]] = stablehlo.concatenate %[[R_A_HI]], %[[R_A_LO]], dim = 0 : (tensor<1x4x1x8xf32>, tensor<1x4x1x8xf32>) -> tensor<2x4x1x8xf32>
// (Repeat for b, c, d, e)
// CHECK: %[[CAT_ALL:.*]] = stablehlo.concatenate %[[CAT_A]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, dim = 2 : (tensor<2x4x1x8xf32>, tensor<2x4x1x8xf32>, tensor<2x4x3x8xf32>, tensor<2x4x1x8xf32>, tensor<2x4x1x8xf32>) -> tensor<2x4x7x8xf32>
// CHECK: %[[C_ALL_F64:.*]] = stablehlo.convert %[[CAT_ALL]] : (tensor<2x4x7x8xf32>) -> tensor<2x4x7x8xf64>
// CHECK: %[[ZERO:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK: %[[RES:.*]] = stablehlo.reduce(%[[C_ALL_F64]] init: %[[ZERO]]) applies stablehlo.add across dimensions = [0] : (tensor<2x4x7x8xf64>, tensor<f64>) -> tensor<4x7x8xf64>
// CHECK: return %[[RES]] : tensor<4x7x8xf64>
// No f64 operations should remain inside any function body
// CHECK-NOT: stablehlo.concatenate {{.*}}f64
