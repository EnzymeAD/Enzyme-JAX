// RUN: enzymexlamlir-opt %s --arith-raise | FileCheck %s

module {
  func.func @f32(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
    %0 = "math.copysign"(%arg0, %arg1) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    return %0 : tensor<4xf32>
  }
  func.func @f64(%arg0: tensor<4xf64>, %arg1: tensor<4xf64>) -> tensor<4xf64> {
    %0 = "math.copysign"(%arg0, %arg1) : (tensor<4xf64>, tensor<4xf64>) -> tensor<4xf64>
    return %0 : tensor<4xf64>
  }
}

// The copysign returns a value with the magnitude of the first operand and the sign of the second operand.


// CHECK-LABEL:   func.func @f32(
// CHECK-SAME:                     %[[VAL_0:.*]]: tensor<4xf32>,
// CHECK-SAME:                     %[[VAL_1:.*]]: tensor<4xf32>) -> tensor<4xf32> {
// CHECK:           %[[VAL_2:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<4xf32>
// CHECK:           %[[VAL_3:.*]] = stablehlo.compare  GE, %[[VAL_1]], %[[VAL_2]] : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
// CHECK:           %[[VAL_4:.*]] = stablehlo.compare  GE, %[[VAL_0]], %[[VAL_2]] : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
// CHECK:           %[[VAL_5:.*]] = stablehlo.xor %[[VAL_3]], %[[VAL_4]] : tensor<4xi1>
// CHECK:           %[[VAL_6:.*]] = stablehlo.negate %[[VAL_0]] : tensor<4xf32>
// CHECK:           %[[VAL_7:.*]] = stablehlo.select %[[VAL_5]], %[[VAL_6]], %[[VAL_0]] : tensor<4xi1>, tensor<4xf32>
// CHECK:           return %[[VAL_7]] : tensor<4xf32>
// CHECK:         }

