// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=first" %s | FileCheck --check-prefix=FIRST %s
// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=last" %s | FileCheck --check-prefix=LAST %s
// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=tuple" %s | FileCheck --check-prefix=TUPLE %s

func.func @rotate(%arg0: tensor<4x4xf64>) -> tensor<4x4xf64> {
  %0 = "enzymexla.rotate"(%arg0) <{amount = 1 : i32, dimension = 0 : i32}> : (tensor<4x4xf64>) -> tensor<4x4xf64>
  return %0 : tensor<4x4xf64>
}

// TUPLE-LABEL: func.func @rotate
// TUPLE:     %[[V_0:.*]] = stablehlo.convert %arg0 : (tensor<4x4xf64>) -> tensor<4x4xf32>
// TUPLE:     %[[V_1:.*]] = stablehlo.convert %[[V_0]] : (tensor<4x4xf32>) -> tensor<4x4xf64>
// TUPLE:     %[[V_2:.*]] = stablehlo.subtract %arg0, %[[V_1]] : tensor<4x4xf64>
// TUPLE:     %[[V_3:.*]] = stablehlo.convert %[[V_2]] : (tensor<4x4xf64>) -> tensor<4x4xf32>
// TUPLE:     %[[V_4:.*]] = stablehlo.tuple %[[V_0]], %[[V_3]] : tuple<tensor<4x4xf32>, tensor<4x4xf32>>
// TUPLE:     %[[V_5:.*]] = stablehlo.get_tuple_element %[[V_4]][0] : (tuple<tensor<4x4xf32>, tensor<4x4xf32>>) -> tensor<4x4xf32>
// TUPLE:     %[[V_6:.*]] = stablehlo.get_tuple_element %[[V_4]][1] : (tuple<tensor<4x4xf32>, tensor<4x4xf32>>) -> tensor<4x4xf32>
// TUPLE:     %[[V_7:.*]] = "enzymexla.rotate"(%[[V_5]]) <{amount = 1 : i32, dimension = 0 : i32}> : (tensor<4x4xf32>) -> tensor<4x4xf32>
// TUPLE:     %[[V_8:.*]] = "enzymexla.rotate"(%[[V_6]]) <{amount = 1 : i32, dimension = 0 : i32}> : (tensor<4x4xf32>) -> tensor<4x4xf32>
// TUPLE:     %[[V_9:.*]] = stablehlo.tuple %[[V_7]], %[[V_8]] : tuple<tensor<4x4xf32>, tensor<4x4xf32>>
// TUPLE:     %[[V_10:.*]] = stablehlo.convert %[[V_7]] : (tensor<4x4xf32>) -> tensor<4x4xf64>
// TUPLE:     %[[V_11:.*]] = stablehlo.convert %[[V_8]] : (tensor<4x4xf32>) -> tensor<4x4xf64>
// TUPLE:     %[[V_12:.*]] = stablehlo.add %[[V_10]], %[[V_11]] : tensor<4x4xf64>
// TUPLE:     return %[[V_12]] : tensor<4x4xf64>

// FIRST-LABEL: func.func @rotate
// FIRST:     %[[V_0:.*]] = stablehlo.convert %arg0 : (tensor<4x4xf64>) -> tensor<4x4xf32>
// FIRST:     %[[V_1:.*]] = stablehlo.convert %[[V_0]] : (tensor<4x4xf32>) -> tensor<4x4xf64>
// FIRST:     %[[V_2:.*]] = stablehlo.subtract %arg0, %[[V_1]] : tensor<4x4xf64>
// FIRST:     %[[V_3:.*]] = stablehlo.convert %[[V_2]] : (tensor<4x4xf64>) -> tensor<4x4xf32>
// FIRST:     %[[V_4:.*]] = stablehlo.reshape %[[V_0]] : (tensor<4x4xf32>) -> tensor<1x4x4xf32>
// FIRST:     %[[V_5:.*]] = stablehlo.reshape %[[V_3]] : (tensor<4x4xf32>) -> tensor<1x4x4xf32>
// FIRST:     %[[V_6:.*]] = stablehlo.concatenate %[[V_4]], %[[V_5]], dim = 0 : (tensor<1x4x4xf32>, tensor<1x4x4xf32>) -> tensor<2x4x4xf32>
// FIRST:     %[[V_7:.*]] = "enzymexla.rotate"(%[[V_6]]) <{amount = 1 : i32, dimension = 1 : i32}> : (tensor<2x4x4xf32>) -> tensor<2x4x4xf32>
// FIRST:     %[[V_8:.*]] = stablehlo.convert %[[V_7]] : (tensor<2x4x4xf32>) -> tensor<2x4x4xf64>
// FIRST:     %[[CST:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// FIRST:     %[[V_9:.*]] = stablehlo.reduce(%[[V_8]] init: %[[CST]]) applies stablehlo.add across dimensions = [0] : (tensor<2x4x4xf64>, tensor<f64>) -> tensor<4x4xf64>
// FIRST:     return %[[V_9]] : tensor<4x4xf64>

// LAST-LABEL: func.func @rotate
// LAST:     %[[V_0:.*]] = stablehlo.convert %arg0 : (tensor<4x4xf64>) -> tensor<4x4xf32>
// LAST:     %[[V_1:.*]] = stablehlo.convert %[[V_0]] : (tensor<4x4xf32>) -> tensor<4x4xf64>
// LAST:     %[[V_2:.*]] = stablehlo.subtract %arg0, %[[V_1]] : tensor<4x4xf64>
// LAST:     %[[V_3:.*]] = stablehlo.convert %[[V_2]] : (tensor<4x4xf64>) -> tensor<4x4xf32>
// LAST:     %[[V_4:.*]] = stablehlo.reshape %[[V_0]] : (tensor<4x4xf32>) -> tensor<4x4x1xf32>
// LAST:     %[[V_5:.*]] = stablehlo.reshape %[[V_3]] : (tensor<4x4xf32>) -> tensor<4x4x1xf32>
// LAST:     %[[V_6:.*]] = stablehlo.concatenate %[[V_4]], %[[V_5]], dim = 2 : (tensor<4x4x1xf32>, tensor<4x4x1xf32>) -> tensor<4x4x2xf32>
// LAST:     %[[V_7:.*]] = "enzymexla.rotate"(%[[V_6]]) <{amount = 1 : i32, dimension = 0 : i32}> : (tensor<4x4x2xf32>) -> tensor<4x4x2xf32>
// LAST:     %[[V_8:.*]] = stablehlo.convert %[[V_7]] : (tensor<4x4x2xf32>) -> tensor<4x4x2xf64>
// LAST:     %[[CST:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// LAST:     %[[V_9:.*]] = stablehlo.reduce(%[[V_8]] init: %[[CST]]) applies stablehlo.add across dimensions = [2] : (tensor<4x4x2xf64>, tensor<f64>) -> tensor<4x4xf64>
// LAST:     return %[[V_9]] : tensor<4x4xf64>
