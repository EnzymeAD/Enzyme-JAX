// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=first" %s | FileCheck --check-prefix=FIRST %s
// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=last" %s | FileCheck --check-prefix=LAST %s
// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=tuple" %s | FileCheck --check-prefix=TUPLE %s
// RUN: enzymexlamlir-opt %s --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=first" | stablehlo-translate - --interpret --allow-unregistered-dialect
// RUN: enzymexlamlir-opt %s --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=last" | stablehlo-translate - --interpret --allow-unregistered-dialect
// RUN: enzymexlamlir-opt %s --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=tuple" | stablehlo-translate - --interpret --allow-unregistered-dialect

func.func @pad_zero(%arg0: tensor<2xf64>) -> tensor<4xf64> {
  %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
  %0 = stablehlo.pad %arg0, %cst, low = [1], high = [1], interior = [0] : (tensor<2xf64>, tensor<f64>) -> tensor<4xf64>
  return %0 : tensor<4xf64>
}

func.func @pad_tuple(%arg0: tensor<2xf64>) -> tensor<4xf64> {
  %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
  %0 = stablehlo.pad %arg0, %cst, low = [1], high = [1], interior = [0] : (tensor<2xf64>, tensor<f64>) -> tensor<4xf64>
  return %0 : tensor<4xf64>
}

// FIRST-LABEL: func.func @pad_zero(
// FIRST:     %[[V_0:.*]] = stablehlo.convert %arg0 : (tensor<2xf64>) -> tensor<2xf32>
// FIRST:     %[[V_1:.*]] = stablehlo.convert %[[V_0]] : (tensor<2xf32>) -> tensor<2xf64>
// FIRST:     %[[V_2:.*]] = stablehlo.subtract %arg0, %[[V_1]] : tensor<2xf64>
// FIRST:     %[[V_3:.*]] = stablehlo.convert %[[V_2]] : (tensor<2xf64>) -> tensor<2xf32>
// FIRST:     %[[V_4:.*]] = stablehlo.reshape %[[V_0]] : (tensor<2xf32>) -> tensor<1x2xf32>
// FIRST:     %[[V_5:.*]] = stablehlo.reshape %[[V_3]] : (tensor<2xf32>) -> tensor<1x2xf32>
// FIRST:     %[[V_6:.*]] = stablehlo.concatenate %[[V_4]], %[[V_5]], dim = 0 : (tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<2x2xf32>
// FIRST:     %[[CST:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<1xf32>
// FIRST:     %[[CST_0:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<1xf32>
// FIRST:     %[[V_7:.*]] = stablehlo.concatenate %[[CST]], %[[CST_0]], dim = 0 : (tensor<1xf32>, tensor<1xf32>) -> tensor<2xf32>
// FIRST:     %[[CST_1:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// FIRST:     %[[V_8:.*]] = stablehlo.pad %[[V_6]], %[[CST_1]], low = [0, 1], high = [0, 1], interior = [0, 0] : (tensor<2x2xf32>, tensor<f32>) -> tensor<2x4xf32>
// FIRST:     %[[V_9:.*]] = stablehlo.convert %[[V_8]] : (tensor<2x4xf32>) -> tensor<2x4xf64>
// FIRST:     %[[CST_2:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// FIRST:     %[[V_10:.*]] = stablehlo.reduce(%[[V_9]] init: %[[CST_2]]) applies stablehlo.add across dimensions = [0] : (tensor<2x4xf64>, tensor<f64>) -> tensor<4xf64>
// FIRST:     return %[[V_10]] : tensor<4xf64>

// LAST-LABEL: func.func @pad_zero(
// LAST:     %[[V_0:.*]] = stablehlo.convert %arg0 : (tensor<2xf64>) -> tensor<2xf32>
// LAST:     %[[V_1:.*]] = stablehlo.convert %[[V_0]] : (tensor<2xf32>) -> tensor<2xf64>
// LAST:     %[[V_2:.*]] = stablehlo.subtract %arg0, %[[V_1]] : tensor<2xf64>
// LAST:     %[[V_3:.*]] = stablehlo.convert %[[V_2]] : (tensor<2xf64>) -> tensor<2xf32>
// LAST:     %[[V_4:.*]] = stablehlo.reshape %[[V_0]] : (tensor<2xf32>) -> tensor<2x1xf32>
// LAST:     %[[V_5:.*]] = stablehlo.reshape %[[V_3]] : (tensor<2xf32>) -> tensor<2x1xf32>
// LAST:     %[[V_6:.*]] = stablehlo.concatenate %[[V_4]], %[[V_5]], dim = 1 : (tensor<2x1xf32>, tensor<2x1xf32>) -> tensor<2x2xf32>
// LAST:     %[[CST:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<1xf32>
// LAST:     %[[CST_0:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<1xf32>
// LAST:     %[[V_7:.*]] = stablehlo.concatenate %[[CST]], %[[CST_0]], dim = 0 : (tensor<1xf32>, tensor<1xf32>) -> tensor<2xf32>
// LAST:     %[[CST_1:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// LAST:     %[[V_8:.*]] = stablehlo.pad %[[V_6]], %[[CST_1]], low = [1, 0], high = [1, 0], interior = [0, 0] : (tensor<2x2xf32>, tensor<f32>) -> tensor<4x2xf32>
// LAST:     %[[V_9:.*]] = stablehlo.convert %[[V_8]] : (tensor<4x2xf32>) -> tensor<4x2xf64>
// LAST:     %[[CST_2:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// LAST:     %[[V_10:.*]] = stablehlo.reduce(%[[V_9]] init: %[[CST_2]]) applies stablehlo.add across dimensions = [1] : (tensor<4x2xf64>, tensor<f64>) -> tensor<4xf64>
// LAST:     return %[[V_10]] : tensor<4xf64>

// TUPLE-LABEL: func.func @pad_tuple(
// TUPLE:     %[[V_0:.*]] = stablehlo.convert %arg0 : (tensor<2xf64>) -> tensor<2xf32>
// TUPLE:     %[[V_1:.*]] = stablehlo.convert %[[V_0]] : (tensor<2xf32>) -> tensor<2xf64>
// TUPLE:     %[[V_2:.*]] = stablehlo.subtract %arg0, %[[V_1]] : tensor<2xf64>
// TUPLE:     %[[V_3:.*]] = stablehlo.convert %[[V_2]] : (tensor<2xf64>) -> tensor<2xf32>
// TUPLE:     %[[V_4:.*]] = stablehlo.tuple %[[V_0]], %[[V_3]] : tuple<tensor<2xf32>, tensor<2xf32>>
// TUPLE:     %[[CST:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// TUPLE:     %[[CST_0:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// TUPLE:     %[[V_5:.*]] = stablehlo.tuple %[[CST]], %[[CST_0]] : tuple<tensor<f32>, tensor<f32>>
// TUPLE:     %[[V_6:.*]] = stablehlo.get_tuple_element %[[V_4]][0] : (tuple<tensor<2xf32>, tensor<2xf32>>) -> tensor<2xf32>
// TUPLE:     %[[V_7:.*]] = stablehlo.get_tuple_element %[[V_4]][1] : (tuple<tensor<2xf32>, tensor<2xf32>>) -> tensor<2xf32>
// TUPLE:     %[[V_8:.*]] = stablehlo.pad %[[V_6]], %[[CST]], low = [1], high = [1], interior = [0] : (tensor<2xf32>, tensor<f32>) -> tensor<4xf32>
// TUPLE:     %[[V_9:.*]] = stablehlo.pad %[[V_7]], %[[CST_0]], low = [1], high = [1], interior = [0] : (tensor<2xf32>, tensor<f32>) -> tensor<4xf32>
// TUPLE:     %[[V_10:.*]] = stablehlo.tuple %[[V_8]], %[[V_9]] : tuple<tensor<4xf32>, tensor<4xf32>>
// TUPLE:     %[[V_11:.*]] = stablehlo.convert %[[V_8]] : (tensor<4xf32>) -> tensor<4xf64>
// TUPLE:     %[[V_12:.*]] = stablehlo.convert %[[V_9]] : (tensor<4xf32>) -> tensor<4xf64>
// TUPLE:     %[[V_13:.*]] = stablehlo.add %[[V_11]], %[[V_12]] : tensor<4xf64>
// TUPLE:     return %[[V_13]] : tensor<4xf64>

// FIRST-LABEL: func.func @main

// LAST-LABEL: func.func @main

// TUPLE-LABEL: func.func @main

func.func @main() attributes {enzyme.no_multifloat} {
  %cst = stablehlo.constant dense<[1.1, 2.2]> : tensor<2xf64>
  
  %expected_mf = stablehlo.constant dense<[0.0, 1.0999999999999996, 2.1999999999999993, 0.0]> : tensor<4xf64>
  
  %res = func.call @pad_zero(%cst) : (tensor<2xf64>) -> tensor<4xf64>
  
  // Strict test against Julia MultiFloat
  "check.expect_close"(%res, %expected_mf) {max_ulp_difference = 3 : ui64} : (tensor<4xf64>, tensor<4xf64>) -> ()
  
  // Approximate test against regular f64
  %expected_f64 = stablehlo.constant dense<[0.0, 1.1, 2.2, 0.0]> : tensor<4xf64>
  "check.expect_close"(%res, %expected_f64) {max_ulp_difference = 3 : ui64} : (tensor<4xf64>, tensor<4xf64>) -> ()
  return
}
// FIRST:     %[[CST:.*]] = stablehlo.constant dense<{{[^>]*}}> : tensor<2xf64>
// FIRST:     %[[CST_0:.*]] = stablehlo.constant dense<{{[^>]*}}> : tensor<4xf64>
// FIRST:     %[[V_0:.*]] = call @pad_zero(%[[CST]]) : (tensor<2xf64>) -> tensor<4xf64>
// FIRST:     check.expect_close %[[V_0]], %[[CST_0]], max_ulp_difference = 3 : tensor<4xf64>, tensor<4xf64>
// FIRST:     %[[CST_1:.*]] = stablehlo.constant dense<{{[^>]*}}> : tensor<4xf64>
// FIRST:     check.expect_close %[[V_0]], %[[CST_1]], max_ulp_difference = 3 : tensor<4xf64>, tensor<4xf64>
// FIRST:     return
