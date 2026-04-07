// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=first expansion-size=2" %s | FileCheck --check-prefix=FIRST %s
// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=last expansion-size=2" %s | FileCheck --check-prefix=LAST %s
// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=tuple expansion-size=2" %s | FileCheck --check-prefix=TUPLE %s
// RUN: enzymexlamlir-opt %s --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=first" | stablehlo-translate - --interpret --allow-unregistered-dialect

func.func @convert_f64_to_f32(%arg0: tensor<2xf64>) -> tensor<2xf32> {
  %0 = stablehlo.convert %arg0 : (tensor<2xf64>) -> tensor<2xf32>
  return %0 : tensor<2xf32>
}

func.func @convert_f32_to_f64(%arg0: tensor<2xf32>) -> tensor<2xf64> {
  %0 = stablehlo.convert %arg0 : (tensor<2xf32>) -> tensor<2xf64>
  return %0 : tensor<2xf64>
}

func.func @convert_i32_to_f64(%arg0: tensor<2xi32>) -> tensor<2xf64> {
  %0 = stablehlo.convert %arg0 : (tensor<2xi32>) -> tensor<2xf64>
  return %0 : tensor<2xf64>
}

func.func @convert_f64_to_i32(%arg0: tensor<2xf64>) -> tensor<2xi32> {
  %0 = stablehlo.convert %arg0 : (tensor<2xf64>) -> tensor<2xi32>
  return %0 : tensor<2xi32>
}

// TUPLE-LABEL: func.func @convert_f64_to_f32(
// TUPLE:     %[[V_0:.*]] = stablehlo.convert %arg0 : (tensor<2xf64>) -> tensor<2xf32>
// TUPLE:     %[[V_1:.*]] = stablehlo.convert %[[V_0]] : (tensor<2xf32>) -> tensor<2xf64>
// TUPLE:     %[[V_2:.*]] = stablehlo.subtract %arg0, %[[V_1]] : tensor<2xf64>
// TUPLE:     %[[V_3:.*]] = stablehlo.convert %[[V_2]] : (tensor<2xf64>) -> tensor<2xf32>
// TUPLE:     %[[V_4:.*]] = stablehlo.tuple %[[V_0]], %[[V_3]] : tuple<tensor<2xf32>, tensor<2xf32>>
// TUPLE:     %[[V_5:.*]] = stablehlo.get_tuple_element %[[V_4]][0] : (tuple<tensor<2xf32>, tensor<2xf32>>) -> tensor<2xf32>
// TUPLE:     return %[[V_5]] : tensor<2xf32>
// TUPLE-LABEL: func.func @convert_f32_to_f64
// TUPLE:     %[[CST:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<2xf32>
// TUPLE:     %[[V_0:.*]] = stablehlo.tuple %arg0, %[[CST]] : tuple<tensor<2xf32>, tensor<2xf32>>
// TUPLE:     %[[V_1:.*]] = stablehlo.convert %arg0 : (tensor<2xf32>) -> tensor<2xf64>
// TUPLE:     %[[V_2:.*]] = stablehlo.convert %[[CST]] : (tensor<2xf32>) -> tensor<2xf64>
// TUPLE:     %[[V_3:.*]] = stablehlo.add %[[V_1]], %[[V_2]] : tensor<2xf64>
// TUPLE:     return %[[V_3]] : tensor<2xf64>
// TUPLE-LABEL: func.func @convert_i32_to_f64
// TUPLE:     %[[V_0:.*]] = stablehlo.convert %arg0 : (tensor<2xi32>) -> tensor<2xf32>
// TUPLE:     %[[V_1:.*]] = stablehlo.convert %[[V_0]] : (tensor<2xf32>) -> tensor<2xi32>
// TUPLE:     %[[V_2:.*]] = stablehlo.subtract %arg0, %[[V_1]] : tensor<2xi32>
// TUPLE:     %[[V_3:.*]] = stablehlo.convert %[[V_2]] : (tensor<2xi32>) -> tensor<2xf32>
// TUPLE:     %[[V_4:.*]] = stablehlo.tuple %[[V_0]], %[[V_3]] : tuple<tensor<2xf32>, tensor<2xf32>>
// TUPLE:     %[[V_5:.*]] = stablehlo.convert %[[V_0]] : (tensor<2xf32>) -> tensor<2xf64>
// TUPLE:     %[[V_6:.*]] = stablehlo.convert %[[V_3]] : (tensor<2xf32>) -> tensor<2xf64>
// TUPLE:     %[[V_7:.*]] = stablehlo.add %[[V_5]], %[[V_6]] : tensor<2xf64>
// TUPLE:     return %[[V_7]] : tensor<2xf64>
// TUPLE-LABEL: func.func @convert_f64_to_i32
// TUPLE:     %[[V_0:.*]] = stablehlo.convert %arg0 : (tensor<2xf64>) -> tensor<2xf32>
// TUPLE:     %[[V_1:.*]] = stablehlo.convert %[[V_0]] : (tensor<2xf32>) -> tensor<2xf64>
// TUPLE:     %[[V_2:.*]] = stablehlo.subtract %arg0, %[[V_1]] : tensor<2xf64>
// TUPLE:     %[[V_3:.*]] = stablehlo.convert %[[V_2]] : (tensor<2xf64>) -> tensor<2xf32>
// TUPLE:     %[[V_4:.*]] = stablehlo.tuple %[[V_0]], %[[V_3]] : tuple<tensor<2xf32>, tensor<2xf32>>
// TUPLE:     %[[V_5:.*]] = stablehlo.get_tuple_element %[[V_4]][0] : (tuple<tensor<2xf32>, tensor<2xf32>>) -> tensor<2xf32>
// TUPLE:     %[[V_6:.*]] = stablehlo.convert %[[V_5]] : (tensor<2xf32>) -> tensor<2xi32>
// TUPLE:     %[[V_7:.*]] = stablehlo.get_tuple_element %[[V_4]][1] : (tuple<tensor<2xf32>, tensor<2xf32>>) -> tensor<2xf32>
// TUPLE:     %[[V_8:.*]] = stablehlo.convert %[[V_7]] : (tensor<2xf32>) -> tensor<2xi32>
// TUPLE:     %[[V_9:.*]] = stablehlo.add %[[V_6]], %[[V_8]] : tensor<2xi32>
// TUPLE:     return %[[V_9]] : tensor<2xi32>

// FIRST-LABEL: func.func @convert_f64_to_f32(
// FIRST:     %[[V_0:.*]] = stablehlo.convert %arg0 : (tensor<2xf64>) -> tensor<2xf32>
// FIRST:     %[[V_1:.*]] = stablehlo.convert %[[V_0]] : (tensor<2xf32>) -> tensor<2xf64>
// FIRST:     %[[V_2:.*]] = stablehlo.subtract %arg0, %[[V_1]] : tensor<2xf64>
// FIRST:     %[[V_3:.*]] = stablehlo.convert %[[V_2]] : (tensor<2xf64>) -> tensor<2xf32>
// FIRST:     %[[V_4:.*]] = stablehlo.reshape %[[V_0]] : (tensor<2xf32>) -> tensor<1x2xf32>
// FIRST:     %[[V_5:.*]] = stablehlo.reshape %[[V_3]] : (tensor<2xf32>) -> tensor<1x2xf32>
// FIRST:     %[[V_6:.*]] = stablehlo.concatenate %[[V_4]], %[[V_5]], dim = 0 : (tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<2x2xf32>
// FIRST:     %[[V_7:.*]] = stablehlo.slice %[[V_6]] [0:1, 0:2] : (tensor<2x2xf32>) -> tensor<1x2xf32>
// FIRST:     %[[V_8:.*]] = stablehlo.reshape %[[V_7]] : (tensor<1x2xf32>) -> tensor<2xf32>
// FIRST:     return %[[V_8]] : tensor<2xf32>
// FIRST-LABEL: func.func @convert_f32_to_f64
// FIRST:     %[[V_0:.*]] = stablehlo.reshape %arg0 : (tensor<2xf32>) -> tensor<1x2xf32>
// FIRST:     %[[CST:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<1x2xf32>
// FIRST:     %[[V_1:.*]] = stablehlo.concatenate %[[V_0]], %[[CST]], dim = 0 : (tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<2x2xf32>
// FIRST:     %[[V_2:.*]] = stablehlo.convert %[[V_1]] : (tensor<2x2xf32>) -> tensor<2x2xf64>
// FIRST:     %[[CST_0:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// FIRST:     %[[V_3:.*]] = stablehlo.reduce(%[[V_2]] init: %[[CST_0]]) applies stablehlo.add across dimensions = [0] : (tensor<2x2xf64>, tensor<f64>) -> tensor<2xf64>
// FIRST:     return %[[V_3]] : tensor<2xf64>
// FIRST-LABEL: func.func @convert_i32_to_f64
// FIRST:     %[[V_0:.*]] = stablehlo.convert %arg0 : (tensor<2xi32>) -> tensor<2xf32>
// FIRST:     %[[V_1:.*]] = stablehlo.convert %[[V_0]] : (tensor<2xf32>) -> tensor<2xi32>
// FIRST:     %[[V_2:.*]] = stablehlo.subtract %arg0, %[[V_1]] : tensor<2xi32>
// FIRST:     %[[V_3:.*]] = stablehlo.convert %[[V_2]] : (tensor<2xi32>) -> tensor<2xf32>
// FIRST:     %[[V_4:.*]] = stablehlo.reshape %[[V_0]] : (tensor<2xf32>) -> tensor<1x2xf32>
// FIRST:     %[[V_5:.*]] = stablehlo.reshape %[[V_3]] : (tensor<2xf32>) -> tensor<1x2xf32>
// FIRST:     %[[V_6:.*]] = stablehlo.concatenate %[[V_4]], %[[V_5]], dim = 0 : (tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<2x2xf32>
// FIRST:     %[[V_7:.*]] = stablehlo.convert %[[V_6]] : (tensor<2x2xf32>) -> tensor<2x2xf64>
// FIRST:     %[[CST:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// FIRST:     %[[V_8:.*]] = stablehlo.reduce(%[[V_7]] init: %[[CST]]) applies stablehlo.add across dimensions = [0] : (tensor<2x2xf64>, tensor<f64>) -> tensor<2xf64>
// FIRST:     return %[[V_8]] : tensor<2xf64>
// FIRST-LABEL: func.func @convert_f64_to_i32
// FIRST:     %[[V_0:.*]] = stablehlo.convert %arg0 : (tensor<2xf64>) -> tensor<2xf32>
// FIRST:     %[[V_1:.*]] = stablehlo.convert %[[V_0]] : (tensor<2xf32>) -> tensor<2xf64>
// FIRST:     %[[V_2:.*]] = stablehlo.subtract %arg0, %[[V_1]] : tensor<2xf64>
// FIRST:     %[[V_3:.*]] = stablehlo.convert %[[V_2]] : (tensor<2xf64>) -> tensor<2xf32>
// FIRST:     %[[V_4:.*]] = stablehlo.reshape %[[V_0]] : (tensor<2xf32>) -> tensor<1x2xf32>
// FIRST:     %[[V_5:.*]] = stablehlo.reshape %[[V_3]] : (tensor<2xf32>) -> tensor<1x2xf32>
// FIRST:     %[[V_6:.*]] = stablehlo.concatenate %[[V_4]], %[[V_5]], dim = 0 : (tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<2x2xf32>
// FIRST:     %[[V_7:.*]] = stablehlo.slice %[[V_6]] [0:1, 0:2] : (tensor<2x2xf32>) -> tensor<1x2xf32>
// FIRST:     %[[V_8:.*]] = stablehlo.reshape %[[V_7]] : (tensor<1x2xf32>) -> tensor<2xf32>
// FIRST:     %[[V_9:.*]] = stablehlo.convert %[[V_8]] : (tensor<2xf32>) -> tensor<2xi32>
// FIRST:     %[[V_10:.*]] = stablehlo.slice %[[V_6]] [1:2, 0:2] : (tensor<2x2xf32>) -> tensor<1x2xf32>
// FIRST:     %[[V_11:.*]] = stablehlo.reshape %[[V_10]] : (tensor<1x2xf32>) -> tensor<2xf32>
// FIRST:     %[[V_12:.*]] = stablehlo.convert %[[V_11]] : (tensor<2xf32>) -> tensor<2xi32>
// FIRST:     %[[V_13:.*]] = stablehlo.add %[[V_9]], %[[V_12]] : tensor<2xi32>
// FIRST:     return %[[V_13]] : tensor<2xi32>

// LAST-LABEL: func.func @convert_f64_to_f32(
// LAST:     %[[V_0:.*]] = stablehlo.convert %arg0 : (tensor<2xf64>) -> tensor<2xf32>
// LAST:     %[[V_1:.*]] = stablehlo.convert %[[V_0]] : (tensor<2xf32>) -> tensor<2xf64>
// LAST:     %[[V_2:.*]] = stablehlo.subtract %arg0, %[[V_1]] : tensor<2xf64>
// LAST:     %[[V_3:.*]] = stablehlo.convert %[[V_2]] : (tensor<2xf64>) -> tensor<2xf32>
// LAST:     %[[V_4:.*]] = stablehlo.reshape %[[V_0]] : (tensor<2xf32>) -> tensor<2x1xf32>
// LAST:     %[[V_5:.*]] = stablehlo.reshape %[[V_3]] : (tensor<2xf32>) -> tensor<2x1xf32>
// LAST:     %[[V_6:.*]] = stablehlo.concatenate %[[V_4]], %[[V_5]], dim = 1 : (tensor<2x1xf32>, tensor<2x1xf32>) -> tensor<2x2xf32>
// LAST:     %[[V_7:.*]] = stablehlo.slice %[[V_6]] [0:2, 0:1] : (tensor<2x2xf32>) -> tensor<2x1xf32>
// LAST:     %[[V_8:.*]] = stablehlo.reshape %[[V_7]] : (tensor<2x1xf32>) -> tensor<2xf32>
// LAST:     return %[[V_8]] : tensor<2xf32>
// LAST-LABEL: func.func @convert_f32_to_f64
// LAST:     %[[V_0:.*]] = stablehlo.reshape %arg0 : (tensor<2xf32>) -> tensor<2x1xf32>
// LAST:     %[[CST:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<2x1xf32>
// LAST:     %[[V_1:.*]] = stablehlo.concatenate %[[V_0]], %[[CST]], dim = 1 : (tensor<2x1xf32>, tensor<2x1xf32>) -> tensor<2x2xf32>
// LAST:     %[[V_2:.*]] = stablehlo.convert %[[V_1]] : (tensor<2x2xf32>) -> tensor<2x2xf64>
// LAST:     %[[CST_0:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// LAST:     %[[V_3:.*]] = stablehlo.reduce(%[[V_2]] init: %[[CST_0]]) applies stablehlo.add across dimensions = [1] : (tensor<2x2xf64>, tensor<f64>) -> tensor<2xf64>
// LAST:     return %[[V_3]] : tensor<2xf64>
// LAST-LABEL: func.func @convert_i32_to_f64
// LAST:     %[[V_0:.*]] = stablehlo.convert %arg0 : (tensor<2xi32>) -> tensor<2xf32>
// LAST:     %[[V_1:.*]] = stablehlo.convert %[[V_0]] : (tensor<2xf32>) -> tensor<2xi32>
// LAST:     %[[V_2:.*]] = stablehlo.subtract %arg0, %[[V_1]] : tensor<2xi32>
// LAST:     %[[V_3:.*]] = stablehlo.convert %[[V_2]] : (tensor<2xi32>) -> tensor<2xf32>
// LAST:     %[[V_4:.*]] = stablehlo.reshape %[[V_0]] : (tensor<2xf32>) -> tensor<2x1xf32>
// LAST:     %[[V_5:.*]] = stablehlo.reshape %[[V_3]] : (tensor<2xf32>) -> tensor<2x1xf32>
// LAST:     %[[V_6:.*]] = stablehlo.concatenate %[[V_4]], %[[V_5]], dim = 1 : (tensor<2x1xf32>, tensor<2x1xf32>) -> tensor<2x2xf32>
// LAST:     %[[V_7:.*]] = stablehlo.convert %[[V_6]] : (tensor<2x2xf32>) -> tensor<2x2xf64>
// LAST:     %[[CST:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// LAST:     %[[V_8:.*]] = stablehlo.reduce(%[[V_7]] init: %[[CST]]) applies stablehlo.add across dimensions = [1] : (tensor<2x2xf64>, tensor<f64>) -> tensor<2xf64>
// LAST:     return %[[V_8]] : tensor<2xf64>
// LAST-LABEL: func.func @convert_f64_to_i32
// LAST:     %[[V_0:.*]] = stablehlo.convert %arg0 : (tensor<2xf64>) -> tensor<2xf32>
// LAST:     %[[V_1:.*]] = stablehlo.convert %[[V_0]] : (tensor<2xf32>) -> tensor<2xf64>
// LAST:     %[[V_2:.*]] = stablehlo.subtract %arg0, %[[V_1]] : tensor<2xf64>
// LAST:     %[[V_3:.*]] = stablehlo.convert %[[V_2]] : (tensor<2xf64>) -> tensor<2xf32>
// LAST:     %[[V_4:.*]] = stablehlo.reshape %[[V_0]] : (tensor<2xf32>) -> tensor<2x1xf32>
// LAST:     %[[V_5:.*]] = stablehlo.reshape %[[V_3]] : (tensor<2xf32>) -> tensor<2x1xf32>
// LAST:     %[[V_6:.*]] = stablehlo.concatenate %[[V_4]], %[[V_5]], dim = 1 : (tensor<2x1xf32>, tensor<2x1xf32>) -> tensor<2x2xf32>
// LAST:     %[[V_7:.*]] = stablehlo.slice %[[V_6]] [0:2, 0:1] : (tensor<2x2xf32>) -> tensor<2x1xf32>
// LAST:     %[[V_8:.*]] = stablehlo.reshape %[[V_7]] : (tensor<2x1xf32>) -> tensor<2xf32>
// LAST:     %[[V_9:.*]] = stablehlo.convert %[[V_8]] : (tensor<2xf32>) -> tensor<2xi32>
// LAST:     %[[V_10:.*]] = stablehlo.slice %[[V_6]] [0:2, 1:2] : (tensor<2x2xf32>) -> tensor<2x1xf32>
// LAST:     %[[V_11:.*]] = stablehlo.reshape %[[V_10]] : (tensor<2x1xf32>) -> tensor<2xf32>
// LAST:     %[[V_12:.*]] = stablehlo.convert %[[V_11]] : (tensor<2xf32>) -> tensor<2xi32>
// LAST:     %[[V_13:.*]] = stablehlo.add %[[V_9]], %[[V_12]] : tensor<2xi32>
// LAST:     return %[[V_13]] : tensor<2xi32>

// FIRST-LABEL: func.func @main

// LAST-LABEL: func.func @main

// TUPLE-LABEL: func.func @main

func.func @main() attributes {enzyme.no_multifloat} {
  %cst_f64 = stablehlo.constant dense<[1.1, 2.2]> : tensor<2xf64>
  
  // Test f64 -> f32
  %expected_f32 = stablehlo.constant dense<[1.100000023841858, 2.200000047683716]> : tensor<2xf32>
  %res_f32 = func.call @convert_f64_to_f32(%cst_f64) : (tensor<2xf64>) -> tensor<2xf32>
  "check.expect_close"(%res_f32, %expected_f32) {max_ulp_difference = 3 : ui64} : (tensor<2xf32>, tensor<2xf32>) -> ()
  "check.expect_close"(%res_f32, %expected_f32) {max_ulp_difference = 3 : ui64} : (tensor<2xf32>, tensor<2xf32>) -> ()
  
  // Test f32 -> f64
  %res_f64 = func.call @convert_f32_to_f64(%expected_f32) : (tensor<2xf32>) -> tensor<2xf64>
  %expected_f64 = stablehlo.convert %expected_f32 : (tensor<2xf32>) -> tensor<2xf64>
  "check.expect_close"(%res_f64, %expected_f64) {max_ulp_difference = 3 : ui64} : (tensor<2xf64>, tensor<2xf64>) -> ()
  "check.expect_close"(%res_f64, %expected_f64) {max_ulp_difference = 3 : ui64} : (tensor<2xf64>, tensor<2xf64>) -> ()
  
  // Test i32 -> f64
  %cst_i32 = stablehlo.constant dense<[1, 2]> : tensor<2xi32>
  %res_i32_f64 = func.call @convert_i32_to_f64(%cst_i32) : (tensor<2xi32>) -> tensor<2xf64>
  %expected_i32_f64 = stablehlo.constant dense<[1.0, 2.0]> : tensor<2xf64>
  "check.expect_close"(%res_i32_f64, %expected_i32_f64) {max_ulp_difference = 3 : ui64} : (tensor<2xf64>, tensor<2xf64>) -> ()
  "check.expect_close"(%res_i32_f64, %expected_i32_f64) {max_ulp_difference = 3 : ui64} : (tensor<2xf64>, tensor<2xf64>) -> ()
  
  // Test f64 -> i32
  %cst_f64_2 = stablehlo.constant dense<[1.1, 2.9]> : tensor<2xf64>
  %res_f64_i32 = func.call @convert_f64_to_i32(%cst_f64_2) : (tensor<2xf64>) -> tensor<2xi32>
  %expected_i32 = stablehlo.constant dense<[1, 2]> : tensor<2xi32>
  
  // Convert i32 to f64 to check!
  %res_f64_i32_f64 = stablehlo.convert %res_f64_i32 : (tensor<2xi32>) -> tensor<2xf64>
  %expected_i32_f64_2 = stablehlo.constant dense<[1.0, 2.0]> : tensor<2xf64>
  "check.expect_close"(%res_f64_i32_f64, %expected_i32_f64_2) {max_ulp_difference = 3 : ui64} : (tensor<2xf64>, tensor<2xf64>) -> ()
  "check.expect_close"(%res_f64_i32_f64, %expected_i32_f64_2) {max_ulp_difference = 3 : ui64} : (tensor<2xf64>, tensor<2xf64>) -> ()
  
  return
}
// FIRST:     %[[CST:.*]] = stablehlo.constant dense<{{[^>]*}}> : tensor<2xf64>
// FIRST:     %[[CST_0:.*]] = stablehlo.constant dense<{{[^>]*}}> : tensor<2xf32>
// FIRST:     %[[V_0:.*]] = call @convert_f64_to_f32(%[[CST]]) : (tensor<2xf64>) -> tensor<2xf32>
// FIRST:     check.expect_close %[[V_0]], %[[CST_0]], max_ulp_difference = 3 : tensor<2xf32>, tensor<2xf32>
// FIRST:     check.expect_close %[[V_0]], %[[CST_0]], max_ulp_difference = 3 : tensor<2xf32>, tensor<2xf32>
// FIRST:     %[[V_1:.*]] = call @convert_f32_to_f64(%[[CST_0]]) : (tensor<2xf32>) -> tensor<2xf64>
// FIRST:     %[[V_2:.*]] = stablehlo.convert %[[CST_0]] : (tensor<2xf32>) -> tensor<2xf64>
// FIRST:     check.expect_close %[[V_1]], %[[V_2]], max_ulp_difference = 3 : tensor<2xf64>, tensor<2xf64>
// FIRST:     check.expect_close %[[V_1]], %[[V_2]], max_ulp_difference = 3 : tensor<2xf64>, tensor<2xf64>
// FIRST:     %[[C:.*]] = stablehlo.constant dense<{{[^>]*}}> : tensor<2xi32>
// FIRST:     %[[V_3:.*]] = call @convert_i32_to_f64(%[[C]]) : (tensor<2xi32>) -> tensor<2xf64>
// FIRST:     %[[CST_1:.*]] = stablehlo.constant dense<{{[^>]*}}> : tensor<2xf64>
// FIRST:     check.expect_close %[[V_3]], %[[CST_1]], max_ulp_difference = 3 : tensor<2xf64>, tensor<2xf64>
// FIRST:     check.expect_close %[[V_3]], %[[CST_1]], max_ulp_difference = 3 : tensor<2xf64>, tensor<2xf64>
// FIRST:     %[[CST_2:.*]] = stablehlo.constant dense<{{[^>]*}}> : tensor<2xf64>
// FIRST:     %[[V_4:.*]] = call @convert_f64_to_i32(%[[CST_2]]) : (tensor<2xf64>) -> tensor<2xi32>
// FIRST:     %[[C_3:.*]] = stablehlo.constant dense<{{[^>]*}}> : tensor<2xi32>
// FIRST:     %[[V_5:.*]] = stablehlo.convert %[[V_4]] : (tensor<2xi32>) -> tensor<2xf64>
// FIRST:     %[[CST_4:.*]] = stablehlo.constant dense<{{[^>]*}}> : tensor<2xf64>
// FIRST:     check.expect_close %[[V_5]], %[[CST_4]], max_ulp_difference = 3 : tensor<2xf64>, tensor<2xf64>
// FIRST:     check.expect_close %[[V_5]], %[[CST_4]], max_ulp_difference = 3 : tensor<2xf64>, tensor<2xf64>
// FIRST:     return
