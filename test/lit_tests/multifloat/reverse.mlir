// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=first" %s | FileCheck --check-prefix=FIRST %s
// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=last" %s | FileCheck --check-prefix=LAST %s
// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=tuple" %s | FileCheck --check-prefix=TUPLE %s
// RUN: enzymexlamlir-opt %s --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=first" | stablehlo-translate - --interpret --allow-unregistered-dialect

func.func @reverse(%arg0: tensor<2xf64>) -> tensor<2xf64> {
  %0 = stablehlo.reverse %arg0, dims = [0] : tensor<2xf64>
  return %0 : tensor<2xf64>
}

// TUPLE-LABEL: {{.*}}func.func @reverse(
// TUPLE:     %[[V_0:.*]] = stablehlo.convert %arg0 : (tensor<2xf64>) -> tensor<2xf32>
// TUPLE:     %[[V_1:.*]] = stablehlo.convert %[[V_0]] : (tensor<2xf32>) -> tensor<2xf64>
// TUPLE:     %[[V_2:.*]] = stablehlo.subtract %arg0, %[[V_1]] : tensor<2xf64>
// TUPLE:     %[[V_3:.*]] = stablehlo.convert %[[V_2]] : (tensor<2xf64>) -> tensor<2xf32>
// TUPLE:     %[[V_4:.*]] = stablehlo.tuple %[[V_0]], %[[V_3]] : tuple<tensor<2xf32>, tensor<2xf32>>
// TUPLE:     %[[V_5:.*]] = stablehlo.get_tuple_element %[[V_4]][0] : (tuple<tensor<2xf32>, tensor<2xf32>>) -> tensor<2xf32>
// TUPLE:     %[[V_6:.*]] = stablehlo.get_tuple_element %[[V_4]][1] : (tuple<tensor<2xf32>, tensor<2xf32>>) -> tensor<2xf32>
// TUPLE:     %[[V_7:.*]] = stablehlo.reverse %[[V_5]], dims = [0] : tensor<2xf32>
// TUPLE:     %[[V_8:.*]] = stablehlo.reverse %[[V_6]], dims = [0] : tensor<2xf32>
// TUPLE:     %[[V_9:.*]] = stablehlo.tuple %[[V_7]], %[[V_8]] : tuple<tensor<2xf32>, tensor<2xf32>>
// TUPLE:     %[[V_10:.*]] = stablehlo.convert %[[V_7]] : (tensor<2xf32>) -> tensor<2xf64>
// TUPLE:     %[[V_11:.*]] = stablehlo.convert %[[V_8]] : (tensor<2xf32>) -> tensor<2xf64>
// TUPLE:     %[[V_12:.*]] = stablehlo.add %[[V_10]], %[[V_11]] : tensor<2xf64>
// TUPLE:     return %[[V_12]] : tensor<2xf64>

// FIRST-LABEL: {{.*}}func.func @reverse(
// FIRST:     %[[V_0:.*]] = stablehlo.convert %arg0 : (tensor<2xf64>) -> tensor<2xf32>
// FIRST:     %[[V_1:.*]] = stablehlo.convert %[[V_0]] : (tensor<2xf32>) -> tensor<2xf64>
// FIRST:     %[[V_2:.*]] = stablehlo.subtract %arg0, %[[V_1]] : tensor<2xf64>
// FIRST:     %[[V_3:.*]] = stablehlo.convert %[[V_2]] : (tensor<2xf64>) -> tensor<2xf32>
// FIRST:     %[[V_4:.*]] = stablehlo.reshape %[[V_0]] : (tensor<2xf32>) -> tensor<1x2xf32>
// FIRST:     %[[V_5:.*]] = stablehlo.reshape %[[V_3]] : (tensor<2xf32>) -> tensor<1x2xf32>
// FIRST:     %[[V_6:.*]] = stablehlo.concatenate %[[V_4]], %[[V_5]], dim = 0 : (tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<2x2xf32>
// FIRST:     %[[V_7:.*]] = stablehlo.reverse %[[V_6]], dims = [1] : tensor<2x2xf32>
// FIRST:     %[[V_8:.*]] = stablehlo.convert %[[V_7]] : (tensor<2x2xf32>) -> tensor<2x2xf64>
// FIRST:     %[[CST:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// FIRST:     %[[V_9:.*]] = stablehlo.reduce(%[[V_8]] init: %[[CST]]) applies stablehlo.add across dimensions = [0] : (tensor<2x2xf64>, tensor<f64>) -> tensor<2xf64>
// FIRST:     return %[[V_9]] : tensor<2xf64>

// LAST-LABEL: {{.*}}func.func @reverse(
// LAST:     %[[V_0:.*]] = stablehlo.convert %arg0 : (tensor<2xf64>) -> tensor<2xf32>
// LAST:     %[[V_1:.*]] = stablehlo.convert %[[V_0]] : (tensor<2xf32>) -> tensor<2xf64>
// LAST:     %[[V_2:.*]] = stablehlo.subtract %arg0, %[[V_1]] : tensor<2xf64>
// LAST:     %[[V_3:.*]] = stablehlo.convert %[[V_2]] : (tensor<2xf64>) -> tensor<2xf32>
// LAST:     %[[V_4:.*]] = stablehlo.reshape %[[V_0]] : (tensor<2xf32>) -> tensor<2x1xf32>
// LAST:     %[[V_5:.*]] = stablehlo.reshape %[[V_3]] : (tensor<2xf32>) -> tensor<2x1xf32>
// LAST:     %[[V_6:.*]] = stablehlo.concatenate %[[V_4]], %[[V_5]], dim = 1 : (tensor<2x1xf32>, tensor<2x1xf32>) -> tensor<2x2xf32>
// LAST:     %[[V_7:.*]] = stablehlo.reverse %[[V_6]], dims = [0] : tensor<2x2xf32>
// LAST:     %[[V_8:.*]] = stablehlo.convert %[[V_7]] : (tensor<2x2xf32>) -> tensor<2x2xf64>
// LAST:     %[[CST:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// LAST:     %[[V_9:.*]] = stablehlo.reduce(%[[V_8]] init: %[[CST]]) applies stablehlo.add across dimensions = [1] : (tensor<2x2xf64>, tensor<f64>) -> tensor<2xf64>
// LAST:     return %[[V_9]] : tensor<2xf64>

// FIRST-LABEL: func.func @main

// LAST-LABEL: func.func @main

// TUPLE-LABEL: func.func @main

func.func @main() attributes {enzyme.no_multifloat} {
  %cst = stablehlo.constant dense<[1.1, 2.2]> : tensor<2xf64>
  
  %expected_mf = stablehlo.constant dense<[2.1999999999999993, 1.0999999999999996]> : tensor<2xf64>
  
  %res = func.call @reverse(%cst) : (tensor<2xf64>) -> tensor<2xf64>
  
  // Strict test against Julia MultiFloat
  "check.expect_close"(%res, %expected_mf) {max_ulp_difference = 3 : ui64} : (tensor<2xf64>, tensor<2xf64>) -> ()
  
  // Approximate test against regular f64
  %expected_f64 = stablehlo.constant dense<[2.2, 1.1]> : tensor<2xf64>
  "check.expect_close"(%res, %expected_f64) {max_ulp_difference = 3 : ui64} : (tensor<2xf64>, tensor<2xf64>) -> ()
  return
}
// FIRST:     %[[CST:.*]] = stablehlo.constant dense<{{[^>]*}}> : tensor<2xf64>
// FIRST:     %[[CST_0:.*]] = stablehlo.constant dense<{{[^>]*}}> : tensor<2xf64>
// FIRST:     %[[V_0:.*]] = call @reverse(%[[CST]]) : (tensor<2xf64>) -> tensor<2xf64>
// FIRST:     check.expect_close %[[V_0]], %[[CST_0]], max_ulp_difference = 3 : tensor<2xf64>, tensor<2xf64>
// FIRST:     %[[CST_1:.*]] = stablehlo.constant dense<{{[^>]*}}> : tensor<2xf64>
// FIRST:     check.expect_close %[[V_0]], %[[CST_1]], max_ulp_difference = 3 : tensor<2xf64>, tensor<2xf64>
// FIRST:     return
