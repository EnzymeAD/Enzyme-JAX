// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f32 target-type=bf16 concat-dimension=first expansion-size=2" %s | FileCheck --check-prefix=FIRST %s
// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f32 target-type=bf16 concat-dimension=last expansion-size=2" %s | FileCheck --check-prefix=LAST %s
// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f32 target-type=bf16 concat-dimension=tuple expansion-size=2" %s | FileCheck --check-prefix=TUPLE %s
// RUN: enzymexlamlir-opt %s --multi-float-conversion="source-type=f32 target-type=bf16 concat-dimension=first expansion-size=2" | stablehlo-translate - --interpret --allow-unregistered-dialect
// RUN: enzymexlamlir-opt %s --multi-float-conversion="source-type=f32 target-type=bf16 concat-dimension=last expansion-size=2" | stablehlo-translate - --interpret --allow-unregistered-dialect
// RUN: enzymexlamlir-opt %s --multi-float-conversion="source-type=f32 target-type=bf16 concat-dimension=tuple expansion-size=2" | stablehlo-translate - --interpret --allow-unregistered-dialect

func.func @convert_f32_to_f64_expanded(%arg0: tensor<2xf32>) -> tensor<2xf64> {
  %0 = stablehlo.convert %arg0 : (tensor<2xf32>) -> tensor<2xf64>
  return %0 : tensor<2xf64>
}

// FIRST-LABEL: func.func @convert_f32_to_f64_expanded
// FIRST:     %[[V_0:.*]] = stablehlo.convert %arg0 : (tensor<2xf32>) -> tensor<2xbf16>
// FIRST:     %[[V_1:.*]] = stablehlo.convert %[[V_0]] : (tensor<2xbf16>) -> tensor<2xf32>
// FIRST:     %[[V_2:.*]] = stablehlo.subtract %arg0, %[[V_1]] : tensor<2xf32>
// FIRST:     %[[V_3:.*]] = stablehlo.convert %[[V_2]] : (tensor<2xf32>) -> tensor<2xbf16>
// FIRST:     %[[V_4:.*]] = stablehlo.reshape %[[V_0]] : (tensor<2xbf16>) -> tensor<1x2xbf16>
// FIRST:     %[[V_5:.*]] = stablehlo.reshape %[[V_3]] : (tensor<2xbf16>) -> tensor<1x2xbf16>
// FIRST:     %[[V_6:.*]] = stablehlo.concatenate %[[V_4]], %[[V_5]], dim = 0 : (tensor<1x2xbf16>, tensor<1x2xbf16>) -> tensor<2x2xbf16>
// FIRST:     %[[V_7:.*]] = stablehlo.convert %[[V_6]] : (tensor<2x2xbf16>) -> tensor<2x2xf64>
// FIRST:     %[[CST:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// FIRST:     %[[V_8:.*]] = stablehlo.reduce(%[[V_7]] init: %[[CST]]) applies stablehlo.add across dimensions = [0] : (tensor<2x2xf64>, tensor<f64>) -> tensor<2xf64>
// FIRST:     return %[[V_8]] : tensor<2xf64>

// LAST-LABEL: func.func @convert_f32_to_f64_expanded
// LAST:     %[[V_0:.*]] = stablehlo.convert %arg0 : (tensor<2xf32>) -> tensor<2xbf16>
// LAST:     %[[V_1:.*]] = stablehlo.convert %[[V_0]] : (tensor<2xbf16>) -> tensor<2xf32>
// LAST:     %[[V_2:.*]] = stablehlo.subtract %arg0, %[[V_1]] : tensor<2xf32>
// LAST:     %[[V_3:.*]] = stablehlo.convert %[[V_2]] : (tensor<2xf32>) -> tensor<2xbf16>
// LAST:     %[[V_4:.*]] = stablehlo.reshape %[[V_0]] : (tensor<2xbf16>) -> tensor<2x1xbf16>
// LAST:     %[[V_5:.*]] = stablehlo.reshape %[[V_3]] : (tensor<2xbf16>) -> tensor<2x1xbf16>
// LAST:     %[[V_6:.*]] = stablehlo.concatenate %[[V_4]], %[[V_5]], dim = 1 : (tensor<2x1xbf16>, tensor<2x1xbf16>) -> tensor<2x2xbf16>
// LAST:     %[[V_7:.*]] = stablehlo.convert %[[V_6]] : (tensor<2x2xbf16>) -> tensor<2x2xf64>
// LAST:     %[[CST:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// LAST:     %[[V_8:.*]] = stablehlo.reduce(%[[V_7]] init: %[[CST]]) applies stablehlo.add across dimensions = [1] : (tensor<2x2xf64>, tensor<f64>) -> tensor<2xf64>
// LAST:     return %[[V_8]] : tensor<2xf64>

// TUPLE-LABEL: func.func @convert_f32_to_f64_expanded
// TUPLE:     %[[V_0:.*]] = stablehlo.convert %arg0 : (tensor<2xf32>) -> tensor<2xbf16>
// TUPLE:     %[[V_1:.*]] = stablehlo.convert %[[V_0]] : (tensor<2xbf16>) -> tensor<2xf32>
// TUPLE:     %[[V_2:.*]] = stablehlo.subtract %arg0, %[[V_1]] : tensor<2xf32>
// TUPLE:     %[[V_3:.*]] = stablehlo.convert %[[V_2]] : (tensor<2xf32>) -> tensor<2xbf16>
// TUPLE:     %[[V_4:.*]] = stablehlo.tuple %[[V_0]], %[[V_3]] : tuple<tensor<2xbf16>, tensor<2xbf16>>
// TUPLE:     %[[V_5:.*]] = stablehlo.get_tuple_element %[[V_4]][0] : (tuple<tensor<2xbf16>, tensor<2xbf16>>) -> tensor<2xbf16>
// TUPLE:     %[[V_6:.*]] = stablehlo.convert %[[V_5]] : (tensor<2xbf16>) -> tensor<2xf64>
// TUPLE:     %[[V_7:.*]] = stablehlo.get_tuple_element %[[V_4]][1] : (tuple<tensor<2xbf16>, tensor<2xbf16>>) -> tensor<2xbf16>
// TUPLE:     %[[V_8:.*]] = stablehlo.convert %[[V_7]] : (tensor<2xbf16>) -> tensor<2xf64>
// TUPLE:     %[[V_9:.*]] = stablehlo.add %[[V_6]], %[[V_8]] : tensor<2xf64>
// TUPLE:     return %[[V_9]] : tensor<2xf64>

func.func @main() attributes {enzyme.no_multifloat} {
  %cst = stablehlo.constant dense<[1.1, 2.2]> : tensor<2xf32>
  
  %expected = stablehlo.constant dense<[1.1, 2.2]> : tensor<2xf64>
  
  %res = func.call @convert_f32_to_f64_expanded(%cst) : (tensor<2xf32>) -> tensor<2xf64>
  
  "check.expect_almost_eq"(%res, %expected) : (tensor<2xf64>, tensor<2xf64>) -> ()
  return
}

// FIRST-LABEL: func.func @main
// FIRST:     %[[CST:.*]] = stablehlo.constant dense<{{[^>]*}}> : tensor<2xf32>
// FIRST:     %[[CST_0:.*]] = stablehlo.constant dense<{{[^>]*}}> : tensor<2xf64>
// FIRST:     %[[V_0:.*]] = call @convert_f32_to_f64_expanded(%[[CST]]) : (tensor<2xf32>) -> tensor<2xf64>
// FIRST:     check.expect_almost_eq %[[V_0]], %[[CST_0]] : tensor<2xf64>
// FIRST:     return
