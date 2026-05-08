// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f16 concat-dimension=first expansion-size=2" %s | FileCheck --check-prefix=FIRST %s
// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f16 concat-dimension=last expansion-size=2" %s | FileCheck --check-prefix=LAST %s
// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f16 concat-dimension=tuple expansion-size=2" %s | FileCheck --check-prefix=TUPLE %s
// RUN: enzymexlamlir-opt %s --multi-float-conversion="source-type=f64 target-type=f16 concat-dimension=first" | stablehlo-translate - --interpret --allow-unregistered-dialect

func.func @test_convert(%arg0: tensor<4x4xf32>) -> tensor<4x4xf64> {
  %0 = stablehlo.convert %arg0 : (tensor<4x4xf32>) -> tensor<4x4xf64>
  return %0 : tensor<4x4xf64>
}

// TUPLE-LABEL: func.func @test_convert(
// TUPLE:     %[[V_0:.*]] = stablehlo.convert %arg0 : (tensor<4x4xf32>) -> tensor<4x4xf16>
// TUPLE:     %[[V_1:.*]] = stablehlo.convert %[[V_0]] : (tensor<4x4xf16>) -> tensor<4x4xf32>
// TUPLE:     %[[V_2:.*]] = stablehlo.subtract %arg0, %[[V_1]] : tensor<4x4xf32>
// TUPLE:     %[[V_3:.*]] = stablehlo.convert %[[V_2]] : (tensor<4x4xf32>) -> tensor<4x4xf16>
// TUPLE:     %[[V_4:.*]] = stablehlo.tuple %[[V_0]], %[[V_3]] : tuple<tensor<4x4xf16>, tensor<4x4xf16>>
// TUPLE:     %[[V_5:.*]] = stablehlo.convert %[[V_0]] : (tensor<4x4xf16>) -> tensor<4x4xf64>
// TUPLE:     %[[V_6:.*]] = stablehlo.convert %[[V_3]] : (tensor<4x4xf16>) -> tensor<4x4xf64>
// TUPLE:     %[[V_7:.*]] = stablehlo.add %[[V_5]], %[[V_6]] : tensor<4x4xf64>
// TUPLE:     return %[[V_7]] : tensor<4x4xf64>

// FIRST-LABEL: func.func @test_convert(
// FIRST:     %[[V_0:.*]] = stablehlo.convert %arg0 : (tensor<4x4xf32>) -> tensor<4x4xf16>
// FIRST:     %[[V_1:.*]] = stablehlo.convert %[[V_0]] : (tensor<4x4xf16>) -> tensor<4x4xf32>
// FIRST:     %[[V_2:.*]] = stablehlo.subtract %arg0, %[[V_1]] : tensor<4x4xf32>
// FIRST:     %[[V_3:.*]] = stablehlo.convert %[[V_2]] : (tensor<4x4xf32>) -> tensor<4x4xf16>
// FIRST:     %[[V_4:.*]] = stablehlo.reshape %[[V_0]] : (tensor<4x4xf16>) -> tensor<1x4x4xf16>
// FIRST:     %[[V_5:.*]] = stablehlo.reshape %[[V_3]] : (tensor<4x4xf16>) -> tensor<1x4x4xf16>
// FIRST:     %[[V_6:.*]] = stablehlo.concatenate %[[V_4]], %[[V_5]], dim = 0 : (tensor<1x4x4xf16>, tensor<1x4x4xf16>) -> tensor<2x4x4xf16>
// FIRST:     %[[V_7:.*]] = stablehlo.convert %[[V_6]] : (tensor<2x4x4xf16>) -> tensor<2x4x4xf64>
// FIRST:     %[[CST:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// FIRST:     %[[V_8:.*]] = stablehlo.reduce(%[[V_7]] init: %[[CST]]) applies stablehlo.add across dimensions = [0] : (tensor<2x4x4xf64>, tensor<f64>) -> tensor<4x4xf64>
// FIRST:     return %[[V_8]] : tensor<4x4xf64>

// LAST-LABEL: func.func @test_convert(
// LAST:     %[[V_0:.*]] = stablehlo.convert %arg0 : (tensor<4x4xf32>) -> tensor<4x4xf16>
// LAST:     %[[V_1:.*]] = stablehlo.convert %[[V_0]] : (tensor<4x4xf16>) -> tensor<4x4xf32>
// LAST:     %[[V_2:.*]] = stablehlo.subtract %arg0, %[[V_1]] : tensor<4x4xf32>
// LAST:     %[[V_3:.*]] = stablehlo.convert %[[V_2]] : (tensor<4x4xf32>) -> tensor<4x4xf16>
// LAST:     %[[V_4:.*]] = stablehlo.reshape %[[V_0]] : (tensor<4x4xf16>) -> tensor<4x4x1xf16>
// LAST:     %[[V_5:.*]] = stablehlo.reshape %[[V_3]] : (tensor<4x4xf16>) -> tensor<4x4x1xf16>
// LAST:     %[[V_6:.*]] = stablehlo.concatenate %[[V_4]], %[[V_5]], dim = 2 : (tensor<4x4x1xf16>, tensor<4x4x1xf16>) -> tensor<4x4x2xf16>
// LAST:     %[[V_7:.*]] = stablehlo.convert %[[V_6]] : (tensor<4x4x2xf16>) -> tensor<4x4x2xf64>
// LAST:     %[[CST:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// LAST:     %[[V_8:.*]] = stablehlo.reduce(%[[V_7]] init: %[[CST]]) applies stablehlo.add across dimensions = [2] : (tensor<4x4x2xf64>, tensor<f64>) -> tensor<4x4xf64>
// LAST:     return %[[V_8]] : tensor<4x4xf64>

// FIRST-LABEL: func.func @main

// LAST-LABEL: func.func @main

// TUPLE-LABEL: func.func @main

func.func @main() attributes {enzyme.no_multifloat} {
  %cst = stablehlo.constant dense<1.100000e+00> : tensor<4x4xf32>
  
  %expected_mf = stablehlo.constant dense<1.0999999046325684> : tensor<4x4xf64>
  
  %res = func.call @test_convert(%cst) : (tensor<4x4xf32>) -> tensor<4x4xf64>
  
  // Strict test against Julia MultiFloat
  "check.expect_close"(%res, %expected_mf) {max_ulp_difference = 3 : ui64} : (tensor<4x4xf64>, tensor<4x4xf64>) -> ()
  
  // Approximate test against regular f64
  %expected_f64 = stablehlo.constant dense<1.1000000238418579> : tensor<4x4xf64>
  %diff = stablehlo.subtract %res, %expected_f64 : tensor<4x4xf64>
  %abs_diff = stablehlo.abs %diff : tensor<4x4xf64>
  %zero = stablehlo.constant dense<0.0> : tensor<4x4xf64>
  "check.expect_almost_eq"(%abs_diff, %zero) {tolerance = 2.0e-7 : f64} : (tensor<4x4xf64>, tensor<4x4xf64>) -> ()
  return
}
// FIRST:     %[[CST:.*]] = stablehlo.constant dense<{{[^>]*}}> : tensor<4x4xf32>
// FIRST:     %[[CST_0:.*]] = stablehlo.constant dense<{{[^>]*}}> : tensor<4x4xf64>
// FIRST:     %[[V_0:.*]] = call @test_convert(%[[CST]]) : (tensor<4x4xf32>) -> tensor<4x4xf64>
// FIRST:     check.expect_close %[[V_0]], %[[CST_0]], max_ulp_difference = 3 : tensor<4x4xf64>, tensor<4x4xf64>
// FIRST:     %[[CST_1:.*]] = stablehlo.constant dense<{{[^>]*}}> : tensor<4x4xf64>
// FIRST:     %[[V_1:.*]] = stablehlo.subtract %[[V_0]], %[[CST_1]] : tensor<4x4xf64>
// FIRST:     %[[V_2:.*]] = stablehlo.abs %[[V_1]] : tensor<4x4xf64>
// FIRST:     %[[CST_2:.*]] = stablehlo.constant dense<{{[^>]*}}> : tensor<4x4xf64>
// FIRST:     check.expect_almost_eq %[[V_2]], %[[CST_2]], tolerance = 2.000000e-07 : tensor<4x4xf64>
// FIRST:     return
