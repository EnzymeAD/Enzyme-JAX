// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=first" %s | FileCheck --check-prefix=FIRST %s
// RUN: enzymexlamlir-opt %s --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=first" | stablehlo-translate - --interpret --allow-unregistered-dialect

func.func @giant_splat() -> tensor<1000x1000xf64> {
  %0 = stablehlo.constant dense<1.100000e+00> : tensor<1000x1000xf64>
  return %0 : tensor<1000x1000xf64>
}

func.func @non_splat() -> tensor<2xf64> {
  %0 = stablehlo.constant dense<[1.100000e+00, 2.200000e+00]> : tensor<2xf64>
  return %0 : tensor<2xf64>
}
// FIRST-LABEL: func.func @giant_splat
// FIRST:     %[[CST:.*]] = stablehlo.constant dense<1.100000e+00> : tensor<1x1000x1000xf32>
// FIRST:     %[[CST_0:.*]] = stablehlo.constant dense<-2.38418583E-8> : tensor<1x1000x1000xf32>
// FIRST:     %[[V_0:.*]] = stablehlo.concatenate %[[CST]], %[[CST_0]], dim = 0 : (tensor<1x1000x1000xf32>, tensor<1x1000x1000xf32>) -> tensor<2x1000x1000xf32>
// FIRST:     %[[V_1:.*]] = stablehlo.convert %[[V_0]] : (tensor<2x1000x1000xf32>) -> tensor<2x1000x1000xf64>
// FIRST:     %[[CST_1:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// FIRST:     %[[V_2:.*]] = stablehlo.reduce(%[[V_1]] init: %[[CST_1]]) applies stablehlo.add across dimensions = [0] : (tensor<2x1000x1000xf64>, tensor<f64>) -> tensor<1000x1000xf64>
// FIRST:     return %[[V_2]] : tensor<1000x1000xf64>
// FIRST-LABEL: func.func @non_splat
// FIRST:     %[[CST:.*]] = stablehlo.constant dense<{{\[\[}}1.100000e+00, 2.200000e+00{{\]\]}}> : tensor<1x2xf32>
// FIRST:     %[[CST_0:.*]] = stablehlo.constant dense<{{\[\[}}-2.38418583E-8, -4.76837165E-8{{\]\]}}> : tensor<1x2xf32>
// FIRST:     %[[V_0:.*]] = stablehlo.concatenate %[[CST]], %[[CST_0]], dim = 0 : (tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<2x2xf32>
// FIRST:     %[[V_1:.*]] = stablehlo.convert %[[V_0]] : (tensor<2x2xf32>) -> tensor<2x2xf64>
// FIRST:     %[[CST_1:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// FIRST:     %[[V_2:.*]] = stablehlo.reduce(%[[V_1]] init: %[[CST_1]]) applies stablehlo.add across dimensions = [0] : (tensor<2x2xf64>, tensor<f64>) -> tensor<2xf64>
// FIRST:     return %[[V_2]] : tensor<2xf64>

// FIRST-LABEL: func.func @main

func.func @main() attributes {enzyme.no_multifloat} {
  %expected = stablehlo.constant dense<[1.0999999999999996, 2.1999999999999993]> : tensor<2xf64>
  
  %res = func.call @non_splat() : () -> tensor<2xf64>
  
  // Strict test against Julia MultiFloat
  "check.expect_close"(%res, %expected) {max_ulp_difference = 0 : ui64} : (tensor<2xf64>, tensor<2xf64>) -> ()
  
  // Approximate test against regular f64
  %expected_f64 = stablehlo.constant dense<[1.100000e+00, 2.200000e+00]> : tensor<2xf64>
  "check.expect_close"(%res, %expected_f64) {max_ulp_difference = 10 : ui64} : (tensor<2xf64>, tensor<2xf64>) -> ()
  return
}
// FIRST:     %[[CST_MAIN:.*]] = stablehlo.constant dense<{{[^>]*}}> : tensor<2xf64>
// FIRST:     %[[V_MAIN_0:.*]] = call @non_splat() : () -> tensor<2xf64>
// FIRST:     check.expect_close %[[V_MAIN_0]], %[[CST_MAIN]], max_ulp_difference = 0 : tensor<2xf64>, tensor<2xf64>
// FIRST:     %[[CST_MAIN_0:.*]] = stablehlo.constant dense<{{[^>]*}}> : tensor<2xf64>
// FIRST:     check.expect_close %[[V_MAIN_0]], %[[CST_MAIN_0]], max_ulp_difference = 10 : tensor<2xf64>, tensor<2xf64>
// FIRST:     return
