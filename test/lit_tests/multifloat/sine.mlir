// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=first" %s | FileCheck --check-prefix=FIRST %s
// RUN: enzymexlamlir-opt %s --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=first" | stablehlo-translate - --interpret --allow-unregistered-dialect

func.func @sine_test(%arg0: tensor<15xf64>) -> tensor<15xf64> {
  %0 = stablehlo.sine %arg0 : tensor<15xf64>
  return %0 : tensor<15xf64>
}

func.func @sine(%arg0: tensor<2xf64>) -> tensor<2xf64> {
  %0 = stablehlo.sine %arg0 : tensor<2xf64>
  return %0 : tensor<2xf64>
}

func.func @main() attributes {enzyme.no_multifloat} {
  %cst = stablehlo.constant dense<[0.0, 1.1, 1.5707963267948966, 2.2, 3.141592653589793, 4.4, 4.71238898038469, 5.5, 6.283185307179586, -1.1, -2.2, -4.4, -5.5, 7.383185307179586, 8.483185307179586]> : tensor<15xf64>
  %expected_mf = stablehlo.constant dense<[0.0, 0.8912073600614363, 1.0, 0.8084964038195892, 5.580589596813817e-15, -0.9516020738895179, -1.0, -0.7055403255704036, -1.1161179193627634e-14, -0.8912073600614363, -0.8084964038195892, 0.9516020738895179, 0.7055403255704036, 0.8912073600614292, 0.8084964038196087]> : tensor<15xf64>
  
  %res = func.call @sine_test(%cst) : (tensor<15xf64>) -> tensor<15xf64>
  
  // Strict test against Julia MultiFloat
  %diff = stablehlo.subtract %res, %expected_mf : tensor<15xf64>
  %abs_diff = stablehlo.abs %diff : tensor<15xf64>
  %zero = stablehlo.constant dense<0.0> : tensor<15xf64>
  "check.expect_almost_eq"(%abs_diff, %zero) {tolerance = 1.0e-13 : f64} : (tensor<15xf64>, tensor<15xf64>) -> ()
  
  // Approximate test against regular f64
  %expected_f64 = stablehlo.constant dense<[0.0, 0.8912073600614354, 1.0, 0.8084964038195901, 1.2246467991473532e-16, -0.951602073889516, -1.0, -0.7055403255703919, -2.4492935982947064e-16, -0.8912073600614354, -0.8084964038195901, 0.951602073889516, 0.7055403255703919, 0.8912073600614351, 0.8084964038195908]> : tensor<15xf64>
  %diff_f64 = stablehlo.subtract %res, %expected_f64 : tensor<15xf64>
  %abs_diff_f64 = stablehlo.abs %diff_f64 : tensor<15xf64>
  "check.expect_almost_eq"(%abs_diff_f64, %zero) {tolerance = 1.0e-13 : f64} : (tensor<15xf64>, tensor<15xf64>) -> ()
  return
}

// FIRST-LABEL: func.func @main
// FIRST:     %[[CST:.*]] = stablehlo.constant dense<{{[^>]*}}> : tensor<15xf64>
// FIRST:     %[[CST_0:.*]] = stablehlo.constant dense<{{[^>]*}}> : tensor<15xf64>
// FIRST:     %[[V_0:.*]] = call @sine_test(%[[CST]]) : (tensor<15xf64>) -> tensor<15xf64>
// FIRST:     %[[V_1:.*]] = stablehlo.subtract %[[V_0]], %[[CST_0]] : tensor<15xf64>
// FIRST:     %[[V_2:.*]] = stablehlo.abs %[[V_1]] : tensor<15xf64>
// FIRST:     %[[CST_1:.*]] = stablehlo.constant dense<{{[^>]*}}> : tensor<15xf64>
// FIRST:     check.expect_almost_eq %[[V_2]], %[[CST_1]], tolerance = 1.000000e-13 : tensor<15xf64>
// FIRST:     %[[CST_2:.*]] = stablehlo.constant dense<{{[^>]*}}> : tensor<15xf64>
// FIRST:     %[[V_3:.*]] = stablehlo.subtract %[[V_0]], %[[CST_2]] : tensor<15xf64>
// FIRST:     %[[V_4:.*]] = stablehlo.abs %[[V_3]] : tensor<15xf64>
// FIRST:     check.expect_almost_eq %[[V_4]], %[[CST_1]], tolerance = 1.000000e-13 : tensor<15xf64>
// FIRST:     return
