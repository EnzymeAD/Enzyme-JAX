// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=tuple convert-signatures=false" %s | FileCheck --check-prefix=TUPLE %s
// RUN: enzymexlamlir-opt %s --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=tuple convert-signatures=false" | stablehlo-translate - --interpret --allow-unregistered-dialect

module {
  func.func @test_while_mixed(%arg0: tensor<i1>, %arg1: tensor<i64>, %arg2: tensor<f64>) -> (tensor<i64>, tensor<f64>) {
// TUPLE-LABEL: func.func @test_while_mixed
// TUPLE:     %[[V_0:.*]] = stablehlo.convert %arg2 : (tensor<f64>) -> tensor<f32>
// TUPLE:     %[[V_1:.*]] = stablehlo.convert %[[V_0]] : (tensor<f32>) -> tensor<f64>
// TUPLE:     %[[V_2:.*]] = stablehlo.subtract %arg2, %[[V_1]] : tensor<f64>
// TUPLE:     %[[V_3:.*]] = stablehlo.convert %[[V_2]] : (tensor<f64>) -> tensor<f32>
// TUPLE:     %[[V_4:.*]] = stablehlo.tuple %[[V_0]], %[[V_3]] : tuple<tensor<f32>, tensor<f32>>
// TUPLE:     %[[V_5:.*]] = stablehlo.get_tuple_element %[[V_4]][0] : (tuple<tensor<f32>, tensor<f32>>) -> tensor<f32>
// TUPLE:     %[[V_6:.*]] = stablehlo.get_tuple_element %[[V_4]][1] : (tuple<tensor<f32>, tensor<f32>>) -> tensor<f32>
// TUPLE:     %[[V_7:.*]]:3 = stablehlo.while(%iterArg = %arg1, %iterArg_0 = %[[V_5]], %iterArg_1 = %[[V_6]]) : tensor<i64>, tensor<f32>, tensor<f32>
// TUPLE:     cond {
// TUPLE:       stablehlo.return %arg0 : tensor<i1>
// TUPLE:       stablehlo.return %iterArg, %iterArg_0, %iterArg_1 : tensor<i64>, tensor<f32>, tensor<f32>
// TUPLE:     %[[V_8:.*]] = stablehlo.tuple %[[V_7]]#1, %[[V_7]]#2 : tuple<tensor<f32>, tensor<f32>>
// TUPLE:     %[[V_9:.*]] = stablehlo.convert %[[V_7]]#1 : (tensor<f32>) -> tensor<f64>
// TUPLE:     %[[V_10:.*]] = stablehlo.convert %[[V_7]]#2 : (tensor<f32>) -> tensor<f64>
// TUPLE:     %[[V_11:.*]] = stablehlo.add %[[V_9]], %[[V_10]] : tensor<f64>
// TUPLE:     return %[[V_7]]#0, %[[V_11]] : tensor<i64>, tensor<f64>

    %0:2 = stablehlo.while(%iter_counter = %arg1, %iter_val = %arg2) : tensor<i64>, tensor<f64>
      cond {
        stablehlo.return %arg0 : tensor<i1>
      } do {
        stablehlo.return %iter_counter, %iter_val : tensor<i64>, tensor<f64>
      }
    return %0#0, %0#1 : tensor<i64>, tensor<f64>
  }

  func.func @main() attributes {enzyme.no_multifloat} {
    %cond = stablehlo.constant dense<false> : tensor<i1>
    %counter = stablehlo.constant dense<42> : tensor<i64>
    %val = stablehlo.constant dense<1.100000e+00> : tensor<f64>
    
    %res:2 = func.call @test_while_mixed(%cond, %counter, %val) : (tensor<i1>, tensor<i64>, tensor<f64>) -> (tensor<i64>, tensor<f64>)
    
    %val_expected = stablehlo.constant dense<1.100000e+00> : tensor<f64>
    %res_c_f64 = stablehlo.convert %res#0 : (tensor<i64>) -> tensor<f64>
    %c_expected_f64 = stablehlo.constant dense<42.0> : tensor<f64>
    "check.expect_close"(%res_c_f64, %c_expected_f64) {max_ulp_difference = 0 : ui64} : (tensor<f64>, tensor<f64>) -> ()
    "check.expect_close"(%res#1, %val_expected) {max_ulp_difference = 3 : ui64} : (tensor<f64>, tensor<f64>) -> ()
    
    return
  }
}
// TUPLE:     %[[C:.*]] = stablehlo.constant dense<false> : tensor<i1>
// TUPLE:     %[[C_0:.*]] = stablehlo.constant dense<42> : tensor<i64>
// TUPLE:     %[[CST:.*]] = stablehlo.constant dense<1.100000e+00> : tensor<f64>
// TUPLE:     %[[V_0:.*]]:2 = call @test_while_mixed(%c, %[[C_0]], %cst) : (tensor<i1>, tensor<i64>, tensor<f64>) -> (tensor<i64>, tensor<f64>)
// TUPLE:     %[[CST_1:.*]] = stablehlo.constant dense<1.100000e+00> : tensor<f64>
// TUPLE:     %[[V_1:.*]] = stablehlo.convert %[[V_0]]#0 : (tensor<i64>) -> tensor<f64>
// TUPLE:     %[[CST_2:.*]] = stablehlo.constant dense<4.200000e+01> : tensor<f64>
// TUPLE:     check.expect_close %[[V_1]], %[[CST_2]], max_ulp_difference = 0 : tensor<f64>, tensor<f64>
// TUPLE:     check.expect_close %[[V_0]]#1, %[[CST_1]], max_ulp_difference = {{[0-9]+}} : tensor<f64>, tensor<f64>
// TUPLE:     return
