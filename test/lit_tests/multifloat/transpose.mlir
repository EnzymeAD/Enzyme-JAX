// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=first" %s | FileCheck --check-prefix=FIRST %s
// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=last" %s | FileCheck --check-prefix=LAST %s
// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=tuple" %s | FileCheck --check-prefix=TUPLE %s
// RUN: enzymexlamlir-opt %s --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=first" | stablehlo-translate - --interpret --allow-unregistered-dialect
// RUN: enzymexlamlir-opt %s --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=last" | stablehlo-translate - --interpret --allow-unregistered-dialect
// RUN: enzymexlamlir-opt %s --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=tuple" | stablehlo-translate - --interpret --allow-unregistered-dialect

func.func @test_transpose(%arg0: tensor<2x3xf64>) -> tensor<3x2xf64> {
  // FIRST-LABEL: func.func @test_transpose
  // FIRST: %[[A:.*]] = stablehlo.concatenate %{{.*}}, %{{.*}}, dim = 0 : (tensor<1x2x3xf32>, tensor<1x2x3xf32>) -> tensor<2x2x3xf32>
  // FIRST: %[[TRANS:.*]] = stablehlo.transpose %[[A]], dims = [0, 2, 1] : (tensor<2x2x3xf32>) -> tensor<2x3x2xf32>
  // FIRST: %[[CST:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
  // FIRST: %[[OUT:.*]] = stablehlo.reduce(%{{.*}} init: %[[CST]]) applies stablehlo.add across dimensions = [0] : (tensor<2x3x2xf64>, tensor<f64>) -> tensor<3x2xf64>
  // FIRST: return %[[OUT]] : tensor<3x2xf64>

  // LAST-LABEL: func.func @test_transpose
  // LAST: %[[A:.*]] = stablehlo.concatenate %{{.*}}, %{{.*}}, dim = 2 : (tensor<2x3x1xf32>, tensor<2x3x1xf32>) -> tensor<2x3x2xf32>
  // LAST: %[[TRANS:.*]] = stablehlo.transpose %[[A]], dims = [1, 0, 2] : (tensor<2x3x2xf32>) -> tensor<3x2x2xf32>
  // LAST: %[[CST:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
  // LAST: %[[OUT:.*]] = stablehlo.reduce(%{{.*}} init: %[[CST]]) applies stablehlo.add across dimensions = [2] : (tensor<3x2x2xf64>, tensor<f64>) -> tensor<3x2xf64>
  // LAST: return %[[OUT]] : tensor<3x2xf64>

  // TUPLE-LABEL: func.func @test_transpose
  // TUPLE: %[[TUPLE:.*]] = stablehlo.tuple %{{.*}}, %{{.*}} : tuple<tensor<2x3xf32>, tensor<2x3xf32>>
  // TUPLE: %[[HIGH:.*]] = stablehlo.get_tuple_element %[[TUPLE]][0] : (tuple<tensor<2x3xf32>, tensor<2x3xf32>>) -> tensor<2x3xf32>
  // TUPLE: %[[LOW:.*]] = stablehlo.get_tuple_element %[[TUPLE]][1] : (tuple<tensor<2x3xf32>, tensor<2x3xf32>>) -> tensor<2x3xf32>
  // TUPLE: %[[TRANS_HIGH:.*]] = stablehlo.transpose %[[HIGH]], dims = [1, 0] : (tensor<2x3xf32>) -> tensor<3x2xf32>
  // TUPLE: %[[TRANS_LOW:.*]] = stablehlo.transpose %[[LOW]], dims = [1, 0] : (tensor<2x3xf32>) -> tensor<3x2xf32>
  // TUPLE: %[[CONV1:.*]] = stablehlo.convert %[[TRANS_HIGH]] : (tensor<3x2xf32>) -> tensor<3x2xf64>
  // TUPLE: %[[CONV2:.*]] = stablehlo.convert %[[TRANS_LOW]] : (tensor<3x2xf32>) -> tensor<3x2xf64>
  // TUPLE: %[[OUT:.*]] = stablehlo.add %[[CONV1]], %[[CONV2]] : tensor<3x2xf64>
  // TUPLE: return %[[OUT]] : tensor<3x2xf64>
  
  %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<2x3xf64>) -> tensor<3x2xf64>
  return %0 : tensor<3x2xf64>
}

func.func @main() attributes {enzyme.no_multifloat} {
  %cst = stablehlo.constant dense<[[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]]> : tensor<2x3xf64>
  
  %expected_mf = stablehlo.constant dense<[[1.0999999999999996, 4.399999999999999],
                                           [2.1999999999999993, 5.5],
                                           [3.3000000000000007, 6.600000000000001]]> : tensor<3x2xf64>
  
  %res = func.call @test_transpose(%cst) : (tensor<2x3xf64>) -> tensor<3x2xf64>
  
  // Strict test against Julia MultiFloat
  "check.expect_close"(%res, %expected_mf) {max_ulp_difference = 3 : ui64} : (tensor<3x2xf64>, tensor<3x2xf64>) -> ()
  
  // Approximate test against regular f64
  %expected_f64 = stablehlo.constant dense<[[1.1, 4.4],
                                           [2.2, 5.5],
                                           [3.3, 6.6]]> : tensor<3x2xf64>
  "check.expect_close"(%res, %expected_f64) {max_ulp_difference = 3 : ui64} : (tensor<3x2xf64>, tensor<3x2xf64>) -> ()
  return
}
