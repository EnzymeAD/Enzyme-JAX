// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=first" %s | FileCheck --check-prefix=FIRST %s
// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=last" %s | FileCheck --check-prefix=LAST %s
// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=tuple" %s | FileCheck --check-prefix=TUPLE %s
// RUN: enzymexlamlir-opt %s --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=first" | stablehlo-translate - --interpret --allow-unregistered-dialect
// RUN: enzymexlamlir-opt %s --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=last" | stablehlo-translate - --interpret --allow-unregistered-dialect
// RUN: enzymexlamlir-opt %s --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=tuple" | stablehlo-translate - --interpret --allow-unregistered-dialect

func.func @test_slice(%arg0: tensor<5xf64>) -> tensor<3xf64> {
  // FIRST-LABEL: func.func @test_slice
  // FIRST: %[[A:.*]] = stablehlo.concatenate %{{.*}}, %{{.*}}, dim = 0 : (tensor<1x5xf32>, tensor<1x5xf32>) -> tensor<2x5xf32>
  // FIRST: %[[SLICE:.*]] = stablehlo.slice %[[A]] [0:2, 1:4] : (tensor<2x5xf32>) -> tensor<2x3xf32>
  // FIRST: %[[CST:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
  // FIRST: %[[OUT:.*]] = stablehlo.reduce(%{{.*}} init: %[[CST]]) applies stablehlo.add across dimensions = [0] : (tensor<2x3xf64>, tensor<f64>) -> tensor<3xf64>
  // FIRST: return %[[OUT]] : tensor<3xf64>

  // LAST-LABEL: func.func @test_slice
  // LAST: %[[A:.*]] = stablehlo.concatenate %{{.*}}, %{{.*}}, dim = 1 : (tensor<5x1xf32>, tensor<5x1xf32>) -> tensor<5x2xf32>
  // LAST: %[[SLICE:.*]] = stablehlo.slice %[[A]] [1:4, 0:2] : (tensor<5x2xf32>) -> tensor<3x2xf32>
  // LAST: %[[CST:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
  // LAST: %[[OUT:.*]] = stablehlo.reduce(%{{.*}} init: %[[CST]]) applies stablehlo.add across dimensions = [1] : (tensor<3x2xf64>, tensor<f64>) -> tensor<3xf64>
  // LAST: return %[[OUT]] : tensor<3xf64>

  // TUPLE-LABEL: func.func @test_slice
  // TUPLE: %[[TUPLE:.*]] = stablehlo.tuple %{{.*}}, %{{.*}} : tuple<tensor<5xf32>, tensor<5xf32>>
  // TUPLE: %[[HIGH:.*]] = stablehlo.get_tuple_element %[[TUPLE]][0] : (tuple<tensor<5xf32>, tensor<5xf32>>) -> tensor<5xf32>
  // TUPLE: %[[LOW:.*]] = stablehlo.get_tuple_element %[[TUPLE]][1] : (tuple<tensor<5xf32>, tensor<5xf32>>) -> tensor<5xf32>
  // TUPLE: %[[SLICE_HIGH:.*]] = stablehlo.slice %[[HIGH]] [1:4] : (tensor<5xf32>) -> tensor<3xf32>
  // TUPLE: %[[SLICE_LOW:.*]] = stablehlo.slice %[[LOW]] [1:4] : (tensor<5xf32>) -> tensor<3xf32>
  // TUPLE: %[[CONV1:.*]] = stablehlo.convert %[[SLICE_HIGH]] : (tensor<3xf32>) -> tensor<3xf64>
  // TUPLE: %[[CONV2:.*]] = stablehlo.convert %[[SLICE_LOW]] : (tensor<3xf32>) -> tensor<3xf64>
  // TUPLE: %[[OUT:.*]] = stablehlo.add %[[CONV1]], %[[CONV2]] : tensor<3xf64>
  // TUPLE: return %[[OUT]] : tensor<3xf64>

  %0 = stablehlo.slice %arg0 [1:4] : (tensor<5xf64>) -> tensor<3xf64>
  return %0 : tensor<3xf64>
}

func.func @main() attributes {enzyme.no_multifloat} {
  %cst = stablehlo.constant dense<[1.1, 2.2, 3.3, 4.4, 5.5]> : tensor<5xf64>
  
  %expected_mf = stablehlo.constant dense<[2.1999999999999993, 3.3000000000000007, 4.399999999999999]> : tensor<3xf64>
  
  %res = func.call @test_slice(%cst) : (tensor<5xf64>) -> tensor<3xf64>
  
  // Strict test against Julia MultiFloat
  "check.expect_close"(%res, %expected_mf) {max_ulp_difference = 3 : ui64} : (tensor<3xf64>, tensor<3xf64>) -> ()
  
  // Approximate test against regular f64
  %expected_f64 = stablehlo.constant dense<[2.2, 3.3, 4.4]> : tensor<3xf64>
  "check.expect_close"(%res, %expected_f64) {max_ulp_difference = 3 : ui64} : (tensor<3xf64>, tensor<3xf64>) -> ()
  return
}
