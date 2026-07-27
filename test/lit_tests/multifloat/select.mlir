// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=first" %s | FileCheck --check-prefix=FIRST %s
// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=last" %s | FileCheck --check-prefix=LAST %s
// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=tuple" %s | FileCheck --check-prefix=TUPLE %s
// RUN: enzymexlamlir-opt %s --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=first" | stablehlo-translate - --interpret --allow-unregistered-dialect
// RUN: enzymexlamlir-opt %s --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=last" | stablehlo-translate - --interpret --allow-unregistered-dialect
// RUN: enzymexlamlir-opt %s --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=tuple" | stablehlo-translate - --interpret --allow-unregistered-dialect

func.func @test_select(%pred: tensor<2xi1>, %on_true: tensor<2xf64>, %on_false: tensor<2xf64>) -> tensor<2xf64> {
  // FIRST-LABEL: func.func @test_select
  // FIRST: %[[C0:.*]] = stablehlo.convert %arg1 : (tensor<2xf64>) -> tensor<2xf32>
  // FIRST: %[[C1:.*]] = stablehlo.convert %[[C0]] : (tensor<2xf32>) -> tensor<2xf64>
  // FIRST: %[[SUB1:.*]] = stablehlo.subtract %arg1, %[[C1]] : tensor<2xf64>
  // FIRST: %[[C2:.*]] = stablehlo.convert %[[SUB1]] : (tensor<2xf64>) -> tensor<2xf32>
  // FIRST: %[[R1:.*]] = stablehlo.reshape %[[C0]] : (tensor<2xf32>) -> tensor<1x2xf32>
  // FIRST: %[[R2:.*]] = stablehlo.reshape %[[C2]] : (tensor<2xf32>) -> tensor<1x2xf32>
  // FIRST: %[[CAT1:.*]] = stablehlo.concatenate %[[R1]], %[[R2]], dim = 0 : (tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<2x2xf32>
  // FIRST: %[[C3:.*]] = stablehlo.convert %arg2 : (tensor<2xf64>) -> tensor<2xf32>
  // FIRST: %[[C4:.*]] = stablehlo.convert %[[C3]] : (tensor<2xf32>) -> tensor<2xf64>
  // FIRST: %[[SUB2:.*]] = stablehlo.subtract %arg2, %[[C4]] : tensor<2xf64>
  // FIRST: %[[C5:.*]] = stablehlo.convert %[[SUB2]] : (tensor<2xf64>) -> tensor<2xf32>
  // FIRST: %[[R3:.*]] = stablehlo.reshape %[[C3]] : (tensor<2xf32>) -> tensor<1x2xf32>
  // FIRST: %[[R4:.*]] = stablehlo.reshape %[[C5]] : (tensor<2xf32>) -> tensor<1x2xf32>
  // FIRST: %[[CAT2:.*]] = stablehlo.concatenate %[[R3]], %[[R4]], dim = 0 : (tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<2x2xf32>
  // FIRST: %[[PRED_BCAST:.*]] = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<2xi1>) -> tensor<2x2xi1>
  // FIRST: %[[SELECT:.*]] = stablehlo.select %[[PRED_BCAST]], %[[CAT1]], %[[CAT2]] : tensor<2x2xi1>, tensor<2x2xf32>
  // FIRST: %[[C6:.*]] = stablehlo.convert %[[SELECT]] : (tensor<2x2xf32>) -> tensor<2x2xf64>
  // FIRST: %[[ZERO:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
  // FIRST: %[[RES:.*]] = stablehlo.reduce(%[[C6]] init: %[[ZERO]]) applies stablehlo.add across dimensions = [0] : (tensor<2x2xf64>, tensor<f64>) -> tensor<2xf64>
  // FIRST: return %[[RES]] : tensor<2xf64>

  // LAST-LABEL: func.func @test_select
  // LAST: %[[TRUE_CONV:.*]] = stablehlo.concatenate %{{.*}}, %{{.*}}, dim = 1 : (tensor<2x1xf32>, tensor<2x1xf32>) -> tensor<2x2xf32>
  // LAST: %[[FALSE_CONV:.*]] = stablehlo.concatenate %{{.*}}, %{{.*}}, dim = 1 : (tensor<2x1xf32>, tensor<2x1xf32>) -> tensor<2x2xf32>
  // LAST: %[[PRED_BCAST:.*]] = stablehlo.broadcast_in_dim %arg0, dims = [0] : (tensor<2xi1>) -> tensor<2x2xi1>
  // LAST: %[[SELECT:.*]] = stablehlo.select %[[PRED_BCAST]], %[[TRUE_CONV]], %[[FALSE_CONV]] : tensor<2x2xi1>, tensor<2x2xf32>
  // LAST: %[[CST:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
  // LAST: %[[OUT:.*]] = stablehlo.reduce(%{{.*}} init: %[[CST]]) applies stablehlo.add across dimensions = [1] : (tensor<2x2xf64>, tensor<f64>) -> tensor<2xf64>
  // LAST: return %[[OUT]] : tensor<2xf64>

  // TUPLE-LABEL: func.func @test_select
  // TUPLE: %[[TUPLE:.*]] = stablehlo.tuple %{{.*}}, %{{.*}} : tuple<tensor<2xf32>, tensor<2xf32>>
  // TUPLE: %[[FALSE_TUPLE:.*]] = stablehlo.tuple %{{.*}}, %{{.*}} : tuple<tensor<2xf32>, tensor<2xf32>>
  // TUPLE: %[[TRUE_HIGH:.*]] = stablehlo.get_tuple_element %[[TUPLE]][0] : (tuple<tensor<2xf32>, tensor<2xf32>>) -> tensor<2xf32>
  // TUPLE: %[[TRUE_LOW:.*]] = stablehlo.get_tuple_element %[[TUPLE]][1] : (tuple<tensor<2xf32>, tensor<2xf32>>) -> tensor<2xf32>
  // TUPLE: %[[FALSE_HIGH:.*]] = stablehlo.get_tuple_element %[[FALSE_TUPLE]][0] : (tuple<tensor<2xf32>, tensor<2xf32>>) -> tensor<2xf32>
  // TUPLE: %[[FALSE_LOW:.*]] = stablehlo.get_tuple_element %[[FALSE_TUPLE]][1] : (tuple<tensor<2xf32>, tensor<2xf32>>) -> tensor<2xf32>
  // TUPLE: %[[SELECT_HIGH:.*]] = stablehlo.select %arg0, %[[TRUE_HIGH]], %[[FALSE_HIGH]] : tensor<2xi1>, tensor<2xf32>
  // TUPLE: %[[SELECT_LOW:.*]] = stablehlo.select %arg0, %[[TRUE_LOW]], %[[FALSE_LOW]] : tensor<2xi1>, tensor<2xf32>
  // TUPLE: %[[PACKED:.*]] = stablehlo.tuple %[[SELECT_HIGH]], %[[SELECT_LOW]] : tuple<tensor<2xf32>, tensor<2xf32>>
  // TUPLE: %[[CONV1:.*]] = stablehlo.convert %{{.*}} : (tensor<2xf32>) -> tensor<2xf64>
  // TUPLE: %[[CONV2:.*]] = stablehlo.convert %{{.*}} : (tensor<2xf32>) -> tensor<2xf64>
  // TUPLE: %[[OUT:.*]] = stablehlo.add %[[CONV1]], %[[CONV2]] : tensor<2xf64>
  // TUPLE: return %[[OUT]] : tensor<2xf64>

  %0 = stablehlo.select %pred, %on_true, %on_false : tensor<2xi1>, tensor<2xf64>
  return %0 : tensor<2xf64>
}

func.func @main() attributes {enzyme.no_multifloat} {
  %pred = stablehlo.constant dense<[true, false]> : tensor<2xi1>
  %on_true = stablehlo.constant dense<[1.1, 2.2]> : tensor<2xf64>
  %on_false = stablehlo.constant dense<[3.3, 4.4]> : tensor<2xf64>
  
  %expected_mf = stablehlo.constant dense<[1.0999999999999996, 4.399999999999999]> : tensor<2xf64>
  
  %res = func.call @test_select(%pred, %on_true, %on_false) : (tensor<2xi1>, tensor<2xf64>, tensor<2xf64>) -> tensor<2xf64>
  
  // Strict test against Julia MultiFloat
  "check.expect_close"(%res, %expected_mf) {max_ulp_difference = 3 : ui64} : (tensor<2xf64>, tensor<2xf64>) -> ()
  
  // Approximate test against regular f64
  %expected_f64 = stablehlo.constant dense<[1.1, 4.4]> : tensor<2xf64>
  "check.expect_close"(%res, %expected_f64) {max_ulp_difference = 3 : ui64} : (tensor<2xf64>, tensor<2xf64>) -> ()
  return
}
