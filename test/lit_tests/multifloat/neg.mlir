// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=tuple" %s | FileCheck --check-prefix=TUPLE %s
// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=first" %s | FileCheck --check-prefix=FIRST %s
// RUN: enzymexlamlir-opt %s --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=first" | stablehlo-translate - --interpret --allow-unregistered-dialect

func.func @neg(%arg0: tensor<2xf64>) -> tensor<2xf64> {
  %0 = stablehlo.negate %arg0 : tensor<2xf64>
  return %0 : tensor<2xf64>
}

// TUPLE-LABEL: func.func @neg
// TUPLE: %[[C1:.*]] = stablehlo.convert %arg0 : (tensor<2xf64>) -> tensor<2xf32>
// TUPLE: %[[C2:.*]] = stablehlo.convert %[[C1]] : (tensor<2xf32>) -> tensor<2xf64>
// TUPLE: %[[SUB:.*]] = stablehlo.subtract %arg0, %[[C2]] : tensor<2xf64>
// TUPLE: %[[C3:.*]] = stablehlo.convert %[[SUB]] : (tensor<2xf64>) -> tensor<2xf32>
// TUPLE: %[[TUP:.*]] = stablehlo.tuple %[[C1]], %[[C3]] : tuple<tensor<2xf32>, tensor<2xf32>>
// TUPLE: %[[HI:.*]] = stablehlo.get_tuple_element %[[TUP]][0] : (tuple<tensor<2xf32>, tensor<2xf32>>) -> tensor<2xf32>
// TUPLE: %[[LO:.*]] = stablehlo.get_tuple_element %[[TUP]][1] : (tuple<tensor<2xf32>, tensor<2xf32>>) -> tensor<2xf32>
// TUPLE: %[[NEG_HI:.*]] = stablehlo.negate %[[HI]] : tensor<2xf32>
// TUPLE: %[[NEG_LO:.*]] = stablehlo.negate %[[LO]] : tensor<2xf32>
// TUPLE: %[[RES_TUP:.*]] = stablehlo.tuple %[[NEG_HI]], %[[NEG_LO]] : tuple<tensor<2xf32>, tensor<2xf32>>
// TUPLE: %[[C4:.*]] = stablehlo.convert %[[NEG_HI]] : (tensor<2xf32>) -> tensor<2xf64>
// TUPLE: %[[C5:.*]] = stablehlo.convert %[[NEG_LO]] : (tensor<2xf32>) -> tensor<2xf64>
// TUPLE: %[[ADD:.*]] = stablehlo.add %[[C4]], %[[C5]] : tensor<2xf64>
// TUPLE: return %[[ADD]] : tensor<2xf64>

// FIRST-LABEL: func.func @neg
// FIRST: %[[C1:.*]] = stablehlo.convert %arg0 : (tensor<2xf64>) -> tensor<2xf32>
// FIRST: %[[C2:.*]] = stablehlo.convert %[[C1]] : (tensor<2xf32>) -> tensor<2xf64>
// FIRST: %[[SUB:.*]] = stablehlo.subtract %arg0, %[[C2]] : tensor<2xf64>
// FIRST: %[[C3:.*]] = stablehlo.convert %[[SUB]] : (tensor<2xf64>) -> tensor<2xf32>
// FIRST: %[[R1:.*]] = stablehlo.reshape %[[C1]] : (tensor<2xf32>) -> tensor<1x2xf32>
// FIRST: %[[R2:.*]] = stablehlo.reshape %[[C3]] : (tensor<2xf32>) -> tensor<1x2xf32>
// FIRST: %[[CAT:.*]] = stablehlo.concatenate %[[R1]], %[[R2]], dim = 0 : (tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<2x2xf32>
// FIRST: %[[NEG:.*]] = stablehlo.negate %[[CAT]] : tensor<2x2xf32>
// FIRST: %[[C4:.*]] = stablehlo.convert %[[NEG]] : (tensor<2x2xf32>) -> tensor<2x2xf64>
// FIRST: %[[ZERO:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// FIRST: %[[RES:.*]] = stablehlo.reduce(%[[C4]] init: %[[ZERO]]) applies stablehlo.add across dimensions = [0] : (tensor<2x2xf64>, tensor<f64>) -> tensor<2xf64>
// FIRST: return %[[RES]] : tensor<2xf64>

func.func @main() attributes {enzyme.no_multifloat} {
  %cst = stablehlo.constant dense<[1.1, -2.2]> : tensor<2xf64>
  %expected = stablehlo.constant dense<[-1.1, 2.2]> : tensor<2xf64>
  %res = func.call @neg(%cst) : (tensor<2xf64>) -> tensor<2xf64>
  "check.expect_almost_eq"(%res, %expected) : (tensor<2xf64>, tensor<2xf64>) -> ()
  return
}

