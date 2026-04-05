// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=first" %s | FileCheck --check-prefix=FIRST %s
// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=last" %s | FileCheck --check-prefix=LAST %s
// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=tuple" %s | FileCheck --check-prefix=TUPLE %s
// RUN: enzymexlamlir-opt %s --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=first" | stablehlo-translate - --interpret --allow-unregistered-dialect
// RUN: enzymexlamlir-opt %s --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=last" | stablehlo-translate - --interpret --allow-unregistered-dialect
// RUN: enzymexlamlir-opt %s --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=tuple" | stablehlo-translate - --interpret --allow-unregistered-dialect

func.func @abs(%arg0: tensor<2xf64>) -> tensor<2xf64> {
  %0 = stablehlo.abs %arg0 : tensor<2xf64>
  return %0 : tensor<2xf64>
}

// FIRST-LABEL: func.func @abs
// FIRST: %[[C0:.*]] = stablehlo.convert %arg0 : (tensor<2xf64>) -> tensor<2xf32>
// FIRST: %[[C1:.*]] = stablehlo.convert %[[C0]] : (tensor<2xf32>) -> tensor<2xf64>
// FIRST: %[[SUB:.*]] = stablehlo.subtract %arg0, %[[C1]] : tensor<2xf64>
// FIRST: %[[C2:.*]] = stablehlo.convert %[[SUB]] : (tensor<2xf64>) -> tensor<2xf32>
// FIRST: %[[R1:.*]] = stablehlo.reshape %[[C0]] : (tensor<2xf32>) -> tensor<1x2xf32>
// FIRST: %[[R2:.*]] = stablehlo.reshape %[[C2]] : (tensor<2xf32>) -> tensor<1x2xf32>
// FIRST: %[[CAT:.*]] = stablehlo.concatenate %[[R1]], %[[R2]], dim = 0 : (tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<2x2xf32>
// FIRST: %[[X_HI:.*]] = stablehlo.slice %[[CAT]] [0:1, 0:2] : (tensor<2x2xf32>) -> tensor<1x2xf32>
// FIRST: %[[CST:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<1x2xf32>
// FIRST: %[[CMP:.*]] = stablehlo.compare GE, %[[X_HI]], %[[CST]] : (tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<1x2xi1>
// FIRST: %[[BCAST:.*]] = stablehlo.broadcast_in_dim %[[CMP]], dims = [0, 1] : (tensor<1x2xi1>) -> tensor<2x2xi1>
// FIRST: %[[NEG:.*]] = stablehlo.negate %[[CAT]] : tensor<2x2xf32>
// FIRST: %[[SEL:.*]] = stablehlo.select %[[BCAST]], %[[CAT]], %[[NEG]] : tensor<2x2xi1>, tensor<2x2xf32>
// FIRST: %[[CONV:.*]] = stablehlo.convert %[[SEL]] : (tensor<2x2xf32>) -> tensor<2x2xf64>
// FIRST: %[[ZERO:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// FIRST: %[[RES:.*]] = stablehlo.reduce(%[[CONV]] init: %[[ZERO]]) applies stablehlo.add across dimensions = [0] : (tensor<2x2xf64>, tensor<f64>) -> tensor<2xf64>
// FIRST: return %[[RES]] : tensor<2xf64>

// LAST-LABEL: func.func @abs
// LAST: %[[C0:.*]] = stablehlo.convert %arg0 : (tensor<2xf64>) -> tensor<2xf32>
// LAST: %[[C1:.*]] = stablehlo.convert %[[C0]] : (tensor<2xf32>) -> tensor<2xf64>
// LAST: %[[SUB:.*]] = stablehlo.subtract %arg0, %[[C1]] : tensor<2xf64>
// LAST: %[[C2:.*]] = stablehlo.convert %[[SUB]] : (tensor<2xf64>) -> tensor<2xf32>
// LAST: %[[R1:.*]] = stablehlo.reshape %[[C0]] : (tensor<2xf32>) -> tensor<2x1xf32>
// LAST: %[[R2:.*]] = stablehlo.reshape %[[C2]] : (tensor<2xf32>) -> tensor<2x1xf32>
// LAST: %[[CAT:.*]] = stablehlo.concatenate %[[R1]], %[[R2]], dim = 1 : (tensor<2x1xf32>, tensor<2x1xf32>) -> tensor<2x2xf32>
// LAST: %[[X_HI:.*]] = stablehlo.slice %[[CAT]] [0:2, 0:1] : (tensor<2x2xf32>) -> tensor<2x1xf32>
// LAST: %[[CST:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<2x1xf32>
// LAST: %[[CMP:.*]] = stablehlo.compare GE, %[[X_HI]], %[[CST]] : (tensor<2x1xf32>, tensor<2x1xf32>) -> tensor<2x1xi1>
// LAST: %[[BCAST:.*]] = stablehlo.broadcast_in_dim %[[CMP]], dims = [0, 1] : (tensor<2x1xi1>) -> tensor<2x2xi1>
// LAST: %[[NEG:.*]] = stablehlo.negate %[[CAT]] : tensor<2x2xf32>
// LAST: %[[SEL:.*]] = stablehlo.select %[[BCAST]], %[[CAT]], %[[NEG]] : tensor<2x2xi1>, tensor<2x2xf32>
// LAST: %[[CONV:.*]] = stablehlo.convert %[[SEL]] : (tensor<2x2xf32>) -> tensor<2x2xf64>
// LAST: %[[ZERO:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// LAST: %[[RES:.*]] = stablehlo.reduce(%[[CONV]] init: %[[ZERO]]) applies stablehlo.add across dimensions = [1] : (tensor<2x2xf64>, tensor<f64>) -> tensor<2xf64>
// LAST: return %[[RES]] : tensor<2xf64>

// TUPLE-LABEL: func.func @abs
// TUPLE: %[[C0:.*]] = stablehlo.convert %arg0 : (tensor<2xf64>) -> tensor<2xf32>
// TUPLE: %[[C1:.*]] = stablehlo.convert %[[C0]] : (tensor<2xf32>) -> tensor<2xf64>
// TUPLE: %[[SUB:.*]] = stablehlo.subtract %arg0, %[[C1]] : tensor<2xf64>
// TUPLE: %[[C2:.*]] = stablehlo.convert %[[SUB]] : (tensor<2xf64>) -> tensor<2xf32>
// TUPLE: %[[TUP:.*]] = stablehlo.tuple %[[C0]], %[[C2]] : tuple<tensor<2xf32>, tensor<2xf32>>
// TUPLE: %[[HI:.*]] = stablehlo.get_tuple_element %[[TUP]][0] : (tuple<tensor<2xf32>, tensor<2xf32>>) -> tensor<2xf32>
// TUPLE: %[[CST:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<2xf32>
// TUPLE: %[[CMP:.*]] = stablehlo.compare GE, %[[HI]], %[[CST]] : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>
// TUPLE: %[[LO:.*]] = stablehlo.get_tuple_element %[[TUP]][1] : (tuple<tensor<2xf32>, tensor<2xf32>>) -> tensor<2xf32>
// TUPLE: %[[NEG_HI:.*]] = stablehlo.negate %[[HI]] : tensor<2xf32>
// TUPLE: %[[NEG_LO:.*]] = stablehlo.negate %[[LO]] : tensor<2xf32>
// TUPLE: %[[SEL_HI:.*]] = stablehlo.select %[[CMP]], %[[HI]], %[[NEG_HI]] : tensor<2xi1>, tensor<2xf32>
// TUPLE: %[[SEL_LO:.*]] = stablehlo.select %[[CMP]], %[[LO]], %[[NEG_LO]] : tensor<2xi1>, tensor<2xf32>
// TUPLE: %[[RES_TUP:.*]] = stablehlo.tuple %[[SEL_HI]], %[[SEL_LO]] : tuple<tensor<2xf32>, tensor<2xf32>>
// TUPLE: %[[CONV1:.*]] = stablehlo.convert %[[SEL_HI]] : (tensor<2xf32>) -> tensor<2xf64>
// TUPLE: %[[CONV2:.*]] = stablehlo.convert %[[SEL_LO]] : (tensor<2xf32>) -> tensor<2xf64>
// TUPLE: %[[ADD:.*]] = stablehlo.add %[[CONV1]], %[[CONV2]] : tensor<2xf64>
// TUPLE: return %[[ADD]] : tensor<2xf64>

func.func @main() attributes {enzyme.no_multifloat} {
  %cst = stablehlo.constant dense<[-1.1, 2.2]> : tensor<2xf64>
  %expected = stablehlo.constant dense<[1.1, 2.2]> : tensor<2xf64>
  
  %res = func.call @abs(%cst) : (tensor<2xf64>) -> tensor<2xf64>
  
  "check.expect_almost_eq"(%res, %expected) : (tensor<2xf64>, tensor<2xf64>) -> ()
  return
}
