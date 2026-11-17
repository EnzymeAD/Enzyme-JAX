// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=if_op_lift_common_ops" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s

// A chain of common operations, each using the result of the previous one,
// is lifted whole, in one application of the pattern; the constant too. The
// subtract differs from the add and stays.
func.func @chain(%pred: tensor<i1>, %x: tensor<4xf32>, %y: tensor<4xf32>, %z: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
  %0:2 = "stablehlo.if"(%pred) ({
    %cst = stablehlo.constant dense<2.000000e+00> : tensor<4xf32>
    %a = stablehlo.add %x, %y : tensor<4xf32>
    %b = stablehlo.multiply %a, %z : tensor<4xf32>
    %c = stablehlo.negate %b : tensor<4xf32>
    %d = stablehlo.add %c, %cst : tensor<4xf32>
    stablehlo.return %b, %d : tensor<4xf32>, tensor<4xf32>
  }, {
    %cst = stablehlo.constant dense<2.000000e+00> : tensor<4xf32>
    %a = stablehlo.add %x, %y : tensor<4xf32>
    %b = stablehlo.multiply %a, %z : tensor<4xf32>
    %c = stablehlo.negate %b : tensor<4xf32>
    %d = stablehlo.subtract %c, %cst : tensor<4xf32>
    stablehlo.return %b, %d : tensor<4xf32>, tensor<4xf32>
  }) : (tensor<i1>) -> (tensor<4xf32>, tensor<4xf32>)
  return %0#0, %0#1 : tensor<4xf32>, tensor<4xf32>
}

// CHECK-LABEL: func.func @chain
// CHECK-SAME: (%[[PRED:.+]]: tensor<i1>, %[[X:.+]]: tensor<4xf32>, %[[Y:.+]]: tensor<4xf32>, %[[Z:.+]]: tensor<4xf32>)
// CHECK-NEXT:   %[[CST:.+]] = stablehlo.constant dense<2.000000e+00> : tensor<4xf32>
// CHECK-NEXT:   %[[A:.+]] = stablehlo.add %[[X]], %[[Y]] : tensor<4xf32>
// CHECK-NEXT:   %[[B:.+]] = stablehlo.multiply %[[A]], %[[Z]] : tensor<4xf32>
// CHECK-NEXT:   %[[C:.+]] = stablehlo.negate %[[B]] : tensor<4xf32>
// CHECK-NEXT:   %[[IF:.+]]:2 = "stablehlo.if"(%[[PRED]]) ({
// CHECK-NEXT:     %[[D:.+]] = stablehlo.add %[[C]], %[[CST]] : tensor<4xf32>
// CHECK-NEXT:     stablehlo.return %[[B]], %[[D]] : tensor<4xf32>, tensor<4xf32>
// CHECK-NEXT:   }, {
// CHECK-NEXT:     %[[E:.+]] = stablehlo.subtract %[[C]], %[[CST]] : tensor<4xf32>
// CHECK-NEXT:     stablehlo.return %[[B]], %[[E]] : tensor<4xf32>, tensor<4xf32>
// CHECK-NEXT:   })
// CHECK-NEXT:   return %[[IF]]#0, %[[IF]]#1

// An operation with a region is common when its regions are equivalent.
func.func @region(%pred: tensor<i1>, %x: tensor<4x8xf32>) -> tensor<4xf32> {
  %zero = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %0 = "stablehlo.if"(%pred) ({
    %r = stablehlo.reduce(%x init: %zero) applies stablehlo.add across dimensions = [1] : (tensor<4x8xf32>, tensor<f32>) -> tensor<4xf32>
    %s = stablehlo.add %r, %r : tensor<4xf32>
    stablehlo.return %s : tensor<4xf32>
  }, {
    %r = stablehlo.reduce(%x init: %zero) applies stablehlo.add across dimensions = [1] : (tensor<4x8xf32>, tensor<f32>) -> tensor<4xf32>
    %s = stablehlo.multiply %r, %r : tensor<4xf32>
    stablehlo.return %s : tensor<4xf32>
  }) : (tensor<i1>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// CHECK-LABEL: func.func @region
// CHECK-NEXT:   %[[ZERO:.+]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:   %[[R:.+]] = stablehlo.reduce(%{{.+}} init: %[[ZERO]]) applies stablehlo.add across dimensions = [1]
// CHECK-NEXT:   %[[IF:.+]] = "stablehlo.if"
// CHECK-NEXT:     %[[S:.+]] = stablehlo.add %[[R]], %[[R]] : tensor<4xf32>
// CHECK-NEXT:     stablehlo.return %[[S]]
// CHECK-NEXT:   }, {
// CHECK-NEXT:     %[[T:.+]] = stablehlo.multiply %[[R]], %[[R]] : tensor<4xf32>
// CHECK-NEXT:     stablehlo.return %[[T]]
// CHECK-NEXT:   })
// CHECK-NEXT:   return %[[IF]]

// The adds are the same operation on operands computed differently: nothing
// is common.
func.func @different_operands(%pred: tensor<i1>, %x: tensor<4xf32>, %y: tensor<4xf32>) -> tensor<4xf32> {
  %0 = "stablehlo.if"(%pred) ({
    %a = stablehlo.negate %x : tensor<4xf32>
    %b = stablehlo.add %a, %y : tensor<4xf32>
    stablehlo.return %b : tensor<4xf32>
  }, {
    %a = stablehlo.abs %x : tensor<4xf32>
    %b = stablehlo.add %a, %y : tensor<4xf32>
    stablehlo.return %b : tensor<4xf32>
  }) : (tensor<i1>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// CHECK-LABEL: func.func @different_operands
// CHECK-NEXT:   %[[IF:.+]] = "stablehlo.if"
// CHECK-NEXT:     %[[A:.+]] = stablehlo.negate
// CHECK-NEXT:     %[[B:.+]] = stablehlo.add %[[A]]
// CHECK-NEXT:     stablehlo.return %[[B]]
// CHECK-NEXT:   }, {
// CHECK-NEXT:     %[[C:.+]] = stablehlo.abs
// CHECK-NEXT:     %[[D:.+]] = stablehlo.add %[[C]]
// CHECK-NEXT:     stablehlo.return %[[D]]
// CHECK-NEXT:   })
// CHECK-NEXT:   return %[[IF]]
