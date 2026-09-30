// RUN: enzymexlamlir-opt %s --pass-pipeline='builtin.module(enzyme-hlo-generate-td{patterns=slice_if},transform-interpreter,enzyme-hlo-remove-transform)' | FileCheck %s

// The new if must stay at the old one: result #0 has a user before the slice.

func.func @main(%pred: tensor<i1>, %a: tensor<1x4xf64>, %b: tensor<1x4xf64>, %v: tensor<f64>) -> (tensor<f64>, tensor<1x3xf64>) {
  %r:2 = "stablehlo.if"(%pred) ({
    stablehlo.return %v, %a : tensor<f64>, tensor<1x4xf64>
  }, {
    stablehlo.return %v, %b : tensor<f64>, tensor<1x4xf64>
  }) : (tensor<i1>) -> (tensor<f64>, tensor<1x4xf64>)
  %u = stablehlo.multiply %r#0, %r#0 : tensor<f64>
  %s = stablehlo.slice %r#1 [0:1, 0:3] : (tensor<1x4xf64>) -> tensor<1x3xf64>
  return %u, %s : tensor<f64>, tensor<1x3xf64>
}

// CHECK-LABEL: func.func @main
// CHECK-SAME: (%[[PRED:.+]]: tensor<i1>, %[[A:.+]]: tensor<1x4xf64>, %[[B:.+]]: tensor<1x4xf64>, %[[V:.+]]: tensor<f64>)
// CHECK-NEXT: %[[IF:.+]]:2 = "stablehlo.if"(%[[PRED]]) ({
// CHECK-NEXT: %[[SA:.+]] = stablehlo.slice %[[A]] [0:1, 0:3] : (tensor<1x4xf64>) -> tensor<1x3xf64>
// CHECK-NEXT: stablehlo.return %[[V]], %[[SA]] : tensor<f64>, tensor<1x3xf64>
// CHECK-NEXT: }, {
// CHECK-NEXT: %[[SB:.+]] = stablehlo.slice %[[B]] [0:1, 0:3] : (tensor<1x4xf64>) -> tensor<1x3xf64>
// CHECK-NEXT: stablehlo.return %[[V]], %[[SB]] : tensor<f64>, tensor<1x3xf64>
// CHECK-NEXT: }) : (tensor<i1>) -> (tensor<f64>, tensor<1x3xf64>)
// CHECK-NEXT: %[[U:.+]] = stablehlo.multiply %[[IF]]#0, %[[IF]]#0 : tensor<f64>
// CHECK-NEXT: return %[[U]], %[[IF]]#1 : tensor<f64>, tensor<1x3xf64>
