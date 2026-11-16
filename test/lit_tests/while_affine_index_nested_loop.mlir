// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=remove_no_ops_from_while_loop;greedy_while_loop_batch_fission" --transform-interpreter --enzyme-hlo-remove-transform | FileCheck %s

// The inner loop slices an iota by its own induction variable. When analyzing
// the outer loop, that index is not affine in the outer induction variable, so
// the slice must not be given affine index info (previously it was treated as
// loop invariant with scale 0).
func.func @main(%x: tensor<5xf64>) -> tensor<f64> {
  %c0 = stablehlo.constant dense<0> : tensor<i64>
  %c1 = stablehlo.constant dense<1> : tensor<i64>
  %c2 = stablehlo.constant dense<2> : tensor<i64>
  %c5 = stablehlo.constant dense<5> : tensor<i64>
  %zero = stablehlo.constant dense<0.0> : tensor<f64>
  %iota = stablehlo.iota dim = 0 : tensor<5xi64>
  %0:2 = stablehlo.while(%i = %c0, %acc = %zero) : tensor<i64>, tensor<f64>
   cond {
    %c = stablehlo.compare LT, %i, %c2 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %c : tensor<i1>
  } do {
    %1:2 = stablehlo.while(%j = %c0, %acc2 = %acc) : tensor<i64>, tensor<f64>
     cond {
      %c = stablehlo.compare LT, %j, %c5 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %c : tensor<i1>
    } do {
      %s = stablehlo.dynamic_slice %iota, %j, sizes = [1] : (tensor<5xi64>, tensor<i64>) -> tensor<1xi64>
      %sr = stablehlo.reshape %s : (tensor<1xi64>) -> tensor<i64>
      %v = stablehlo.dynamic_slice %x, %sr, sizes = [1] : (tensor<5xf64>, tensor<i64>) -> tensor<1xf64>
      %vr = stablehlo.reshape %v : (tensor<1xf64>) -> tensor<f64>
      %a = stablehlo.add %acc2, %vr : tensor<f64>
      %n = stablehlo.add %j, %c1 : tensor<i64>
      stablehlo.return %n, %a : tensor<i64>, tensor<f64>
    }
    %n = stablehlo.add %i, %c1 : tensor<i64>
    stablehlo.return %n, %1#1 : tensor<i64>, tensor<f64>
  }
  return %0#1 : tensor<f64>
}

// CHECK:  func.func @main(%arg0: tensor<5xf64>) -> tensor<f64> {
// CHECK:    %0:2 = stablehlo.while(%iterArg = %c, %iterArg_2 = %cst) : tensor<i64>, tensor<f64>
// CHECK:    } do {
// CHECK-NEXT:      %1 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<5xf64>, tensor<f64>) -> tensor<f64>
// CHECK-NEXT:      %2 = stablehlo.add %1, %iterArg_2 : tensor<f64>
