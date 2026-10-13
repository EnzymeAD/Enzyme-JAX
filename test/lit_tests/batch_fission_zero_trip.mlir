// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=enzyme_hlo_unroll(16);greedy_while_loop_batch_fission" --transform-interpreter --enzyme-hlo-remove-transform --canonicalize | FileCheck %s

// Unrolling the outer loop leaves inner loops with constant bounds, the first
// of which runs no iterations; batching it must not create zero-sized tensors.
func.func @main(%x: tensor<36xf64>, %idx: tensor<36xi64>) -> tensor<f64> {
  %c0 = stablehlo.constant dense<0> : tensor<i64>
  %c1 = stablehlo.constant dense<1> : tensor<i64>
  %c3 = stablehlo.constant dense<3> : tensor<i64>
  %zero = stablehlo.constant dense<0.0> : tensor<f64>
  %0:2 = stablehlo.while(%i = %c0, %acc = %zero) : tensor<i64>, tensor<f64>
   cond {
    %c = stablehlo.compare LT, %i, %c3 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %c : tensor<i1>
  } do {
    %1:2 = stablehlo.while(%j = %c0, %acc2 = %acc) : tensor<i64>, tensor<f64>
     cond {
      %c = stablehlo.compare LT, %j, %i : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %c : tensor<i1>
    } do {
      %s = stablehlo.dynamic_slice %idx, %j, sizes = [1] : (tensor<36xi64>, tensor<i64>) -> tensor<1xi64>
      %sr = stablehlo.reshape %s : (tensor<1xi64>) -> tensor<i64>
      %v = stablehlo.dynamic_slice %x, %sr, sizes = [1] : (tensor<36xf64>, tensor<i64>) -> tensor<1xf64>
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

// CHECK:  func.func @main(%arg0: tensor<36xf64>, %arg1: tensor<36xi64>) -> tensor<f64> {
// CHECK-NEXT:   %c = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:   %c_0 = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:   %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:   %0 = stablehlo.add %c, %c_0 : tensor<i64>
// CHECK-NEXT:   %1:2 = stablehlo.while(%iterArg = %c, %iterArg_1 = %cst) : tensor<i64>, tensor<f64>
// CHECK-NEXT:   cond {
// CHECK-NEXT:     %4 = stablehlo.compare LT, %iterArg, %0 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:     stablehlo.return %4 : tensor<i1>
// CHECK-NEXT:   } do {
// CHECK-NEXT:     %4 = stablehlo.dynamic_slice %arg1, %iterArg, sizes = [1] : (tensor<36xi64>, tensor<i64>) -> tensor<1xi64>
// CHECK-NEXT:     %5 = stablehlo.reshape %4 : (tensor<1xi64>) -> tensor<i64>
// CHECK-NEXT:     %6 = stablehlo.dynamic_slice %arg0, %5, sizes = [1] : (tensor<36xf64>, tensor<i64>) -> tensor<1xf64>
// CHECK-NEXT:     %7 = stablehlo.reshape %6 : (tensor<1xf64>) -> tensor<f64>
// CHECK-NEXT:     %8 = stablehlo.add %iterArg_1, %7 : tensor<f64>
// CHECK-NEXT:     %9 = stablehlo.add %iterArg, %c_0 : tensor<i64>
// CHECK-NEXT:     stablehlo.return %9, %8 : tensor<i64>, tensor<f64>
// CHECK-NEXT:   }
// CHECK-NEXT:   %2 = stablehlo.add %0, %c_0 : tensor<i64>
// CHECK-NEXT:   %3:2 = stablehlo.while(%iterArg = %c, %iterArg_1 = %1#1) : tensor<i64>, tensor<f64>
// CHECK-NEXT:   cond {
// CHECK-NEXT:     %4 = stablehlo.compare LT, %iterArg, %2 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:     stablehlo.return %4 : tensor<i1>
// CHECK-NEXT:   } do {
// CHECK-NEXT:     %4 = stablehlo.dynamic_slice %arg1, %iterArg, sizes = [1] : (tensor<36xi64>, tensor<i64>) -> tensor<1xi64>
// CHECK-NEXT:     %5 = stablehlo.reshape %4 : (tensor<1xi64>) -> tensor<i64>
// CHECK-NEXT:     %6 = stablehlo.dynamic_slice %arg0, %5, sizes = [1] : (tensor<36xf64>, tensor<i64>) -> tensor<1xf64>
// CHECK-NEXT:     %7 = stablehlo.reshape %6 : (tensor<1xf64>) -> tensor<f64>
// CHECK-NEXT:     %8 = stablehlo.add %iterArg_1, %7 : tensor<f64>
// CHECK-NEXT:     %9 = stablehlo.add %iterArg, %c_0 : tensor<i64>
// CHECK-NEXT:     stablehlo.return %9, %8 : tensor<i64>, tensor<f64>
// CHECK-NEXT:   }
// CHECK-NEXT:   return %3#1 : tensor<f64>
// CHECK-NEXT: }
