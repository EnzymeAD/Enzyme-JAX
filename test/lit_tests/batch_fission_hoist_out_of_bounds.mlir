// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=greedy_while_loop_batch_fission" --transform-interpreter --enzyme-hlo-remove-transform 2>/dev/null | FileCheck %s

// The slice of %q sweeps indices 25..28 of a 1-element operand, so it cannot
// be hoisted. Fission must decide that before creating anything: the loop
// stays as it is and no wrapper function or hoisted op is left behind.
func.func @main(%x: tensor<1xf64>, %p: tensor<25xf64>, %y: tensor<8xf64>) -> (tensor<1xf64>, tensor<25xf64>, tensor<8xf64>) {
  %c0 = stablehlo.constant dense<0> : tensor<i64>
  %c1 = stablehlo.constant dense<1> : tensor<i64>
  %c4 = stablehlo.constant dense<4> : tensor<i64>
  %c25 = stablehlo.constant dense<25> : tensor<i64>
  %0:4 = stablehlo.while(%i = %c0, %q = %x, %w = %p, %acc = %y) : tensor<i64>, tensor<1xf64>, tensor<25xf64>, tensor<8xf64>
   cond {
    %c = stablehlo.compare LT, %i, %c4 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %c : tensor<i1>
  } do {
    %j = stablehlo.add %i, %c25 : tensor<i64>
    %s = stablehlo.dynamic_slice %q, %j, sizes = [1] : (tensor<1xf64>, tensor<i64>) -> tensor<1xf64>
    %sr = stablehlo.reshape %s : (tensor<1xf64>) -> tensor<f64>
    %t = stablehlo.dynamic_slice %w, %i, sizes = [1] : (tensor<25xf64>, tensor<i64>) -> tensor<1xf64>
    %tr = stablehlo.reshape %t : (tensor<1xf64>) -> tensor<f64>
    %m = stablehlo.multiply %sr, %tr : tensor<f64>
    %mr = stablehlo.reshape %m : (tensor<f64>) -> tensor<1xf64>
    %u = stablehlo.dynamic_update_slice %acc, %mr, %i : (tensor<8xf64>, tensor<1xf64>, tensor<i64>) -> tensor<8xf64>
    %n = stablehlo.add %i, %c1 : tensor<i64>
    stablehlo.return %n, %q, %w, %u : tensor<i64>, tensor<1xf64>, tensor<25xf64>, tensor<8xf64>
  }
  return %0#1, %0#2, %0#3 : tensor<1xf64>, tensor<25xf64>, tensor<8xf64>
}

// CHECK-NOT: func.func private
// CHECK:  func.func @main(%arg0: tensor<1xf64>, %arg1: tensor<25xf64>, %arg2: tensor<8xf64>) -> (tensor<1xf64>, tensor<25xf64>, tensor<8xf64>) {
// CHECK-NEXT:   %c = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:   %c_0 = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:   %c_1 = stablehlo.constant dense<4> : tensor<i64>
// CHECK-NEXT:   %c_2 = stablehlo.constant dense<25> : tensor<i64>
// CHECK-NEXT:   %0:4 = stablehlo.while(%iterArg = %c, %iterArg_3 = %arg0, %iterArg_4 = %arg1, %iterArg_5 = %arg2) : tensor<i64>, tensor<1xf64>, tensor<25xf64>, tensor<8xf64>
// CHECK-NEXT:   cond {
// CHECK-NEXT:     %1 = stablehlo.compare LT, %iterArg, %c_1 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:     stablehlo.return %1 : tensor<i1>
// CHECK-NEXT:   } do {
// CHECK-NEXT:     %1 = stablehlo.add %iterArg, %c_2 : tensor<i64>
// CHECK-NEXT:     %2 = stablehlo.dynamic_slice %iterArg_3, %1, sizes = [1] : (tensor<1xf64>, tensor<i64>) -> tensor<1xf64>
// CHECK-NEXT:     %3 = stablehlo.reshape %2 : (tensor<1xf64>) -> tensor<f64>
// CHECK-NEXT:     %4 = stablehlo.dynamic_slice %iterArg_4, %iterArg, sizes = [1] : (tensor<25xf64>, tensor<i64>) -> tensor<1xf64>
// CHECK-NEXT:     %5 = stablehlo.reshape %4 : (tensor<1xf64>) -> tensor<f64>
// CHECK-NEXT:     %6 = stablehlo.multiply %3, %5 : tensor<f64>
// CHECK-NEXT:     %7 = stablehlo.reshape %6 : (tensor<f64>) -> tensor<1xf64>
// CHECK-NEXT:     %8 = stablehlo.dynamic_update_slice %iterArg_5, %7, %iterArg : (tensor<8xf64>, tensor<1xf64>, tensor<i64>) -> tensor<8xf64>
// CHECK-NEXT:     %9 = stablehlo.add %iterArg, %c_0 : tensor<i64>
// CHECK-NEXT:     stablehlo.return %9, %iterArg_3, %iterArg_4, %8 : tensor<i64>, tensor<1xf64>, tensor<25xf64>, tensor<8xf64>
// CHECK-NEXT:   }
// CHECK-NEXT:   return %0#1, %0#2, %0#3 : tensor<1xf64>, tensor<25xf64>, tensor<8xf64>
// CHECK-NEXT: }
// CHECK-NOT: func.func private
