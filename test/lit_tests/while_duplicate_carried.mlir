// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=while_duplicate_carried;while_deadresult" --transform-interpreter --enzyme-hlo-remove-transform | FileCheck %s

// A raised kernel carries its accumulator twice: the body reads one copy and
// adds to it, the other is only yielded, and that copy is what the code after
// the loop reads. Both start at the same value and yield the same value, so
// the loop carries one of them and the reader takes its result.
func.func @mirrored(%x: tensor<9x4xf64>) -> tensor<4xf64> {
  %c0 = stablehlo.constant dense<0> : tensor<i64>
  %c1 = stablehlo.constant dense<1> : tensor<i64>
  %c9 = stablehlo.constant dense<9> : tensor<i64>
  %zero = stablehlo.constant dense<0.0> : tensor<4xf64>
  %0:3 = stablehlo.while(%i = %c0, %acc = %zero, %copy = %zero) : tensor<i64>, tensor<4xf64>, tensor<4xf64>
   cond {
    %c = stablehlo.compare LT, %i, %c9 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %c : tensor<i1>
  } do {
    %row = stablehlo.dynamic_slice %x, %i, %c0, sizes = [1, 4] : (tensor<9x4xf64>, tensor<i64>, tensor<i64>) -> tensor<1x4xf64>
    %r = stablehlo.reshape %row : (tensor<1x4xf64>) -> tensor<4xf64>
    %sum = stablehlo.add %acc, %r : tensor<4xf64>
    %n = stablehlo.add %i, %c1 : tensor<i64>
    stablehlo.return %n, %sum, %sum : tensor<i64>, tensor<4xf64>, tensor<4xf64>
  }
  return %0#2 : tensor<4xf64>
}

// CHECK:  func.func @mirrored(%arg0: tensor<9x4xf64>) -> tensor<4xf64> {
// CHECK-NEXT:   %c = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:   %c_0 = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:   %c_1 = stablehlo.constant dense<9> : tensor<i64>
// CHECK-NEXT:   %cst = stablehlo.constant dense<0.000000e+00> : tensor<4xf64>
// CHECK-NEXT:   %0:2 = stablehlo.while(%iterArg = %c, %iterArg_2 = %cst) : tensor<i64>, tensor<4xf64>
// CHECK-NEXT:   cond {
// CHECK-NEXT:     %1 = stablehlo.compare LT, %iterArg, %c_1 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:     stablehlo.return %1 : tensor<i1>
// CHECK-NEXT:   } do {
// CHECK-NEXT:     %1 = stablehlo.dynamic_slice %arg0, %iterArg, %c, sizes = [1, 4] : (tensor<9x4xf64>, tensor<i64>, tensor<i64>) -> tensor<1x4xf64>
// CHECK-NEXT:     %2 = stablehlo.reshape %1 : (tensor<1x4xf64>) -> tensor<4xf64>
// CHECK-NEXT:     %3 = stablehlo.add %iterArg_2, %2 : tensor<4xf64>
// CHECK-NEXT:     %4 = stablehlo.add %iterArg, %c_0 : tensor<i64>
// CHECK-NEXT:     stablehlo.return %4, %3 : tensor<i64>, tensor<4xf64>
// CHECK-NEXT:   }
// CHECK-NEXT:   return %0#1 : tensor<4xf64>
// CHECK-NEXT: }

// -----

// The two positions yield different values, so both are carried.
func.func @distinct(%x: tensor<9x4xf64>) -> (tensor<4xf64>, tensor<4xf64>) {
  %c0 = stablehlo.constant dense<0> : tensor<i64>
  %c1 = stablehlo.constant dense<1> : tensor<i64>
  %c9 = stablehlo.constant dense<9> : tensor<i64>
  %zero = stablehlo.constant dense<0.0> : tensor<4xf64>
  %0:3 = stablehlo.while(%i = %c0, %a = %zero, %b = %zero) : tensor<i64>, tensor<4xf64>, tensor<4xf64>
   cond {
    %c = stablehlo.compare LT, %i, %c9 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %c : tensor<i1>
  } do {
    %row = stablehlo.dynamic_slice %x, %i, %c0, sizes = [1, 4] : (tensor<9x4xf64>, tensor<i64>, tensor<i64>) -> tensor<1x4xf64>
    %r = stablehlo.reshape %row : (tensor<1x4xf64>) -> tensor<4xf64>
    %sa = stablehlo.add %a, %r : tensor<4xf64>
    %sb = stablehlo.multiply %b, %r : tensor<4xf64>
    %n = stablehlo.add %i, %c1 : tensor<i64>
    stablehlo.return %n, %sa, %sb : tensor<i64>, tensor<4xf64>, tensor<4xf64>
  }
  return %0#1, %0#2 : tensor<4xf64>, tensor<4xf64>
}

// CHECK:  func.func @distinct(%arg0: tensor<9x4xf64>) -> (tensor<4xf64>, tensor<4xf64>) {
// CHECK-NEXT:   %c = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:   %c_0 = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:   %c_1 = stablehlo.constant dense<9> : tensor<i64>
// CHECK-NEXT:   %cst = stablehlo.constant dense<0.000000e+00> : tensor<4xf64>
// CHECK-NEXT:   %0:3 = stablehlo.while(%iterArg = %c, %iterArg_2 = %cst, %iterArg_3 = %cst) : tensor<i64>, tensor<4xf64>, tensor<4xf64>
// CHECK-NEXT:   cond {
// CHECK-NEXT:     %1 = stablehlo.compare LT, %iterArg, %c_1 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:     stablehlo.return %1 : tensor<i1>
// CHECK-NEXT:   } do {
// CHECK-NEXT:     %1 = stablehlo.dynamic_slice %arg0, %iterArg, %c, sizes = [1, 4] : (tensor<9x4xf64>, tensor<i64>, tensor<i64>) -> tensor<1x4xf64>
// CHECK-NEXT:     %2 = stablehlo.reshape %1 : (tensor<1x4xf64>) -> tensor<4xf64>
// CHECK-NEXT:     %3 = stablehlo.add %iterArg_2, %2 : tensor<4xf64>
// CHECK-NEXT:     %4 = stablehlo.multiply %iterArg_3, %2 : tensor<4xf64>
// CHECK-NEXT:     %5 = stablehlo.add %iterArg, %c_0 : tensor<i64>
// CHECK-NEXT:     stablehlo.return %5, %3, %4 : tensor<i64>, tensor<4xf64>, tensor<4xf64>
// CHECK-NEXT:   }
// CHECK-NEXT:   return %0#1, %0#2 : tensor<4xf64>, tensor<4xf64>
// CHECK-NEXT: }
