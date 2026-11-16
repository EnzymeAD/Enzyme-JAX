// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=elementwise_slice_to_batch" --transform-interpreter --enzyme-hlo-remove-transform | FileCheck %s

// The while between the two multiplies takes none of them as an operand, but
// its body captures the first one. Batching the multiplies moves the ops of
// the range around them, and the while has to stay after the value its body
// reads.
func.func @region_capture(%x: tensor<2x4xf64>, %c: tensor<4xf64>) -> (tensor<4xf64>, tensor<4xf64>, tensor<4xf64>) {
  %c0 = stablehlo.constant dense<0> : tensor<i64>
  %c1 = stablehlo.constant dense<1> : tensor<i64>
  %c8 = stablehlo.constant dense<8> : tensor<i64>
  %s0 = stablehlo.slice %x [0:1, 0:4] : (tensor<2x4xf64>) -> tensor<1x4xf64>
  %r0 = stablehlo.reshape %s0 : (tensor<1x4xf64>) -> tensor<4xf64>
  %m0 = stablehlo.multiply %r0, %c : tensor<4xf64>
  %w:2 = stablehlo.while(%i = %c0, %acc = %c) : tensor<i64>, tensor<4xf64>
   cond {
    %t = stablehlo.compare LT, %i, %c8 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %t : tensor<i1>
  } do {
    %u = stablehlo.add %acc, %m0 : tensor<4xf64>
    %n = stablehlo.add %i, %c1 : tensor<i64>
    stablehlo.return %n, %u : tensor<i64>, tensor<4xf64>
  }
  %s1 = stablehlo.slice %x [1:2, 0:4] : (tensor<2x4xf64>) -> tensor<1x4xf64>
  %r1 = stablehlo.reshape %s1 : (tensor<1x4xf64>) -> tensor<4xf64>
  %m1 = stablehlo.multiply %r1, %c : tensor<4xf64>
  return %m0, %m1, %w#1 : tensor<4xf64>, tensor<4xf64>, tensor<4xf64>
}

// CHECK:  func.func @region_capture(%arg0: tensor<2x4xf64>, %arg1: tensor<4xf64>) -> (tensor<4xf64>, tensor<4xf64>, tensor<4xf64>) {
// CHECK-NEXT:   %c = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:   %c_0 = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:   %c_1 = stablehlo.constant dense<8> : tensor<i64>
// CHECK-NEXT:   %0 = stablehlo.reshape %arg1 : (tensor<4xf64>) -> tensor<1x4xf64>
// CHECK-NEXT:   %1 = stablehlo.reshape %arg1 : (tensor<4xf64>) -> tensor<1x4xf64>
// CHECK-NEXT:   %2 = stablehlo.concatenate %0, %1, dim = 0 : (tensor<1x4xf64>, tensor<1x4xf64>) -> tensor<2x4xf64>
// CHECK-NEXT:   %3 = stablehlo.multiply %arg0, %2 : tensor<2x4xf64>
// CHECK-NEXT:   %4 = stablehlo.slice %3 [0:1, 0:4] : (tensor<2x4xf64>) -> tensor<1x4xf64>
// CHECK-NEXT:   %5 = stablehlo.reshape %4 : (tensor<1x4xf64>) -> tensor<4xf64>
// CHECK-NEXT:   %6 = stablehlo.slice %3 [1:2, 0:4] : (tensor<2x4xf64>) -> tensor<1x4xf64>
// CHECK-NEXT:   %7 = stablehlo.reshape %6 : (tensor<1x4xf64>) -> tensor<4xf64>
// CHECK-NEXT:   %8:2 = stablehlo.while(%iterArg = %c, %iterArg_2 = %arg1) : tensor<i64>, tensor<4xf64>
// CHECK-NEXT:   cond {
// CHECK-NEXT:     %9 = stablehlo.compare LT, %iterArg, %c_1 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:     stablehlo.return %9 : tensor<i1>
// CHECK-NEXT:   } do {
// CHECK-NEXT:     %9 = stablehlo.add %iterArg_2, %5 : tensor<4xf64>
// CHECK-NEXT:     %10 = stablehlo.add %iterArg, %c_0 : tensor<i64>
// CHECK-NEXT:     stablehlo.return %10, %9 : tensor<i64>, tensor<4xf64>
// CHECK-NEXT:   }
// CHECK-NEXT:   return %5, %7, %8#1 : tensor<4xf64>, tensor<4xf64>, tensor<4xf64>
// CHECK-NEXT: }
