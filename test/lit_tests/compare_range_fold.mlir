// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=compare_op_canon" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s

// The guard of a grid's last block, (n - 1 - lane) - 2 * block >= 0, over 4
// blocks of 2 lanes with n = 8: every lane of every block passes.
func.func @grid_tail(%x: tensor<4x2xf64>) -> tensor<4x2xi1> {
  %n = stablehlo.constant dense<7> : tensor<4x2xi64>
  %two = stablehlo.constant dense<-2> : tensor<4xi64>
  %zero = stablehlo.constant dense<0> : tensor<4x2xi64>
  %lane = stablehlo.iota dim = 1 : tensor<4x2xi64>
  %block = stablehlo.iota dim = 0 : tensor<4xi64>
  %a = stablehlo.subtract %n, %lane : tensor<4x2xi64>
  %b = stablehlo.multiply %two, %block : tensor<4xi64>
  %c = stablehlo.broadcast_in_dim %b, dims = [0] : (tensor<4xi64>) -> tensor<4x2xi64>
  %d = stablehlo.add %a, %c : tensor<4x2xi64>
  %e = stablehlo.compare GE, %d, %zero : (tensor<4x2xi64>, tensor<4x2xi64>) -> tensor<4x2xi1>
  return %e : tensor<4x2xi1>
}

// CHECK-LABEL: func.func @grid_tail(%arg0: tensor<4x2xf64>) -> tensor<4x2xi1> {
// CHECK-NEXT: %c = stablehlo.constant dense<true> : tensor<4x2xi1>
// CHECK-NEXT: return %c : tensor<4x2xi1>
// CHECK-NEXT: }

// With n = 7 the last block's second lane fails: the ranges overlap and the
// compare stays.
func.func @grid_tail_partial(%x: tensor<4x2xf64>) -> tensor<4x2xi1> {
  %n = stablehlo.constant dense<6> : tensor<4x2xi64>
  %two = stablehlo.constant dense<-2> : tensor<4xi64>
  %zero = stablehlo.constant dense<0> : tensor<4x2xi64>
  %lane = stablehlo.iota dim = 1 : tensor<4x2xi64>
  %block = stablehlo.iota dim = 0 : tensor<4xi64>
  %a = stablehlo.subtract %n, %lane : tensor<4x2xi64>
  %b = stablehlo.multiply %two, %block : tensor<4xi64>
  %c = stablehlo.broadcast_in_dim %b, dims = [0] : (tensor<4xi64>) -> tensor<4x2xi64>
  %d = stablehlo.add %a, %c : tensor<4x2xi64>
  %e = stablehlo.compare GE, %d, %zero : (tensor<4x2xi64>, tensor<4x2xi64>) -> tensor<4x2xi1>
  return %e : tensor<4x2xi1>
}

// CHECK-LABEL: func.func @grid_tail_partial(%arg0: tensor<4x2xf64>) -> tensor<4x2xi1> {
// CHECK-NEXT: %c = stablehlo.constant dense<6> : tensor<4x2xi64>
// CHECK-NEXT: %c_0 = stablehlo.constant dense<-2> : tensor<4xi64>
// CHECK-NEXT: %c_1 = stablehlo.constant dense<0> : tensor<4x2xi64>
// CHECK-NEXT: %0 = stablehlo.iota dim = 1 : tensor<4x2xi64>
// CHECK-NEXT: %1 = stablehlo.iota dim = 0 : tensor<4xi64>
// CHECK-NEXT: %2 = stablehlo.subtract %c, %0 {enzymexla.non_negative = [#enzymexla.guaranteed<NOTGUARANTEED>]} : tensor<4x2xi64>
// CHECK-NEXT: %3 = stablehlo.multiply %c_0, %1 : tensor<4xi64>
// CHECK-NEXT: %4 = stablehlo.broadcast_in_dim %3, dims = [0] : (tensor<4xi64>) -> tensor<4x2xi64>
// CHECK-NEXT: %5 = stablehlo.add %2, %4 {enzymexla.non_negative = [#enzymexla.guaranteed<NOTGUARANTEED>]} : tensor<4x2xi64>
// CHECK-NEXT: %6 = stablehlo.compare GE, %5, %c_1 : (tensor<4x2xi64>, tensor<4x2xi64>) -> tensor<4x2xi1>
// CHECK-NEXT: return %6 : tensor<4x2xi1>
// CHECK-NEXT: }

// A padded lane index past the extent never equals a real one.
func.func @never_equal() -> tensor<8xi1> {
  %k = stablehlo.constant dense<8> : tensor<8xi32>
  %i = stablehlo.iota dim = 0 : tensor<8xi32>
  %e = stablehlo.compare EQ, %i, %k : (tensor<8xi32>, tensor<8xi32>) -> tensor<8xi1>
  return %e : tensor<8xi1>
}

// CHECK-LABEL: func.func @never_equal() -> tensor<8xi1> {
// CHECK-NEXT: %c = stablehlo.constant dense<false> : tensor<8xi1>
// CHECK-NEXT: return %c : tensor<8xi1>
// CHECK-NEXT: }

// A product that may wrap the element type (here i8) proves nothing.
func.func @may_wrap() -> tensor<64xi1> {
  %k = stablehlo.constant dense<4> : tensor<64xi8>
  %z = stablehlo.constant dense<0> : tensor<64xi8>
  %i = stablehlo.iota dim = 0 : tensor<64xi8>
  %m = stablehlo.multiply %i, %k : tensor<64xi8>
  %e = stablehlo.compare GE, %m, %z : (tensor<64xi8>, tensor<64xi8>) -> tensor<64xi1>
  return %e : tensor<64xi1>
}

// CHECK-LABEL: func.func @may_wrap() -> tensor<64xi1> {
// CHECK-NEXT: %c = stablehlo.constant {enzymexla.non_negative = [#enzymexla.guaranteed<GUARANTEED>]} dense<4> : tensor<64xi8>
// CHECK-NEXT: %c_0 = stablehlo.constant dense<0> : tensor<64xi8>
// CHECK-NEXT: %0 = stablehlo.iota dim = 0 {enzymexla.non_negative = [#enzymexla.guaranteed<GUARANTEED>]} : tensor<64xi8>
// CHECK-NEXT: %1 = stablehlo.multiply %0, %c {enzymexla.non_negative = [#enzymexla.guaranteed<GUARANTEED>]} : tensor<64xi8>
// CHECK-NEXT: %2 = stablehlo.compare GE, %1, %c_0 : (tensor<64xi8>, tensor<64xi8>) -> tensor<64xi1>
// CHECK-NEXT: return %2 : tensor<64xi1>
// CHECK-NEXT: }
