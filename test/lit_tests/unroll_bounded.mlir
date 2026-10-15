// RUN: enzymexlamlir-opt --enzyme-hlo-unroll=max-num-iterations=4 %s | FileCheck %s
// RUN: enzymexlamlir-opt --enzyme-hlo-unroll="max-num-iterations=4 bounded-guard-if=true" %s | FileCheck %s --check-prefix=IF
// RUN: enzymexlamlir-opt --enzyme-hlo-unroll="max-num-iterations=4 max-bounded-iterations=1" %s | FileCheck %s --check-prefix=ONE

// The grid-stride loop of a kernel launched on 27 blocks over 53 elements:
// block b handles elements b and b + 27, the second only when it exists. The
// limit is 1 or 2, so the loop unrolls to two copies, the second kept only
// where the loop's condition holds.
func.func @grid_stride(%b: tensor<i32>, %x: tensor<53xf64>, %acc: tensor<53xf64>) -> tensor<53xf64> {
  %c0 = stablehlo.constant dense<0> : tensor<i32>
  %c1 = stablehlo.constant dense<1> : tensor<i32>
  %c26 = stablehlo.constant dense<26> : tensor<i32>
  %c27 = stablehlo.constant dense<27> : tensor<i32>
  %c53 = stablehlo.constant dense<53> : tensor<i32>
  %n = stablehlo.subtract %c53, %b : tensor<i32>
  %m = stablehlo.add %n, %c26 : tensor<i32>
  %trip = stablehlo.divide %m, %c27 {enzymexla.bounds = [[1, 2]]} : tensor<i32>
  %0:2 = stablehlo.while(%k = %c0, %a = %acc) : tensor<i32>, tensor<53xf64>
   cond {
    %c = stablehlo.compare LT, %k, %trip : (tensor<i32>, tensor<i32>) -> tensor<i1>
    stablehlo.return %c : tensor<i1>
  } do {
    %s = stablehlo.multiply %k, %c27 : tensor<i32>
    %e = stablehlo.add %b, %s : tensor<i32>
    %v = stablehlo.dynamic_slice %x, %e, sizes = [1] : (tensor<53xf64>, tensor<i32>) -> tensor<1xf64>
    %u = stablehlo.dynamic_update_slice %a, %v, %e : (tensor<53xf64>, tensor<1xf64>, tensor<i32>) -> tensor<53xf64>
    %kn = stablehlo.add %k, %c1 : tensor<i32>
    stablehlo.return %kn, %u : tensor<i32>, tensor<53xf64>
  }
  return %0#1 : tensor<53xf64>
}

// CHECK:  func.func @grid_stride(%arg0: tensor<i32>, %arg1: tensor<53xf64>, %arg2: tensor<53xf64>) -> tensor<53xf64> {
// CHECK-NEXT:  %c = stablehlo.constant dense<0> : tensor<i32>
// CHECK-NEXT:  %c_0 = stablehlo.constant dense<1> : tensor<i32>
// CHECK-NEXT:  %c_1 = stablehlo.constant dense<26> : tensor<i32>
// CHECK-NEXT:  %c_2 = stablehlo.constant dense<27> : tensor<i32>
// CHECK-NEXT:  %c_3 = stablehlo.constant dense<53> : tensor<i32>
// CHECK-NEXT:  %0 = stablehlo.subtract %c_3, %arg0 : tensor<i32>
// CHECK-NEXT:  %1 = stablehlo.add %0, %c_1 : tensor<i32>
// CHECK-NEXT:  %2 = stablehlo.divide %1, %c_2 {enzymexla.bounds = {{\[\[}}1, 2{{\]\]}}} : tensor<i32>
// CHECK-NEXT:  %3 = stablehlo.multiply %c, %c_2 : tensor<i32>
// CHECK-NEXT:  %4 = stablehlo.add %arg0, %3 : tensor<i32>
// CHECK-NEXT:  %5 = stablehlo.dynamic_slice %arg1, %4, sizes = [1] : (tensor<53xf64>, tensor<i32>) -> tensor<1xf64>
// CHECK-NEXT:  %6 = stablehlo.dynamic_update_slice %arg2, %5, %4 : (tensor<53xf64>, tensor<1xf64>, tensor<i32>) -> tensor<53xf64>
// CHECK-NEXT:  %7 = stablehlo.add %c, %c_0 : tensor<i32>
// CHECK-NEXT:  %8 = stablehlo.compare LT, %7, %2 : (tensor<i32>, tensor<i32>) -> tensor<i1>
// CHECK-NEXT:  %9 = stablehlo.multiply %7, %c_2 : tensor<i32>
// CHECK-NEXT:  %10 = stablehlo.add %arg0, %9 : tensor<i32>
// CHECK-NEXT:  %11 = stablehlo.dynamic_slice %arg1, %10, sizes = [1] : (tensor<53xf64>, tensor<i32>) -> tensor<1xf64>
// CHECK-NEXT:  %12 = stablehlo.dynamic_update_slice %6, %11, %10 : (tensor<53xf64>, tensor<1xf64>, tensor<i32>) -> tensor<53xf64>
// CHECK-NEXT:  %13 = stablehlo.broadcast_in_dim %8, dims = [] : (tensor<i1>) -> tensor<53xi1>
// CHECK-NEXT:  %14 = stablehlo.select %13, %12, %6 : tensor<53xi1>, tensor<53xf64>
// CHECK-NEXT:  return %14 : tensor<53xf64>
// CHECK-NEXT:  }

// IF:  func.func @grid_stride
// IF-NOT: stablehlo.while
// IF:    stablehlo.if
// IF:    return

// With one bounded trip allowed the loop stays, while a constant trip count
// of up to four would still unroll.
// ONE:  func.func @grid_stride
// ONE:    stablehlo.while

// A limit with no bound stays a loop.
func.func @unbounded(%trip: tensor<i32>, %x: tensor<53xf64>, %acc: tensor<53xf64>) -> tensor<53xf64> {
  %c0 = stablehlo.constant dense<0> : tensor<i32>
  %c1 = stablehlo.constant dense<1> : tensor<i32>
  %0:2 = stablehlo.while(%k = %c0, %a = %acc) : tensor<i32>, tensor<53xf64>
   cond {
    %c = stablehlo.compare LT, %k, %trip : (tensor<i32>, tensor<i32>) -> tensor<i1>
    stablehlo.return %c : tensor<i1>
  } do {
    %v = stablehlo.dynamic_slice %x, %k, sizes = [1] : (tensor<53xf64>, tensor<i32>) -> tensor<1xf64>
    %u = stablehlo.dynamic_update_slice %a, %v, %k : (tensor<53xf64>, tensor<1xf64>, tensor<i32>) -> tensor<53xf64>
    %kn = stablehlo.add %k, %c1 : tensor<i32>
    stablehlo.return %kn, %u : tensor<i32>, tensor<53xf64>
  }
  return %0#1 : tensor<53xf64>
}

// CHECK:  func.func @unbounded
// CHECK:    stablehlo.while

// A bound past the iteration limit stays a loop.
func.func @too_many(%b: tensor<i32>, %x: tensor<53xf64>, %acc: tensor<53xf64>) -> tensor<53xf64> {
  %c0 = stablehlo.constant dense<0> : tensor<i32>
  %c1 = stablehlo.constant dense<1> : tensor<i32>
  %trip = stablehlo.add %b, %c1 {enzymexla.bounds = [[1, 9]]} : tensor<i32>
  %0:2 = stablehlo.while(%k = %c0, %a = %acc) : tensor<i32>, tensor<53xf64>
   cond {
    %c = stablehlo.compare LT, %k, %trip : (tensor<i32>, tensor<i32>) -> tensor<i1>
    stablehlo.return %c : tensor<i1>
  } do {
    %v = stablehlo.dynamic_slice %x, %k, sizes = [1] : (tensor<53xf64>, tensor<i32>) -> tensor<1xf64>
    %u = stablehlo.dynamic_update_slice %a, %v, %k : (tensor<53xf64>, tensor<1xf64>, tensor<i32>) -> tensor<53xf64>
    %kn = stablehlo.add %k, %c1 : tensor<i32>
    stablehlo.return %kn, %u : tensor<i32>, tensor<53xf64>
  }
  return %0#1 : tensor<53xf64>
}

// CHECK:  func.func @too_many
// CHECK:    stablehlo.while
