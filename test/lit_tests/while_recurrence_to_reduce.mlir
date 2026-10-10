// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=while_recurrence_to_reduce" --transform-interpreter --enzyme-hlo-remove-transform --split-input-file | FileCheck %s

// Lagrange basis values and derivatives over 21 nodes, a node skipped where
// the mask is set: p and q are products, d = d * s + p the derivative by the
// product rule, and the last two carries copies of p and q. The values of
// the iteration go to a parallel loop, one row per iteration; the products
// are reduces, and d is x_0 * prod(s) + sum_k p_k * prod(s_j, j > k), p_k
// the running product before iteration k.
func.func @lagrange(%nodes: tensor<21xf64>, %mask: tensor<21x64xi1>, %x: tensor<64xf64>, %y: tensor<f64>) -> (tensor<64xf64>, tensor<64xf64>, tensor<64xf64>, tensor<64xf64>, tensor<64xf64>) {
  %c0 = stablehlo.constant dense<0> : tensor<i64>
  %c1 = stablehlo.constant dense<1> : tensor<i64>
  %c21 = stablehlo.constant dense<21> : tensor<i64>
  %one = stablehlo.constant dense<1.0> : tensor<64xf64>
  %zero = stablehlo.constant dense<0.0> : tensor<64xf64>
  %0:6 = stablehlo.while(%i = %c0, %p = %one, %q = %one, %d = %zero, %pc = %zero, %qc = %zero) : tensor<i64>, tensor<64xf64>, tensor<64xf64>, tensor<64xf64>, tensor<64xf64>, tensor<64xf64>
   cond {
    %c = stablehlo.compare LT, %i, %c21 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %c : tensor<i1>
  } do {
    %m2 = stablehlo.dynamic_slice %mask, %i, %c0, sizes = [1, 64] : (tensor<21x64xi1>, tensor<i64>, tensor<i64>) -> tensor<1x64xi1>
    %m = stablehlo.reshape %m2 : (tensor<1x64xi1>) -> tensor<64xi1>
    %xk1 = stablehlo.dynamic_slice %nodes, %i, sizes = [1] : (tensor<21xf64>, tensor<i64>) -> tensor<1xf64>
    %xk = stablehlo.reshape %xk1 : (tensor<1xf64>) -> tensor<f64>
    %s0 = stablehlo.subtract %y, %xk : tensor<f64>
    %s = stablehlo.broadcast_in_dim %s0, dims = [] : (tensor<f64>) -> tensor<64xf64>
    %ds = stablehlo.multiply %d, %s : tensor<64xf64>
    %dn = stablehlo.add %p, %ds : tensor<64xf64>
    %pn = stablehlo.multiply %p, %s : tensor<64xf64>
    %xb = stablehlo.broadcast_in_dim %xk1, dims = [0] : (tensor<1xf64>) -> tensor<64xf64>
    %t = stablehlo.subtract %x, %xb : tensor<64xf64>
    %qn = stablehlo.multiply %q, %t : tensor<64xf64>
    %d2 = stablehlo.select %m, %d, %dn : tensor<64xi1>, tensor<64xf64>
    %q2 = stablehlo.select %m, %q, %qn : tensor<64xi1>, tensor<64xf64>
    %p2 = stablehlo.select %m, %p, %pn : tensor<64xi1>, tensor<64xf64>
    %n = stablehlo.add %i, %c1 : tensor<i64>
    stablehlo.return %n, %p2, %q2, %d2, %p2, %q2 : tensor<i64>, tensor<64xf64>, tensor<64xf64>, tensor<64xf64>, tensor<64xf64>, tensor<64xf64>
  }
  return %0#1, %0#2, %0#3, %0#4, %0#5 : tensor<64xf64>, tensor<64xf64>, tensor<64xf64>, tensor<64xf64>, tensor<64xf64>
}

// CHECK:  func.func @lagrange(%arg0: tensor<21xf64>, %arg1: tensor<21x64xi1>, %arg2: tensor<64xf64>, %arg3: tensor<f64>) -> (tensor<64xf64>, tensor<64xf64>, tensor<64xf64>, tensor<64xf64>, tensor<64xf64>) {
// CHECK-NEXT:   %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:   %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<1x64xf64>
// CHECK-NEXT:   %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
// CHECK-NEXT:   %cst_2 = stablehlo.constant dense<1.000000e+00> : tensor<21x64xf64>
// CHECK-NEXT:   %c = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:   %c_3 = stablehlo.constant dense<21> : tensor<i64>
// CHECK-NEXT:   %c_4 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:   %cst_5 = stablehlo.constant dense<1.000000e+00> : tensor<64xf64>
// CHECK-NEXT:   %cst_6 = stablehlo.constant dense<0.000000e+00> : tensor<64xf64>
// CHECK-NEXT:   %c_7 = stablehlo.constant dense<false> : tensor<21x64xi1>
// CHECK-NEXT:   %cst_8 = stablehlo.constant dense<0.000000e+00> : tensor<21x64xf64>
// CHECK-NEXT:   %0:4 = stablehlo.while(%iterArg = %c_4, %iterArg_9 = %c_7, %iterArg_10 = %cst_8, %iterArg_11 = %cst_8) : tensor<i64>, tensor<21x64xi1>, tensor<21x64xf64>, tensor<21x64xf64> attributes {enzymexla.parallel}
// CHECK-NEXT:   cond {
// CHECK-NEXT:     %22 = stablehlo.compare LT, %iterArg, %c_3 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:     stablehlo.return %22 : tensor<i1>
// CHECK-NEXT:   } do {
// CHECK-NEXT:     %22 = stablehlo.dynamic_slice %arg1, %iterArg, %c_4, sizes = [1, 64] : (tensor<21x64xi1>, tensor<i64>, tensor<i64>) -> tensor<1x64xi1>
// CHECK-NEXT:     %23 = stablehlo.reshape %22 : (tensor<1x64xi1>) -> tensor<64xi1>
// CHECK-NEXT:     %24 = stablehlo.dynamic_slice %arg0, %iterArg, sizes = [1] : (tensor<21xf64>, tensor<i64>) -> tensor<1xf64>
// CHECK-NEXT:     %25 = stablehlo.reshape %24 : (tensor<1xf64>) -> tensor<f64>
// CHECK-NEXT:     %26 = stablehlo.subtract %arg3, %25 : tensor<f64>
// CHECK-NEXT:     %27 = stablehlo.broadcast_in_dim %26, dims = [] : (tensor<f64>) -> tensor<64xf64>
// CHECK-NEXT:     %28 = stablehlo.broadcast_in_dim %24, dims = [0] : (tensor<1xf64>) -> tensor<64xf64>
// CHECK-NEXT:     %29 = stablehlo.subtract %arg2, %28 : tensor<64xf64>
// CHECK-NEXT:     %30 = stablehlo.subtract %iterArg, %c_4 : tensor<i64>
// CHECK-NEXT:     %31 = stablehlo.add %iterArg, %c : tensor<i64>
// CHECK-NEXT:     %32 = stablehlo.reshape %23 : (tensor<64xi1>) -> tensor<1x64xi1>
// CHECK-NEXT:     %33 = stablehlo.dynamic_update_slice %iterArg_9, %32, %30, %c_4 : (tensor<21x64xi1>, tensor<1x64xi1>, tensor<i64>, tensor<i64>) -> tensor<21x64xi1>
// CHECK-NEXT:     %34 = stablehlo.reshape %27 : (tensor<64xf64>) -> tensor<1x64xf64>
// CHECK-NEXT:     %35 = stablehlo.dynamic_update_slice %iterArg_10, %34, %30, %c_4 : (tensor<21x64xf64>, tensor<1x64xf64>, tensor<i64>, tensor<i64>) -> tensor<21x64xf64>
// CHECK-NEXT:     %36 = stablehlo.reshape %29 : (tensor<64xf64>) -> tensor<1x64xf64>
// CHECK-NEXT:     %37 = stablehlo.dynamic_update_slice %iterArg_11, %36, %30, %c_4 : (tensor<21x64xf64>, tensor<1x64xf64>, tensor<i64>, tensor<i64>) -> tensor<21x64xf64>
// CHECK-NEXT:     stablehlo.return %31, %33, %35, %37 : tensor<i64>, tensor<21x64xi1>, tensor<21x64xf64>, tensor<21x64xf64>
// CHECK-NEXT:   }
// CHECK-NEXT:   %1 = stablehlo.select %0#1, %cst_2, %0#2 : tensor<21x64xi1>, tensor<21x64xf64>
// CHECK-NEXT:   %2 = stablehlo.select %0#1, %cst_2, %0#3 : tensor<21x64xi1>, tensor<21x64xf64>
// CHECK-NEXT:   %3 = stablehlo.select %0#1, %cst_2, %0#2 : tensor<21x64xi1>, tensor<21x64xf64>
// CHECK-NEXT:   %4 = stablehlo.reduce(%1 init: %cst_1) applies stablehlo.multiply across dimensions = [0] : (tensor<21x64xf64>, tensor<f64>) -> tensor<64xf64>
// CHECK-NEXT:   %5 = stablehlo.multiply %cst_5, %4 : tensor<64xf64>
// CHECK-NEXT:   %6 = stablehlo.reduce(%2 init: %cst_1) applies stablehlo.multiply across dimensions = [0] : (tensor<21x64xf64>, tensor<f64>) -> tensor<64xf64>
// CHECK-NEXT:   %7 = stablehlo.multiply %cst_5, %6 : tensor<64xf64>
// CHECK-NEXT:   %8 = stablehlo.broadcast_in_dim %cst_5, dims = [1] : (tensor<64xf64>) -> tensor<21x64xf64>
// CHECK-NEXT:   %9 = "stablehlo.reduce_window"(%1, %cst_1) <{base_dilations = array<i64: 1, 1>, padding = dense<{{\[\[}}20, 0], [0, 0{{\]\]}}> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 21, 1>, window_strides = array<i64: 1, 1>}> ({
// CHECK-NEXT:   ^bb0(%arg4: tensor<f64>, %arg5: tensor<f64>):
// CHECK-NEXT:     %22 = stablehlo.multiply %arg4, %arg5 : tensor<f64>
// CHECK-NEXT:     stablehlo.return %22 : tensor<f64>
// CHECK-NEXT:   }) : (tensor<21x64xf64>, tensor<f64>) -> tensor<21x64xf64>
// CHECK-NEXT:   %10 = stablehlo.slice %9 [0:20, 0:64] : (tensor<21x64xf64>) -> tensor<20x64xf64>
// CHECK-NEXT:   %11 = stablehlo.concatenate %cst_0, %10, dim = 0 : (tensor<1x64xf64>, tensor<20x64xf64>) -> tensor<21x64xf64>
// CHECK-NEXT:   %12 = stablehlo.multiply %8, %11 : tensor<21x64xf64>
// CHECK-NEXT:   %13 = stablehlo.reduce(%3 init: %cst_1) applies stablehlo.multiply across dimensions = [0] : (tensor<21x64xf64>, tensor<f64>) -> tensor<64xf64>
// CHECK-NEXT:   %14 = stablehlo.multiply %cst_6, %13 : tensor<64xf64>
// CHECK-NEXT:   %15 = stablehlo.select %0#1, %cst_8, %12 : tensor<21x64xi1>, tensor<21x64xf64>
// CHECK-NEXT:   %16 = "stablehlo.reduce_window"(%3, %cst_1) <{base_dilations = array<i64: 1, 1>, padding = dense<{{\[\[}}0, 20], [0, 0{{\]\]}}> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 21, 1>, window_strides = array<i64: 1, 1>}> ({
// CHECK-NEXT:   ^bb0(%arg4: tensor<f64>, %arg5: tensor<f64>):
// CHECK-NEXT:     %22 = stablehlo.multiply %arg4, %arg5 : tensor<f64>
// CHECK-NEXT:     stablehlo.return %22 : tensor<f64>
// CHECK-NEXT:   }) : (tensor<21x64xf64>, tensor<f64>) -> tensor<21x64xf64>
// CHECK-NEXT:   %17 = stablehlo.slice %16 [1:21, 0:64] : (tensor<21x64xf64>) -> tensor<20x64xf64>
// CHECK-NEXT:   %18 = stablehlo.concatenate %17, %cst_0, dim = 0 : (tensor<20x64xf64>, tensor<1x64xf64>) -> tensor<21x64xf64>
// CHECK-NEXT:   %19 = stablehlo.multiply %15, %18 : tensor<21x64xf64>
// CHECK-NEXT:   %20 = stablehlo.reduce(%19 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<21x64xf64>, tensor<f64>) -> tensor<64xf64>
// CHECK-NEXT:   %21 = stablehlo.add %14, %20 : tensor<64xf64>
// CHECK-NEXT:   return %5, %7, %21, %5, %7 : tensor<64xf64>, tensor<64xf64>, tensor<64xf64>, tensor<64xf64>, tensor<64xf64>
// CHECK-NEXT: }

// -----

// A sum, a maximum kept back where a flag is false, and a subtraction (a sum
// of negated values). Nothing is kept back from the sum.
func.func @sum_max(%v: tensor<8x4xf32>, %flags: tensor<8xi1>) -> (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) {
  %c0 = stablehlo.constant dense<0> : tensor<i32>
  %c1 = stablehlo.constant dense<1> : tensor<i32>
  %c8 = stablehlo.constant dense<8> : tensor<i32>
  %z = stablehlo.constant dense<0.0> : tensor<4xf32>
  %ninf = stablehlo.constant dense<0xFF800000> : tensor<4xf32>
  %0:4 = stablehlo.while(%i = %c0, %s = %z, %m = %ninf, %d = %z) : tensor<i32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>
   cond {
    %c = stablehlo.compare LT, %i, %c8 : (tensor<i32>, tensor<i32>) -> tensor<i1>
    stablehlo.return %c : tensor<i1>
  } do {
    %r2 = stablehlo.dynamic_slice %v, %i, %c0, sizes = [1, 4] : (tensor<8x4xf32>, tensor<i32>, tensor<i32>) -> tensor<1x4xf32>
    %r = stablehlo.reshape %r2 : (tensor<1x4xf32>) -> tensor<4xf32>
    %f1 = stablehlo.dynamic_slice %flags, %i, sizes = [1] : (tensor<8xi1>, tensor<i32>) -> tensor<1xi1>
    %f = stablehlo.reshape %f1 : (tensor<1xi1>) -> tensor<i1>
    %s1 = stablehlo.add %s, %r : tensor<4xf32>
    %mx = stablehlo.maximum %r, %m : tensor<4xf32>
    %m1 = stablehlo.select %f, %mx, %m : tensor<i1>, tensor<4xf32>
    %d1 = stablehlo.subtract %d, %r : tensor<4xf32>
    %n = stablehlo.add %i, %c1 : tensor<i32>
    stablehlo.return %n, %s1, %m1, %d1 : tensor<i32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>
  }
  return %0#1, %0#2, %0#3 : tensor<4xf32>, tensor<4xf32>, tensor<4xf32>
}

// CHECK:  func.func @sum_max(%arg0: tensor<8x4xf32>, %arg1: tensor<8xi1>) -> (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) {
// CHECK-NEXT:   %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
// CHECK-NEXT:   %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:   %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<8x4xf32>
// CHECK-NEXT:   %c = stablehlo.constant dense<1> : tensor<i32>
// CHECK-NEXT:   %c_2 = stablehlo.constant dense<8> : tensor<i32>
// CHECK-NEXT:   %c_3 = stablehlo.constant dense<0> : tensor<i32>
// CHECK-NEXT:   %cst_4 = stablehlo.constant dense<0.000000e+00> : tensor<4xf32>
// CHECK-NEXT:   %cst_5 = stablehlo.constant dense<0xFF800000> : tensor<4xf32>
// CHECK-NEXT:   %cst_6 = stablehlo.constant dense<0.000000e+00> : tensor<8x4xf32>
// CHECK-NEXT:   %c_7 = stablehlo.constant dense<false> : tensor<8xi1>
// CHECK-NEXT:   %0:3 = stablehlo.while(%iterArg = %c_3, %iterArg_8 = %cst_6, %iterArg_9 = %c_7) : tensor<i32>, tensor<8x4xf32>, tensor<8xi1> attributes {enzymexla.parallel}
// CHECK-NEXT:   cond {
// CHECK-NEXT:     %10 = stablehlo.compare LT, %iterArg, %c_2 : (tensor<i32>, tensor<i32>) -> tensor<i1>
// CHECK-NEXT:     stablehlo.return %10 : tensor<i1>
// CHECK-NEXT:   } do {
// CHECK-NEXT:     %10 = stablehlo.dynamic_slice %arg0, %iterArg, %c_3, sizes = [1, 4] : (tensor<8x4xf32>, tensor<i32>, tensor<i32>) -> tensor<1x4xf32>
// CHECK-NEXT:     %11 = stablehlo.reshape %10 : (tensor<1x4xf32>) -> tensor<4xf32>
// CHECK-NEXT:     %12 = stablehlo.dynamic_slice %arg1, %iterArg, sizes = [1] : (tensor<8xi1>, tensor<i32>) -> tensor<1xi1>
// CHECK-NEXT:     %13 = stablehlo.reshape %12 : (tensor<1xi1>) -> tensor<i1>
// CHECK-NEXT:     %14 = stablehlo.subtract %iterArg, %c_3 : tensor<i32>
// CHECK-NEXT:     %15 = stablehlo.add %iterArg, %c : tensor<i32>
// CHECK-NEXT:     %16 = stablehlo.reshape %11 : (tensor<4xf32>) -> tensor<1x4xf32>
// CHECK-NEXT:     %17 = stablehlo.dynamic_update_slice %iterArg_8, %16, %14, %c_3 : (tensor<8x4xf32>, tensor<1x4xf32>, tensor<i32>, tensor<i32>) -> tensor<8x4xf32>
// CHECK-NEXT:     %18 = stablehlo.reshape %13 : (tensor<i1>) -> tensor<1xi1>
// CHECK-NEXT:     %19 = stablehlo.dynamic_update_slice %iterArg_9, %18, %14 : (tensor<8xi1>, tensor<1xi1>, tensor<i32>) -> tensor<8xi1>
// CHECK-NEXT:     stablehlo.return %15, %17, %19 : tensor<i32>, tensor<8x4xf32>, tensor<8xi1>
// CHECK-NEXT:   }
// CHECK-NEXT:   %1 = stablehlo.broadcast_in_dim %0#2, dims = [0] : (tensor<8xi1>) -> tensor<8x4xi1>
// CHECK-NEXT:   %2 = stablehlo.select %1, %0#1, %cst_1 : tensor<8x4xi1>, tensor<8x4xf32>
// CHECK-NEXT:   %3 = stablehlo.negate %0#1 : tensor<8x4xf32>
// CHECK-NEXT:   %4 = stablehlo.reduce(%0#1 init: %cst_0) applies stablehlo.add across dimensions = [0] : (tensor<8x4xf32>, tensor<f32>) -> tensor<4xf32>
// CHECK-NEXT:   %5 = stablehlo.add %cst_4, %4 : tensor<4xf32>
// CHECK-NEXT:   %6 = stablehlo.reduce(%2 init: %cst) applies stablehlo.maximum across dimensions = [0] : (tensor<8x4xf32>, tensor<f32>) -> tensor<4xf32>
// CHECK-NEXT:   %7 = stablehlo.maximum %cst_5, %6 : tensor<4xf32>
// CHECK-NEXT:   %8 = stablehlo.reduce(%3 init: %cst_0) applies stablehlo.add across dimensions = [0] : (tensor<8x4xf32>, tensor<f32>) -> tensor<4xf32>
// CHECK-NEXT:   %9 = stablehlo.add %cst_4, %8 : tensor<4xf32>
// CHECK-NEXT:   return %5, %7, %9 : tensor<4xf32>, tensor<4xf32>, tensor<4xf32>
// CHECK-NEXT: }

// -----

// An argmin: the carried minimum is compared, which no reduction or linear
// recurrence does. Kept.
func.func @argmin_kept(%v: tensor<8xf32>) -> (tensor<f32>, tensor<i32>) {
  %c0 = stablehlo.constant dense<0> : tensor<i32>
  %c1 = stablehlo.constant dense<1> : tensor<i32>
  %c8 = stablehlo.constant dense<8> : tensor<i32>
  %inf = stablehlo.constant dense<0x7F800000> : tensor<f32>
  %0:3 = stablehlo.while(%i = %c0, %b = %inf, %at = %c0) : tensor<i32>, tensor<f32>, tensor<i32>
   cond {
    %c = stablehlo.compare LT, %i, %c8 : (tensor<i32>, tensor<i32>) -> tensor<i1>
    stablehlo.return %c : tensor<i1>
  } do {
    %r1 = stablehlo.dynamic_slice %v, %i, sizes = [1] : (tensor<8xf32>, tensor<i32>) -> tensor<1xf32>
    %r = stablehlo.reshape %r1 : (tensor<1xf32>) -> tensor<f32>
    %lt = stablehlo.compare LT, %r, %b : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %b1 = stablehlo.select %lt, %r, %b : tensor<i1>, tensor<f32>
    %a1 = stablehlo.select %lt, %i, %at : tensor<i1>, tensor<i32>
    %n = stablehlo.add %i, %c1 : tensor<i32>
    stablehlo.return %n, %b1, %a1 : tensor<i32>, tensor<f32>, tensor<i32>
  }
  return %0#1, %0#2 : tensor<f32>, tensor<i32>
}

// CHECK:  func.func @argmin_kept(%arg0: tensor<8xf32>) -> (tensor<f32>, tensor<i32>) {
// CHECK-NEXT:   %c = stablehlo.constant dense<0> : tensor<i32>
// CHECK-NEXT:   %c_0 = stablehlo.constant dense<1> : tensor<i32>
// CHECK-NEXT:   %c_1 = stablehlo.constant dense<8> : tensor<i32>
// CHECK-NEXT:   %cst = stablehlo.constant dense<0x7F800000> : tensor<f32>
// CHECK-NEXT:   %0:3 = stablehlo.while(%iterArg = %c, %iterArg_2 = %cst, %iterArg_3 = %c) : tensor<i32>, tensor<f32>, tensor<i32>
// CHECK-NEXT:   cond {
// CHECK-NEXT:     %1 = stablehlo.compare LT, %iterArg, %c_1 : (tensor<i32>, tensor<i32>) -> tensor<i1>
// CHECK-NEXT:     stablehlo.return %1 : tensor<i1>
// CHECK-NEXT:   } do {
// CHECK-NEXT:     %1 = stablehlo.dynamic_slice %arg0, %iterArg, sizes = [1] : (tensor<8xf32>, tensor<i32>) -> tensor<1xf32>
// CHECK-NEXT:     %2 = stablehlo.reshape %1 : (tensor<1xf32>) -> tensor<f32>
// CHECK-NEXT:     %3 = stablehlo.compare LT, %2, %iterArg_2 : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK-NEXT:     %4 = stablehlo.select %3, %2, %iterArg_2 : tensor<i1>, tensor<f32>
// CHECK-NEXT:     %5 = stablehlo.select %3, %iterArg, %iterArg_3 : tensor<i1>, tensor<i32>
// CHECK-NEXT:     %6 = stablehlo.add %iterArg, %c_0 : tensor<i32>
// CHECK-NEXT:     stablehlo.return %6, %4, %5 : tensor<i32>, tensor<f32>, tensor<i32>
// CHECK-NEXT:   }
// CHECK-NEXT:   return %0#1, %0#2 : tensor<f32>, tensor<i32>
// CHECK-NEXT: }

// -----

// x = x * x is not linear in x. Kept.
func.func @square_kept(%x0: tensor<f64>) -> tensor<f64> {
  %c0 = stablehlo.constant dense<0> : tensor<i32>
  %c1 = stablehlo.constant dense<1> : tensor<i32>
  %c4 = stablehlo.constant dense<4> : tensor<i32>
  %0:2 = stablehlo.while(%i = %c0, %x = %x0) : tensor<i32>, tensor<f64>
   cond {
    %c = stablehlo.compare LT, %i, %c4 : (tensor<i32>, tensor<i32>) -> tensor<i1>
    stablehlo.return %c : tensor<i1>
  } do {
    %x1 = stablehlo.multiply %x, %x : tensor<f64>
    %n = stablehlo.add %i, %c1 : tensor<i32>
    stablehlo.return %n, %x1 : tensor<i32>, tensor<f64>
  }
  return %0#1 : tensor<f64>
}

// CHECK:  func.func @square_kept(%arg0: tensor<f64>) -> tensor<f64> {
// CHECK-NEXT:   %c = stablehlo.constant dense<0> : tensor<i32>
// CHECK-NEXT:   %c_0 = stablehlo.constant dense<1> : tensor<i32>
// CHECK-NEXT:   %c_1 = stablehlo.constant dense<4> : tensor<i32>
// CHECK-NEXT:   %0:2 = stablehlo.while(%iterArg = %c, %iterArg_2 = %arg0) : tensor<i32>, tensor<f64>
// CHECK-NEXT:   cond {
// CHECK-NEXT:     %1 = stablehlo.compare LT, %iterArg, %c_1 : (tensor<i32>, tensor<i32>) -> tensor<i1>
// CHECK-NEXT:     stablehlo.return %1 : tensor<i1>
// CHECK-NEXT:   } do {
// CHECK-NEXT:     %1 = stablehlo.multiply %iterArg_2, %iterArg_2 : tensor<f64>
// CHECK-NEXT:     %2 = stablehlo.add %iterArg, %c_0 : tensor<i32>
// CHECK-NEXT:     stablehlo.return %2, %1 : tensor<i32>, tensor<f64>
// CHECK-NEXT:   }
// CHECK-NEXT:   return %0#1 : tensor<f64>
// CHECK-NEXT: }
