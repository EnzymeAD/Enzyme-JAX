// RUN: enzymexlamlir-opt %s --auto-batching --enzyme-hlo-opt | FileCheck %s
// RUN: enzymexlamlir-opt %s --auto-batching --enzyme-hlo-opt | stablehlo-translate - -allow-unregistered-dialect --interpret

module {
  // 1. GreedyWhileLoopBatchFission: loop-invariant predicate with dot_general in branches.
  // CHECK-LABEL: func.func private @while_uniform_if_dot
  // CHECK-NOT: stablehlo.while
  // CHECK: "stablehlo.if"
  // CHECK: stablehlo.dot_general
  // CHECK: stablehlo.dot_general
  func.func private @while_uniform_if_dot(
      %A: tensor<4x2xf32>, %W1: tensor<2x2xf32>, %W2: tensor<2x2xf32>,
      %pred: tensor<i1>) -> tensor<4x2xf32> {
    %c0 = stablehlo.constant dense<0> : tensor<i32>
    %c1 = stablehlo.constant dense<1> : tensor<i32>
    %c4 = stablehlo.constant dense<4> : tensor<i32>
    %init = stablehlo.constant dense<0.0> : tensor<4x2xf32>
    %0:2 = stablehlo.while(%i = %c0, %buf = %init) : tensor<i32>, tensor<4x2xf32>
     cond {
      %cond = stablehlo.compare LT, %i, %c4 : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %cond : tensor<i1>
    } do {
      %ds = stablehlo.dynamic_slice %A, %i, %c0, sizes = [1, 2] : (tensor<4x2xf32>, tensor<i32>, tensor<i32>) -> tensor<1x2xf32>
      %row = stablehlo.reshape %ds : (tensor<1x2xf32>) -> tensor<2xf32>
      %y = "stablehlo.if"(%pred) ({
        %t = stablehlo.dot_general %row, %W1, contracting_dims = [0] x [0] : (tensor<2xf32>, tensor<2x2xf32>) -> tensor<2xf32>
        stablehlo.return %t : tensor<2xf32>
      }, {
        %f = stablehlo.dot_general %row, %W2, contracting_dims = [0] x [0] : (tensor<2xf32>, tensor<2x2xf32>) -> tensor<2xf32>
        stablehlo.return %f : tensor<2xf32>
      }) : (tensor<i1>) -> tensor<2xf32>
      %y2 = stablehlo.reshape %y : (tensor<2xf32>) -> tensor<1x2xf32>
      %nbuf = stablehlo.dynamic_update_slice %buf, %y2, %i, %c0 : (tensor<4x2xf32>, tensor<1x2xf32>, tensor<i32>, tensor<i32>) -> tensor<4x2xf32>
      %ni = stablehlo.add %i, %c1 : tensor<i32>
      stablehlo.return %ni, %nbuf : tensor<i32>, tensor<4x2xf32>
    }
    return %0#1 : tensor<4x2xf32>
  }

  // 2. GreedyWhileLoopBatchFission: varying predicate with pure multi-result branches -> select.
  // CHECK-LABEL: func.func private @while_varying_if_multi_result
  // CHECK-NOT: stablehlo.while
  // CHECK-NOT: "stablehlo.if"
  // CHECK: stablehlo.select
  func.func private @while_varying_if_multi_result(
      %x: tensor<4x2xf32>, %thresh: tensor<4xf32>) -> (tensor<4x2xf32>, tensor<4x2xf32>) {
    %c0 = stablehlo.constant dense<0> : tensor<i32>
    %c1 = stablehlo.constant dense<1> : tensor<i32>
    %c4 = stablehlo.constant dense<4> : tensor<i32>
    %z = stablehlo.constant dense<0.0> : tensor<f32>
    %init = stablehlo.constant dense<0.0> : tensor<4x2xf32>
    %0:3 = stablehlo.while(%i = %c0, %b0 = %init, %b1 = %init) : tensor<i32>, tensor<4x2xf32>, tensor<4x2xf32>
     cond {
      %cond = stablehlo.compare LT, %i, %c4 : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %cond : tensor<i1>
    } do {
      %ds = stablehlo.dynamic_slice %x, %i, %c0, sizes = [1, 2] : (tensor<4x2xf32>, tensor<i32>, tensor<i32>) -> tensor<1x2xf32>
      %row = stablehlo.reshape %ds : (tensor<1x2xf32>) -> tensor<2xf32>
      %ts = stablehlo.dynamic_slice %thresh, %i, sizes = [1] : (tensor<4xf32>, tensor<i32>) -> tensor<1xf32>
      %tscalar = stablehlo.reshape %ts : (tensor<1xf32>) -> tensor<f32>
      %pred = stablehlo.compare GT, %tscalar, %z : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %y:2 = "stablehlo.if"(%pred) ({
        %two = stablehlo.constant dense<2.0> : tensor<2xf32>
        %r0 = stablehlo.multiply %row, %two : tensor<2xf32>
        %r1 = stablehlo.add %row, %two : tensor<2xf32>
        stablehlo.return %r0, %r1 : tensor<2xf32>, tensor<2xf32>
      }, {
        %five = stablehlo.constant dense<5.0> : tensor<2xf32>
        %r0 = stablehlo.negate %row : tensor<2xf32>
        %r1 = stablehlo.subtract %row, %five : tensor<2xf32>
        stablehlo.return %r0, %r1 : tensor<2xf32>, tensor<2xf32>
      }) : (tensor<i1>) -> (tensor<2xf32>, tensor<2xf32>)
      %y0 = stablehlo.reshape %y#0 : (tensor<2xf32>) -> tensor<1x2xf32>
      %y1 = stablehlo.reshape %y#1 : (tensor<2xf32>) -> tensor<1x2xf32>
      %nb0 = stablehlo.dynamic_update_slice %b0, %y0, %i, %c0 : (tensor<4x2xf32>, tensor<1x2xf32>, tensor<i32>, tensor<i32>) -> tensor<4x2xf32>
      %nb1 = stablehlo.dynamic_update_slice %b1, %y1, %i, %c0 : (tensor<4x2xf32>, tensor<1x2xf32>, tensor<i32>, tensor<i32>) -> tensor<4x2xf32>
      %ni = stablehlo.add %i, %c1 : tensor<i32>
      stablehlo.return %ni, %nb0, %nb1 : tensor<i32>, tensor<4x2xf32>, tensor<4x2xf32>
    }
    return %0#1, %0#2 : tensor<4x2xf32>, tensor<4x2xf32>
  }

  // 3. ParallelWhileToBatchedScatter: enzymexla.parallel while with both uniform and varying ifs.
  // CHECK-LABEL: func.func private @parallel_while_if
  // CHECK-NOT: stablehlo.while
  func.func private @parallel_while_if(
      %x: tensor<4xf32>, %pred_uniform: tensor<i1>) -> tensor<4xf32> {
    %c0 = stablehlo.constant dense<0> : tensor<i32>
    %c1 = stablehlo.constant dense<1> : tensor<i32>
    %c2 = stablehlo.constant dense<2> : tensor<i32>
    %c4 = stablehlo.constant dense<4> : tensor<i32>
    %init = stablehlo.constant dense<0.0> : tensor<4xf32>
    %0:2 = stablehlo.while(%i = %c0, %buf = %init) : tensor<i32>, tensor<4xf32> attributes {enzymexla.parallel}
     cond {
      %cond = stablehlo.compare LT, %i, %c4 : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %cond : tensor<i1>
    } do {
      %ds = stablehlo.dynamic_slice %x, %i, sizes = [1] : (tensor<4xf32>, tensor<i32>) -> tensor<1xf32>
      %xi = stablehlo.reshape %ds : (tensor<1xf32>) -> tensor<f32>
      %u = "stablehlo.if"(%pred_uniform) ({
        %three = stablehlo.constant dense<3.0> : tensor<f32>
        %m = stablehlo.multiply %xi, %three : tensor<f32>
        stablehlo.return %m : tensor<f32>
      }, {
        %one = stablehlo.constant dense<1.0> : tensor<f32>
        %a = stablehlo.add %xi, %one : tensor<f32>
        stablehlo.return %a : tensor<f32>
      }) : (tensor<i1>) -> tensor<f32>
      %pvar = stablehlo.compare LT, %i, %c2 : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %v = "stablehlo.if"(%pvar) ({
        %ten = stablehlo.constant dense<10.0> : tensor<f32>
        %a2 = stablehlo.add %u, %ten : tensor<f32>
        stablehlo.return %a2 : tensor<f32>
      }, {
        %n = stablehlo.negate %u : tensor<f32>
        stablehlo.return %n : tensor<f32>
      }) : (tensor<i1>) -> tensor<f32>
      %v1 = stablehlo.reshape %v : (tensor<f32>) -> tensor<1xf32>
      %nbuf = stablehlo.dynamic_update_slice %buf, %v1, %i : (tensor<4xf32>, tensor<1xf32>, tensor<i32>) -> tensor<4xf32>
      %ni = stablehlo.add %i, %c1 : tensor<i32>
      stablehlo.return %ni, %nbuf : tensor<i32>, tensor<4xf32>
    }
    return %0#1 : tensor<4xf32>
  }

  // 4. Safety check: integer divide guarded by varying predicate must NOT speculate.
  // CHECK-LABEL: func.func private @while_guarded_int_div
  // CHECK: stablehlo.while
  // CHECK: "stablehlo.if"
  func.func private @while_guarded_int_div(
      %num: tensor<4xi32>, %den: tensor<4xi32>) -> tensor<4xi32> {
    %c0 = stablehlo.constant dense<0> : tensor<i32>
    %c1 = stablehlo.constant dense<1> : tensor<i32>
    %c4 = stablehlo.constant dense<4> : tensor<i32>
    %init = stablehlo.constant dense<0> : tensor<4xi32>
    %0:2 = stablehlo.while(%i = %c0, %buf = %init) : tensor<i32>, tensor<4xi32>
     cond {
      %cond = stablehlo.compare LT, %i, %c4 : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %cond : tensor<i1>
    } do {
      %ns = stablehlo.dynamic_slice %num, %i, sizes = [1] : (tensor<4xi32>, tensor<i32>) -> tensor<1xi32>
      %ni_val = stablehlo.reshape %ns : (tensor<1xi32>) -> tensor<i32>
      %ds = stablehlo.dynamic_slice %den, %i, sizes = [1] : (tensor<4xi32>, tensor<i32>) -> tensor<1xi32>
      %di_val = stablehlo.reshape %ds : (tensor<1xi32>) -> tensor<i32>
      %nz = stablehlo.compare NE, %di_val, %c0 : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %q = "stablehlo.if"(%nz) ({
        %div = stablehlo.divide %ni_val, %di_val : tensor<i32>
        stablehlo.return %div : tensor<i32>
      }, {
        %fallback = stablehlo.constant dense<-1> : tensor<i32>
        stablehlo.return %fallback : tensor<i32>
      }) : (tensor<i1>) -> tensor<i32>
      %q1 = stablehlo.reshape %q : (tensor<i32>) -> tensor<1xi32>
      %nbuf = stablehlo.dynamic_update_slice %buf, %q1, %i : (tensor<4xi32>, tensor<1xi32>, tensor<i32>) -> tensor<4xi32>
      %ni = stablehlo.add %i, %c1 : tensor<i32>
      stablehlo.return %ni, %nbuf : tensor<i32>, tensor<4xi32>
    }
    return %0#1 : tensor<4xi32>
  }

  func.func @main() {
    %A = stablehlo.constant dense<[[1.0, 2.0], [3.0, 4.0], [-1.0, 0.5], [2.0, -3.0]]> : tensor<4x2xf32>
    %W1 = stablehlo.constant dense<[[2.0, 0.0], [0.0, 3.0]]> : tensor<2x2xf32>
    %W2 = stablehlo.constant dense<[[0.0, 1.0], [-1.0, 0.0]]> : tensor<2x2xf32>
    %t = stablehlo.constant dense<true> : tensor<i1>
    %f = stablehlo.constant dense<false> : tensor<i1>

    %d_true = func.call @while_uniform_if_dot(%A, %W1, %W2, %t) : (tensor<4x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>, tensor<i1>) -> tensor<4x2xf32>
    check.expect_almost_eq_const %d_true, dense<[[2.0, 6.0], [6.0, 12.0], [-2.0, 1.5], [4.0, -9.0]]> : tensor<4x2xf32>

    %d_false = func.call @while_uniform_if_dot(%A, %W1, %W2, %f) : (tensor<4x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>, tensor<i1>) -> tensor<4x2xf32>
    check.expect_almost_eq_const %d_false, dense<[[-2.0, 1.0], [-4.0, 3.0], [-0.5, -1.0], [3.0, 2.0]]> : tensor<4x2xf32>

    %thresh = stablehlo.constant dense<[1.0, -1.0, 2.0, -0.5]> : tensor<4xf32>
    %m:2 = func.call @while_varying_if_multi_result(%A, %thresh) : (tensor<4x2xf32>, tensor<4xf32>) -> (tensor<4x2xf32>, tensor<4x2xf32>)
    check.expect_almost_eq_const %m#0, dense<[[2.0, 4.0], [-3.0, -4.0], [-2.0, 1.0], [-2.0, 3.0]]> : tensor<4x2xf32>
    check.expect_almost_eq_const %m#1, dense<[[3.0, 4.0], [-2.0, -1.0], [1.0, 2.5], [-3.0, -8.0]]> : tensor<4x2xf32>

    %xv = stablehlo.constant dense<[1.0, 2.0, 3.0, 4.0]> : tensor<4xf32>
    %pt = func.call @parallel_while_if(%xv, %t) : (tensor<4xf32>, tensor<i1>) -> tensor<4xf32>
    check.expect_almost_eq_const %pt, dense<[13.0, 16.0, -9.0, -12.0]> : tensor<4xf32>
    %pf = func.call @parallel_while_if(%xv, %f) : (tensor<4xf32>, tensor<i1>) -> tensor<4xf32>
    check.expect_almost_eq_const %pf, dense<[12.0, 13.0, -4.0, -5.0]> : tensor<4xf32>

    %num = stablehlo.constant dense<[20, 99, 28, 77]> : tensor<4xi32>
    %den = stablehlo.constant dense<[4, 0, 7, 0]> : tensor<4xi32>
    %q = func.call @while_guarded_int_div(%num, %den) : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
    check.expect_eq_const %q, dense<[5, -1, 4, -1]> : tensor<4xi32>
    return
  }
}
