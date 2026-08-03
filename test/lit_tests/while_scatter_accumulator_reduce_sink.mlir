// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(enzyme-hlo-generate-td{patterns=while_scatter_accumulator_reduce_sink},transform-interpreter,enzyme-hlo-remove-transform)" %s | FileCheck %s

// The shape reverse-mode AD leaves behind: a loop that scatter-accumulates two
// 8x4 gradient accumulators, followed by an elementwise epilogue over them and
// a loop-invariant tensor, followed by a sum reduction over the non-scattered
// dimension. Row i of every accumulator is written only at iteration i, so the
// whole epilogue can be evaluated a row at a time inside the loop.
func.func @sink_epilogue(%t: tensor<8x4xf32>) -> tensor<8xf32> {
  %c0 = stablehlo.constant dense<0> : tensor<i64>
  %c1 = stablehlo.constant dense<1> : tensor<i64>
  %c8 = stablehlo.constant dense<8> : tensor<i64>
  %zero = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %wide = stablehlo.constant dense<0.000000e+00> : tensor<8x4xf32>
  %row = stablehlo.constant dense<2.000000e+00> : tensor<1x4xf32>

  %w:3 = stablehlo.while(%iv = %c0, %a = %wide, %b = %wide) : tensor<i64>, tensor<8x4xf32>, tensor<8x4xf32>
   cond {
    %p = stablehlo.compare LT, %iv, %c8 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %p : tensor<i1>
  } do {
    %next = stablehlo.add %iv, %c1 : tensor<i64>
    %sa = stablehlo.dynamic_update_slice %wide, %row, %iv, %c0 : (tensor<8x4xf32>, tensor<1x4xf32>, tensor<i64>, tensor<i64>) -> tensor<8x4xf32>
    %na = stablehlo.add %a, %sa : tensor<8x4xf32>
    %sb = stablehlo.dynamic_update_slice %wide, %row, %iv, %c0 : (tensor<8x4xf32>, tensor<1x4xf32>, tensor<i64>, tensor<i64>) -> tensor<8x4xf32>
    %nb = stablehlo.add %b, %sb : tensor<8x4xf32>
    stablehlo.return %next, %na, %nb : tensor<i64>, tensor<8x4xf32>, tensor<8x4xf32>
  }

  %e0 = stablehlo.subtract %w#1, %w#2 : tensor<8x4xf32>
  %e1 = stablehlo.multiply %e0, %t : tensor<8x4xf32>
  %out = stablehlo.reduce(%e1 init: %zero) applies stablehlo.add across dimensions = [1] : (tensor<8x4xf32>, tensor<f32>) -> tensor<8xf32>
  return %out : tensor<8xf32>
}

// The epilogue moves inside the loop and runs on 1x4 rows; the loop grows a
// tensor<8xf32> carry holding the reduced result.
// CHECK-LABEL: func.func @sink_epilogue
// CHECK: stablehlo.while
// CHECK-SAME: tensor<8xf32>
// CHECK: stablehlo.dynamic_slice %arg0
// CHECK-SAME: sizes = [1, 4]
// CHECK: stablehlo.subtract %{{.+}}, %{{.+}} : tensor<1x4xf32>
// CHECK: stablehlo.multiply %{{.+}}, %{{.+}} : tensor<1x4xf32>
// CHECK: stablehlo.reduce
// CHECK-SAME: (tensor<1x4xf32>, tensor<f32>) -> tensor<1xf32>
// CHECK-NOT: stablehlo.subtract %{{.+}}, %{{.+}} : tensor<8x4xf32>

// A reversed row index (start 8-1, step -1) still covers every row exactly
// once, which is the form reverse-mode AD actually emits.
func.func @sink_reverse_index(%t: tensor<8x4xf32>) -> tensor<8xf32> {
  %c0 = stablehlo.constant dense<0> : tensor<i64>
  %c1 = stablehlo.constant dense<1> : tensor<i64>
  %c7 = stablehlo.constant dense<7> : tensor<i64>
  %c8 = stablehlo.constant dense<8> : tensor<i64>
  %zero = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %wide = stablehlo.constant dense<0.000000e+00> : tensor<8x4xf32>
  %row = stablehlo.constant dense<2.000000e+00> : tensor<1x4xf32>

  %w:3 = stablehlo.while(%iv = %c0, %a = %wide, %ri = %c7) : tensor<i64>, tensor<8x4xf32>, tensor<i64>
   cond {
    %p = stablehlo.compare LT, %iv, %c8 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %p : tensor<i1>
  } do {
    %next = stablehlo.add %iv, %c1 : tensor<i64>
    %sa = stablehlo.dynamic_update_slice %wide, %row, %ri, %c0 : (tensor<8x4xf32>, tensor<1x4xf32>, tensor<i64>, tensor<i64>) -> tensor<8x4xf32>
    %na = stablehlo.add %a, %sa : tensor<8x4xf32>
    %rnext = stablehlo.subtract %ri, %c1 : tensor<i64>
    stablehlo.return %next, %na, %rnext : tensor<i64>, tensor<8x4xf32>, tensor<i64>
  }

  %e0 = stablehlo.multiply %w#1, %t : tensor<8x4xf32>
  %out = stablehlo.reduce(%e0 init: %zero) applies stablehlo.add across dimensions = [1] : (tensor<8x4xf32>, tensor<f32>) -> tensor<8xf32>
  return %out : tensor<8xf32>
}

// CHECK-LABEL: func.func @sink_reverse_index
// CHECK: stablehlo.multiply %{{.+}}, %{{.+}} : tensor<1x4xf32>
// CHECK-NOT: stablehlo.multiply %{{.+}}, %{{.+}} : tensor<8x4xf32>

// The loop runs fewer times than there are rows, so rows 4..7 are never
// written. Their epilogue value is f(0, t[i]) which is not generally zero, so
// the sink would be unsound and must not fire.
func.func @no_sink_partial_coverage(%t: tensor<8x4xf32>) -> tensor<8xf32> {
  %c0 = stablehlo.constant dense<0> : tensor<i64>
  %c1 = stablehlo.constant dense<1> : tensor<i64>
  %c4 = stablehlo.constant dense<4> : tensor<i64>
  %zero = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %wide = stablehlo.constant dense<0.000000e+00> : tensor<8x4xf32>
  %row = stablehlo.constant dense<2.000000e+00> : tensor<1x4xf32>

  %w:2 = stablehlo.while(%iv = %c0, %a = %wide) : tensor<i64>, tensor<8x4xf32>
   cond {
    %p = stablehlo.compare LT, %iv, %c4 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %p : tensor<i1>
  } do {
    %next = stablehlo.add %iv, %c1 : tensor<i64>
    %sa = stablehlo.dynamic_update_slice %wide, %row, %iv, %c0 : (tensor<8x4xf32>, tensor<1x4xf32>, tensor<i64>, tensor<i64>) -> tensor<8x4xf32>
    %na = stablehlo.add %a, %sa : tensor<8x4xf32>
    stablehlo.return %next, %na : tensor<i64>, tensor<8x4xf32>
  }

  %e0 = stablehlo.multiply %w#1, %t : tensor<8x4xf32>
  %out = stablehlo.reduce(%e0 init: %zero) applies stablehlo.add across dimensions = [1] : (tensor<8x4xf32>, tensor<f32>) -> tensor<8xf32>
  return %out : tensor<8xf32>
}

// CHECK-LABEL: func.func @no_sink_partial_coverage
// CHECK: stablehlo.multiply %{{.+}}, %{{.+}} : tensor<8x4xf32>

// The epilogue also feeds something that is not a sibling reduction, so the
// wide values stay live and sinking would only add work.
func.func @no_sink_wide_escape(%t: tensor<8x4xf32>) -> (tensor<8xf32>, tensor<8x4xf32>) {
  %c0 = stablehlo.constant dense<0> : tensor<i64>
  %c1 = stablehlo.constant dense<1> : tensor<i64>
  %c8 = stablehlo.constant dense<8> : tensor<i64>
  %zero = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %wide = stablehlo.constant dense<0.000000e+00> : tensor<8x4xf32>
  %row = stablehlo.constant dense<2.000000e+00> : tensor<1x4xf32>

  %w:2 = stablehlo.while(%iv = %c0, %a = %wide) : tensor<i64>, tensor<8x4xf32>
   cond {
    %p = stablehlo.compare LT, %iv, %c8 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %p : tensor<i1>
  } do {
    %next = stablehlo.add %iv, %c1 : tensor<i64>
    %sa = stablehlo.dynamic_update_slice %wide, %row, %iv, %c0 : (tensor<8x4xf32>, tensor<1x4xf32>, tensor<i64>, tensor<i64>) -> tensor<8x4xf32>
    %na = stablehlo.add %a, %sa : tensor<8x4xf32>
    stablehlo.return %next, %na : tensor<i64>, tensor<8x4xf32>
  }

  %e0 = stablehlo.multiply %w#1, %t : tensor<8x4xf32>
  %out = stablehlo.reduce(%e0 init: %zero) applies stablehlo.add across dimensions = [1] : (tensor<8x4xf32>, tensor<f32>) -> tensor<8xf32>
  return %out, %e0 : tensor<8xf32>, tensor<8x4xf32>
}

// CHECK-LABEL: func.func @no_sink_wide_escape
// CHECK: stablehlo.multiply %{{.+}}, %{{.+}} : tensor<8x4xf32>

// Reducing the scattered dimension itself is not row-separable.
func.func @no_sink_reduces_scatter_dim(%t: tensor<8x4xf32>) -> tensor<4xf32> {
  %c0 = stablehlo.constant dense<0> : tensor<i64>
  %c1 = stablehlo.constant dense<1> : tensor<i64>
  %c8 = stablehlo.constant dense<8> : tensor<i64>
  %zero = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %wide = stablehlo.constant dense<0.000000e+00> : tensor<8x4xf32>
  %row = stablehlo.constant dense<2.000000e+00> : tensor<1x4xf32>

  %w:2 = stablehlo.while(%iv = %c0, %a = %wide) : tensor<i64>, tensor<8x4xf32>
   cond {
    %p = stablehlo.compare LT, %iv, %c8 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %p : tensor<i1>
  } do {
    %next = stablehlo.add %iv, %c1 : tensor<i64>
    %sa = stablehlo.dynamic_update_slice %wide, %row, %iv, %c0 : (tensor<8x4xf32>, tensor<1x4xf32>, tensor<i64>, tensor<i64>) -> tensor<8x4xf32>
    %na = stablehlo.add %a, %sa : tensor<8x4xf32>
    stablehlo.return %next, %na : tensor<i64>, tensor<8x4xf32>
  }

  %e0 = stablehlo.multiply %w#1, %t : tensor<8x4xf32>
  %out = stablehlo.reduce(%e0 init: %zero) applies stablehlo.add across dimensions = [0] : (tensor<8x4xf32>, tensor<f32>) -> tensor<4xf32>
  return %out : tensor<4xf32>
}

// CHECK-LABEL: func.func @no_sink_reduces_scatter_dim
// CHECK: stablehlo.multiply %{{.+}}, %{{.+}} : tensor<8x4xf32>
