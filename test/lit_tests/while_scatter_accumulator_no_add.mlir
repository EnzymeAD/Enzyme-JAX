// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(enzyme-hlo-generate-td{patterns=while_scatter_accumulator_no_add},transform-interpreter,enzyme-hlo-remove-transform)" %s | FileCheck %s

// A gradient accumulator that scatters one row per iteration. The row index is
// the induction variable, so it is distinct every iteration and the accumulator
// starts at zero: the add is always an add to zero and collapses to a plain
// read-modify-write of the row.
func.func @forward_index(%row: tensor<1x4xf32>) -> tensor<8x4xf32> {
  %c0 = stablehlo.constant dense<0> : tensor<i64>
  %c1 = stablehlo.constant dense<1> : tensor<i64>
  %c8 = stablehlo.constant dense<8> : tensor<i64>
  %wide = stablehlo.constant dense<0.000000e+00> : tensor<8x4xf32>

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
  return %w#1 : tensor<8x4xf32>
}

// CHECK-LABEL: func.func @forward_index
// CHECK: stablehlo.dynamic_update_slice %iterArg{{[_0-9]*}}, %arg0
// CHECK-NOT: stablehlo.add %{{.+}}, %{{.+}} : tensor<8x4xf32>

// The reverse index derived from the induction variable, which is the form
// reverse-mode AD emits: numIters-1 down to 0, distinct and in bounds.
func.func @reverse_index(%row: tensor<1x4xf32>) -> tensor<8x4xf32> {
  %c0 = stablehlo.constant dense<0> : tensor<i64>
  %c1 = stablehlo.constant dense<1> : tensor<i64>
  %c7 = stablehlo.constant dense<7> : tensor<i64>
  %c8 = stablehlo.constant dense<8> : tensor<i64>
  %wide = stablehlo.constant dense<0.000000e+00> : tensor<8x4xf32>

  %w:2 = stablehlo.while(%iv = %c0, %a = %wide) : tensor<i64>, tensor<8x4xf32>
   cond {
    %p = stablehlo.compare LT, %iv, %c8 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %p : tensor<i1>
  } do {
    %ri = stablehlo.subtract %c7, %iv : tensor<i64>
    %next = stablehlo.add %iv, %c1 : tensor<i64>
    %sa = stablehlo.dynamic_update_slice %wide, %row, %ri, %c0 : (tensor<8x4xf32>, tensor<1x4xf32>, tensor<i64>, tensor<i64>) -> tensor<8x4xf32>
    %na = stablehlo.add %a, %sa : tensor<8x4xf32>
    stablehlo.return %next, %na : tensor<i64>, tensor<8x4xf32>
  }
  return %w#1 : tensor<8x4xf32>
}

// CHECK-LABEL: func.func @reverse_index
// CHECK: stablehlo.dynamic_update_slice %iterArg{{[_0-9]*}}, %arg0
// CHECK-NOT: stablehlo.add %{{.+}}, %{{.+}} : tensor<8x4xf32>

// Partial coverage is fine here, unlike for the epilogue sink: rows 4..7 are
// never written and keep their zero under either form.
func.func @partial_coverage_still_ok(%row: tensor<1x4xf32>) -> tensor<8x4xf32> {
  %c0 = stablehlo.constant dense<0> : tensor<i64>
  %c1 = stablehlo.constant dense<1> : tensor<i64>
  %c4 = stablehlo.constant dense<4> : tensor<i64>
  %wide = stablehlo.constant dense<0.000000e+00> : tensor<8x4xf32>

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
  return %w#1 : tensor<8x4xf32>
}

// CHECK-LABEL: func.func @partial_coverage_still_ok
// CHECK: stablehlo.dynamic_update_slice %iterArg{{[_0-9]*}}, %arg0
// CHECK-NOT: stablehlo.add %{{.+}}, %{{.+}} : tensor<8x4xf32>

// The index runs past the end of the accumulator. dynamic_update_slice clamps,
// so iterations 8..15 all write row 7 and genuinely accumulate there.
func.func @no_fold_out_of_bounds(%row: tensor<1x4xf32>) -> tensor<8x4xf32> {
  %c0 = stablehlo.constant dense<0> : tensor<i64>
  %c1 = stablehlo.constant dense<1> : tensor<i64>
  %c16 = stablehlo.constant dense<16> : tensor<i64>
  %wide = stablehlo.constant dense<0.000000e+00> : tensor<8x4xf32>

  %w:2 = stablehlo.while(%iv = %c0, %a = %wide) : tensor<i64>, tensor<8x4xf32>
   cond {
    %p = stablehlo.compare LT, %iv, %c16 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %p : tensor<i1>
  } do {
    %next = stablehlo.add %iv, %c1 : tensor<i64>
    %sa = stablehlo.dynamic_update_slice %wide, %row, %iv, %c0 : (tensor<8x4xf32>, tensor<1x4xf32>, tensor<i64>, tensor<i64>) -> tensor<8x4xf32>
    %na = stablehlo.add %a, %sa : tensor<8x4xf32>
    stablehlo.return %next, %na : tensor<i64>, tensor<8x4xf32>
  }
  return %w#1 : tensor<8x4xf32>
}

// CHECK-LABEL: func.func @no_fold_out_of_bounds
// CHECK: stablehlo.add %{{.+}}, %{{.+}} : tensor<8x4xf32>

// A loop-invariant index writes the same row every iteration, so the adds
// genuinely accumulate.
func.func @no_fold_constant_index(%row: tensor<1x4xf32>) -> tensor<8x4xf32> {
  %c0 = stablehlo.constant dense<0> : tensor<i64>
  %c1 = stablehlo.constant dense<1> : tensor<i64>
  %c3 = stablehlo.constant dense<3> : tensor<i64>
  %c8 = stablehlo.constant dense<8> : tensor<i64>
  %wide = stablehlo.constant dense<0.000000e+00> : tensor<8x4xf32>

  %w:2 = stablehlo.while(%iv = %c0, %a = %wide) : tensor<i64>, tensor<8x4xf32>
   cond {
    %p = stablehlo.compare LT, %iv, %c8 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %p : tensor<i1>
  } do {
    %next = stablehlo.add %iv, %c1 : tensor<i64>
    %sa = stablehlo.dynamic_update_slice %wide, %row, %c3, %c0 : (tensor<8x4xf32>, tensor<1x4xf32>, tensor<i64>, tensor<i64>) -> tensor<8x4xf32>
    %na = stablehlo.add %a, %sa : tensor<8x4xf32>
    stablehlo.return %next, %na : tensor<i64>, tensor<8x4xf32>
  }
  return %w#1 : tensor<8x4xf32>
}

// CHECK-LABEL: func.func @no_fold_constant_index
// CHECK: stablehlo.add %{{.+}}, %{{.+}} : tensor<8x4xf32>

// A non-zero initialiser means the first write is not an add to zero.
func.func @no_fold_nonzero_init(%row: tensor<1x4xf32>, %init: tensor<8x4xf32>) -> tensor<8x4xf32> {
  %c0 = stablehlo.constant dense<0> : tensor<i64>
  %c1 = stablehlo.constant dense<1> : tensor<i64>
  %c8 = stablehlo.constant dense<8> : tensor<i64>
  %wide = stablehlo.constant dense<0.000000e+00> : tensor<8x4xf32>

  %w:2 = stablehlo.while(%iv = %c0, %a = %init) : tensor<i64>, tensor<8x4xf32>
   cond {
    %p = stablehlo.compare LT, %iv, %c8 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %p : tensor<i1>
  } do {
    %next = stablehlo.add %iv, %c1 : tensor<i64>
    %sa = stablehlo.dynamic_update_slice %wide, %row, %iv, %c0 : (tensor<8x4xf32>, tensor<1x4xf32>, tensor<i64>, tensor<i64>) -> tensor<8x4xf32>
    %na = stablehlo.add %a, %sa : tensor<8x4xf32>
    stablehlo.return %next, %na : tensor<i64>, tensor<8x4xf32>
  }
  return %w#1 : tensor<8x4xf32>
}

// CHECK-LABEL: func.func @no_fold_nonzero_init
// CHECK: stablehlo.add %{{.+}}, %{{.+}} : tensor<8x4xf32>

// The accumulator is read inside the loop as well, so the rows it holds at that
// point are observable and must not change.
func.func @no_fold_extra_reader(%row: tensor<1x4xf32>) -> (tensor<8x4xf32>, tensor<8x4xf32>) {
  %c0 = stablehlo.constant dense<0> : tensor<i64>
  %c1 = stablehlo.constant dense<1> : tensor<i64>
  %c8 = stablehlo.constant dense<8> : tensor<i64>
  %wide = stablehlo.constant dense<0.000000e+00> : tensor<8x4xf32>

  %w:3 = stablehlo.while(%iv = %c0, %a = %wide, %b = %wide) : tensor<i64>, tensor<8x4xf32>, tensor<8x4xf32>
   cond {
    %p = stablehlo.compare LT, %iv, %c8 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %p : tensor<i1>
  } do {
    %next = stablehlo.add %iv, %c1 : tensor<i64>
    %sa = stablehlo.dynamic_update_slice %wide, %row, %iv, %c0 : (tensor<8x4xf32>, tensor<1x4xf32>, tensor<i64>, tensor<i64>) -> tensor<8x4xf32>
    %na = stablehlo.add %a, %sa : tensor<8x4xf32>
    %obs = stablehlo.multiply %a, %a : tensor<8x4xf32>
    stablehlo.return %next, %na, %obs : tensor<i64>, tensor<8x4xf32>, tensor<8x4xf32>
  }
  return %w#1, %w#2 : tensor<8x4xf32>, tensor<8x4xf32>
}

// CHECK-LABEL: func.func @no_fold_extra_reader
// CHECK: stablehlo.add %{{.+}}, %{{.+}} : tensor<8x4xf32>
