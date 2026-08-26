// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(enzyme-hlo-generate-td{patterns=while_op_induction_replacement},transform-interpreter,enzyme-hlo-remove-transform,canonicalize)" %s | FileCheck %s

// A counter that walks backwards is as redundant as one that walks forwards:
// %ri is 7, 6, ... 0 while %iv is 0, 1, ... 7, so uses of %ri can be rewritten
// as an expression of %iv and the carry dropped.
func.func @backward_counter(%row: tensor<1x4xf32>) -> tensor<8x4xf32> {
  %c0 = stablehlo.constant dense<0> : tensor<i64>
  %c1 = stablehlo.constant dense<1> : tensor<i64>
  %c7 = stablehlo.constant dense<7> : tensor<i64>
  %c8 = stablehlo.constant dense<8> : tensor<i64>
  %wide = stablehlo.constant dense<0.000000e+00> : tensor<8x4xf32>

  %w:3 = stablehlo.while(%iv = %c0, %a = %wide, %ri = %c7) : tensor<i64>, tensor<8x4xf32>, tensor<i64>
   cond {
    %p = stablehlo.compare LT, %iv, %c8 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %p : tensor<i1>
  } do {
    %next = stablehlo.add %iv, %c1 : tensor<i64>
    %sa = stablehlo.dynamic_update_slice %wide, %row, %ri, %c0 : (tensor<8x4xf32>, tensor<1x4xf32>, tensor<i64>, tensor<i64>) -> tensor<8x4xf32>
    %rnext = stablehlo.subtract %ri, %c1 : tensor<i64>
    stablehlo.return %next, %sa, %rnext : tensor<i64>, tensor<8x4xf32>, tensor<i64>
  }
  return %w#1 : tensor<8x4xf32>
}

// The scatter index is now derived from the loop counter rather than read from
// a carry, and no negate is emitted -- the sign is folded into the constant.
// CHECK-LABEL: func.func @backward_counter
// CHECK-NOT: stablehlo.negate
// CHECK: stablehlo.while
// CHECK: stablehlo.dynamic_update_slice

// A forward counter keeps working exactly as before.
func.func @forward_counter(%row: tensor<1x4xf32>) -> tensor<8x4xf32> {
  %c0 = stablehlo.constant dense<0> : tensor<i64>
  %c1 = stablehlo.constant dense<1> : tensor<i64>
  %c8 = stablehlo.constant dense<8> : tensor<i64>
  %wide = stablehlo.constant dense<0.000000e+00> : tensor<8x4xf32>

  %w:3 = stablehlo.while(%iv = %c0, %a = %wide, %ri = %c0) : tensor<i64>, tensor<8x4xf32>, tensor<i64>
   cond {
    %p = stablehlo.compare LT, %iv, %c8 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %p : tensor<i1>
  } do {
    %next = stablehlo.add %iv, %c1 : tensor<i64>
    %sa = stablehlo.dynamic_update_slice %wide, %row, %ri, %c0 : (tensor<8x4xf32>, tensor<1x4xf32>, tensor<i64>, tensor<i64>) -> tensor<8x4xf32>
    %rnext = stablehlo.add %ri, %c1 : tensor<i64>
    stablehlo.return %next, %sa, %rnext : tensor<i64>, tensor<8x4xf32>, tensor<i64>
  }
  return %w#1 : tensor<8x4xf32>
}

// CHECK-LABEL: func.func @forward_counter
// CHECK: stablehlo.while
// CHECK: stablehlo.dynamic_update_slice

// `step - iter_arg` is not a counter -- it alternates rather than stepping --
// so it must be left alone.
func.func @not_a_counter(%row: tensor<1x4xf32>) -> tensor<8x4xf32> {
  %c0 = stablehlo.constant dense<0> : tensor<i64>
  %c1 = stablehlo.constant dense<1> : tensor<i64>
  %c7 = stablehlo.constant dense<7> : tensor<i64>
  %c8 = stablehlo.constant dense<8> : tensor<i64>
  %wide = stablehlo.constant dense<0.000000e+00> : tensor<8x4xf32>

  %w:3 = stablehlo.while(%iv = %c0, %a = %wide, %ri = %c7) : tensor<i64>, tensor<8x4xf32>, tensor<i64>
   cond {
    %p = stablehlo.compare LT, %iv, %c8 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %p : tensor<i1>
  } do {
    %next = stablehlo.add %iv, %c1 : tensor<i64>
    %sa = stablehlo.dynamic_update_slice %wide, %row, %ri, %c0 : (tensor<8x4xf32>, tensor<1x4xf32>, tensor<i64>, tensor<i64>) -> tensor<8x4xf32>
    %rnext = stablehlo.subtract %c7, %ri : tensor<i64>
    stablehlo.return %next, %sa, %rnext : tensor<i64>, tensor<8x4xf32>, tensor<i64>
  }
  return %w#1 : tensor<8x4xf32>
}

// CHECK-LABEL: func.func @not_a_counter
// CHECK: stablehlo.subtract %{{.+}}, %iterArg
