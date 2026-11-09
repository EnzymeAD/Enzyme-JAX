// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(enzyme-hlo-opt{no_signed_wrap=true})" | FileCheck %s --check-prefixes=CHECK,NSW
// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt | FileCheck %s --check-prefixes=CHECK,WRAP

// [Pattern: AbsPositiveSimplify]
// abs(x * x) -> x * x requires x * x >= 0 (NSW or bounded x).

// CHECK-LABEL: func.func @abs_of_square_unbounded(
// CHECK-SAME:    %[[X:.+]]: tensor<4xi32>)
func.func @abs_of_square_unbounded(%x: tensor<4xi32>) -> tensor<4xi32> {
  // NSW-NEXT:  %[[SQ:.+]] = stablehlo.multiply %[[X]], %[[X]]
  // NSW-NEXT:  return %[[SQ]]
  // WRAP-NEXT: %[[SQ:.+]] = stablehlo.multiply %[[X]], %[[X]]
  // WRAP-NEXT: %[[ABS:.+]] = stablehlo.abs %[[SQ]]
  // WRAP-NEXT: return %[[ABS]]
  %sq = stablehlo.multiply %x, %x : tensor<4xi32>
  %abs = stablehlo.abs %sq : tensor<4xi32>
  return %abs : tensor<4xi32>
}

// CHECK-LABEL: func.func @abs_of_square_bounded(
// CHECK-SAME:    %[[X:.+]]: tensor<4xi32> {enzymexla.bounds = [-100, 100]})
func.func @abs_of_square_bounded(%x: tensor<4xi32> {enzymexla.bounds = [-100, 100]}) -> tensor<4xi32> {
  // CHECK-NEXT: %[[SQ:.+]] = stablehlo.multiply %[[X]], %[[X]]
  // CHECK-NEXT: return %[[SQ]]
  %sq = stablehlo.multiply %x, %x : tensor<4xi32>
  %abs = stablehlo.abs %sq : tensor<4xi32>
  return %abs : tensor<4xi32>
}

// [Pattern: NoNanCompareAbs]
// Ordering folds against 0 fail at INT_MIN unless NSW is enabled; EQ/NE hold unconditionally.

// CHECK-LABEL: func.func @abs_le_zero_unbounded(
// CHECK-SAME:    %[[X:.+]]: tensor<4xi32>)
func.func @abs_le_zero_unbounded(%x: tensor<4xi32>) -> tensor<4xi1> {
  // NSW:  stablehlo.compare EQ, %[[X]]
  // WRAP: %[[ABS:.+]] = stablehlo.abs %[[X]]
  // WRAP: stablehlo.compare LE, %[[ABS]]
  %c0 = stablehlo.constant dense<0> : tensor<4xi32>
  %abs = stablehlo.abs %x : tensor<4xi32>
  %cmp = stablehlo.compare LE, %abs, %c0, SIGNED : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi1>
  return %cmp : tensor<4xi1>
}

// CHECK-LABEL: func.func @abs_eq_zero_unbounded(
// CHECK-SAME:    %[[X:.+]]: tensor<4xi32>)
func.func @abs_eq_zero_unbounded(%x: tensor<4xi32>) -> tensor<4xi1> {
  // CHECK-NOT: stablehlo.abs
  // CHECK:     stablehlo.compare EQ, %[[X]]
  %c0 = stablehlo.constant dense<0> : tensor<4xi32>
  %abs = stablehlo.abs %x : tensor<4xi32>
  %cmp = stablehlo.compare EQ, %abs, %c0, SIGNED : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi1>
  return %cmp : tensor<4xi1>
}

// [Pattern: CompareMul]
// Odd constant factors are invertible mod 2^w for EQ/NE; even factors and
// ordering comparisons require no signed wrap.

// CHECK-LABEL: func.func @compare_mul_lt_unbounded(
// CHECK-SAME:    %[[X:.+]]: tensor<4xi8>)
func.func @compare_mul_lt_unbounded(%x: tensor<4xi8>) -> tensor<4xi1> {
  // NSW-NOT: stablehlo.multiply
  // NSW:     stablehlo.compare LT, %[[X]]
  // WRAP:    %[[MUL:.+]] = stablehlo.multiply %[[X]]
  // WRAP:    stablehlo.compare LT, %[[MUL]]
  %c0 = stablehlo.constant dense<0> : tensor<4xi8>
  %c4 = stablehlo.constant dense<4> : tensor<4xi8>
  %mul = stablehlo.multiply %x, %c4 : tensor<4xi8>
  %cmp = stablehlo.compare LT, %mul, %c0, SIGNED : (tensor<4xi8>, tensor<4xi8>) -> tensor<4xi1>
  return %cmp : tensor<4xi1>
}

// CHECK-LABEL: func.func @compare_mul_lt_bounded(
// CHECK-SAME:    %[[X:.+]]: tensor<4xi8> {enzymexla.bounds = [-10, 10]})
func.func @compare_mul_lt_bounded(%x: tensor<4xi8> {enzymexla.bounds = [-10, 10]}) -> tensor<4xi1> {
  // CHECK-NOT: stablehlo.multiply
  // CHECK:     stablehlo.compare LT, %[[X]]
  %c0 = stablehlo.constant dense<0> : tensor<4xi8>
  %c4 = stablehlo.constant dense<4> : tensor<4xi8>
  %mul = stablehlo.multiply %x, %c4 : tensor<4xi8>
  %cmp = stablehlo.compare LT, %mul, %c0, SIGNED : (tensor<4xi8>, tensor<4xi8>) -> tensor<4xi1>
  return %cmp : tensor<4xi1>
}

// CHECK-LABEL: func.func @compare_mul_even_eq_unbounded(
// CHECK-SAME:    %[[X:.+]]: tensor<4xi8>)
func.func @compare_mul_even_eq_unbounded(%x: tensor<4xi8>) -> tensor<4xi1> {
  // NSW-NOT: stablehlo.multiply
  // NSW:     stablehlo.compare EQ, %[[X]]
  // WRAP:    %[[MUL:.+]] = stablehlo.multiply %[[X]]
  // WRAP:    stablehlo.compare EQ, %[[MUL]]
  %c0 = stablehlo.constant dense<0> : tensor<4xi8>
  %c4 = stablehlo.constant dense<4> : tensor<4xi8>
  %mul = stablehlo.multiply %x, %c4 : tensor<4xi8>
  %cmp = stablehlo.compare EQ, %mul, %c0, SIGNED : (tensor<4xi8>, tensor<4xi8>) -> tensor<4xi1>
  return %cmp : tensor<4xi1>
}

// CHECK-LABEL: func.func @compare_mul_odd_eq_unbounded(
// CHECK-SAME:    %[[X:.+]]: tensor<4xi8>)
func.func @compare_mul_odd_eq_unbounded(%x: tensor<4xi8>) -> tensor<4xi1> {
  // CHECK-NOT: stablehlo.multiply
  // CHECK:     stablehlo.compare EQ, %[[X]]
  %c0 = stablehlo.constant dense<0> : tensor<4xi8>
  %c3 = stablehlo.constant dense<3> : tensor<4xi8>
  %mul = stablehlo.multiply %x, %c3 : tensor<4xi8>
  %cmp = stablehlo.compare EQ, %mul, %c0, SIGNED : (tensor<4xi8>, tensor<4xi8>) -> tensor<4xi1>
  return %cmp : tensor<4xi1>
}

// [Pattern: CompareNegateConstSimplify]

// CHECK-LABEL: func.func @compare_neg_neg_unbounded(
// CHECK-SAME:    %[[X:.+]]: tensor<4xi32>, %[[Y:.+]]: tensor<4xi32>)
func.func @compare_neg_neg_unbounded(%x: tensor<4xi32>, %y: tensor<4xi32>) -> tensor<4xi1> {
  // NSW-NOT: stablehlo.negate
  // NSW:     stablehlo.compare LT, %[[Y]], %[[X]]
  // WRAP:    %[[NX:.+]] = stablehlo.negate %[[X]]
  // WRAP:    %[[NY:.+]] = stablehlo.negate %[[Y]]
  // WRAP:    stablehlo.compare LT, %[[NX]], %[[NY]]
  %nx = stablehlo.negate %x : tensor<4xi32>
  %ny = stablehlo.negate %y : tensor<4xi32>
  %cmp = stablehlo.compare LT, %nx, %ny, SIGNED : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi1>
  return %cmp : tensor<4xi1>
}

// CHECK-LABEL: func.func @compare_neg_neg_bounded(
// CHECK-SAME:    %[[X:.+]]: tensor<4xi32> {enzymexla.bounds = [-100, 100]},
// CHECK-SAME:    %[[Y:.+]]: tensor<4xi32> {enzymexla.bounds = [-100, 100]})
func.func @compare_neg_neg_bounded(%x: tensor<4xi32> {enzymexla.bounds = [-100, 100]}, %y: tensor<4xi32> {enzymexla.bounds = [-100, 100]}) -> tensor<4xi1> {
  // CHECK-NOT: stablehlo.negate
  // CHECK:     stablehlo.compare LT, %[[Y]], %[[X]]
  %nx = stablehlo.negate %x : tensor<4xi32>
  %ny = stablehlo.negate %y : tensor<4xi32>
  %cmp = stablehlo.compare LT, %nx, %ny, SIGNED : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi1>
  return %cmp : tensor<4xi1>
}

// CHECK-LABEL: func.func @compare_neg_const_unbounded(
// CHECK-SAME:    %[[X:.+]]: tensor<4xi32>, %[[Y:.+]]: tensor<4xi32>)
func.func @compare_neg_const_unbounded(%x: tensor<4xi32>, %y: tensor<4xi32>) -> tensor<4xi1> {
  // NSW-NOT: stablehlo.negate
  // NSW:     %[[XOR:.+]] = stablehlo.xor %[[X]], %[[Y]]
  // NSW:     stablehlo.compare GT, %[[XOR]]
  // WRAP:    %[[XOR:.+]] = stablehlo.xor %[[X]], %[[Y]]
  // WRAP:    %[[NEG:.+]] = stablehlo.negate %[[XOR]]
  // WRAP:    stablehlo.compare LT, %[[NEG]]
  %xor = stablehlo.xor %x, %y : tensor<4xi32>
  %neg = stablehlo.negate %xor : tensor<4xi32>
  %c5 = stablehlo.constant dense<5> : tensor<4xi32>
  %cmp = stablehlo.compare LT, %neg, %c5, SIGNED : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi1>
  return %cmp : tensor<4xi1>
}

// Synthesizing -(-128) in i8 overflows at compile time even when the input
// expression (-0 > -128) does not overflow, so this must not fold even under NSW.
// CHECK-LABEL: func.func @no_fold_compare_neg_int_min_const(
// CHECK-SAME:    %[[X:.+]]: tensor<4xi8>, %[[Y:.+]]: tensor<4xi8>)
func.func @no_fold_compare_neg_int_min_const(%x: tensor<4xi8>, %y: tensor<4xi8>) -> tensor<4xi1> {
  // CHECK: %[[XOR:.+]] = stablehlo.xor %[[X]], %[[Y]]
  // CHECK: %[[NEG:.+]] = stablehlo.negate %[[XOR]]
  // CHECK: stablehlo.compare GT, %[[NEG]]
  %xor = stablehlo.xor %x, %y : tensor<4xi8>
  %neg = stablehlo.negate %xor : tensor<4xi8>
  %min = stablehlo.constant dense<-128> : tensor<4xi8>
  %cmp = stablehlo.compare GT, %neg, %min, SIGNED : (tensor<4xi8>, tensor<4xi8>) -> tensor<4xi1>
  return %cmp : tensor<4xi1>
}

// [Pattern: CompareSubtractConstSimplify]

// CHECK-LABEL: func.func @compare_sub_const_unbounded(
// CHECK-SAME:    %[[X:.+]]: tensor<4xi32>)
func.func @compare_sub_const_unbounded(%x: tensor<4xi32>) -> tensor<4xi1> {
  // NSW-NOT: stablehlo.subtract
  // NSW:     stablehlo.compare GT, %[[X]]
  // WRAP:    %[[SUB:.+]] = stablehlo.subtract
  // WRAP:    stablehlo.compare LT, %[[SUB]]
  %c5 = stablehlo.constant dense<5> : tensor<4xi32>
  %c3 = stablehlo.constant dense<3> : tensor<4xi32>
  %sub = stablehlo.subtract %c5, %x : tensor<4xi32>
  %cmp = stablehlo.compare LT, %sub, %c3, SIGNED : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi1>
  return %cmp : tensor<4xi1>
}

// CHECK-LABEL: func.func @compare_sub_const_bounded(
// CHECK-SAME:    %[[X:.+]]: tensor<4xi32> {enzymexla.bounds = [-100, 100]})
func.func @compare_sub_const_bounded(%x: tensor<4xi32> {enzymexla.bounds = [-100, 100]}) -> tensor<4xi1> {
  // CHECK-NOT: stablehlo.subtract
  // CHECK:     stablehlo.compare GT, %[[X]]
  %c5 = stablehlo.constant dense<5> : tensor<4xi32>
  %c3 = stablehlo.constant dense<3> : tensor<4xi32>
  %sub = stablehlo.subtract %c5, %x : tensor<4xi32>
  %cmp = stablehlo.compare LT, %sub, %c3, SIGNED : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi1>
  return %cmp : tensor<4xi1>
}

// Synthesizing (-100 - 50) overflows i8 even when (-100 - x) does not (e.g. x = 0).
// CHECK-LABEL: func.func @no_fold_compare_sub_const_overflow(
// CHECK-SAME:    %[[X:.+]]: tensor<4xi8>)
func.func @no_fold_compare_sub_const_overflow(%x: tensor<4xi8>) -> tensor<4xi1> {
  // CHECK: %[[SUB:.+]] = stablehlo.subtract %{{.+}}, %[[X]]
  // CHECK: stablehlo.compare LT, %[[SUB]]
  %c_neg100 = stablehlo.constant dense<-100> : tensor<4xi8>
  %c50 = stablehlo.constant dense<50> : tensor<4xi8>
  %sub = stablehlo.subtract %c_neg100, %x : tensor<4xi8>
  %cmp = stablehlo.compare LT, %sub, %c50, SIGNED : (tensor<4xi8>, tensor<4xi8>) -> tensor<4xi1>
  return %cmp : tensor<4xi1>
}

// [Pattern: ReduceMaxMinMulPositiveScalar]

// CHECK-LABEL: func.func @reduce_max_mul_scalar_unbounded(
// CHECK-SAME:    %[[X:.+]]: tensor<32xi32>)
func.func @reduce_max_mul_scalar_unbounded(%x: tensor<32xi32>) -> tensor<i32> {
  // NSW:  %[[RED:.+]] = stablehlo.reduce(%[[X]] init: %{{.+}}) applies stablehlo.maximum
  // NSW:  stablehlo.multiply %[[RED]]
  // WRAP: %[[MUL:.+]] = stablehlo.multiply
  // WRAP: stablehlo.reduce(%[[MUL]] init:
  %c0 = stablehlo.constant dense<0> : tensor<i32>
  %c4 = stablehlo.constant dense<4> : tensor<i32>
  %bcast = stablehlo.broadcast_in_dim %c4, dims = [] : (tensor<i32>) -> tensor<32xi32>
  %mul = stablehlo.multiply %bcast, %x : tensor<32xi32>
  %red = stablehlo.reduce(%mul init: %c0) applies stablehlo.maximum across dimensions = [0] : (tensor<32xi32>, tensor<i32>) -> tensor<i32>
  return %red : tensor<i32>
}

// CHECK-LABEL: func.func @reduce_max_mul_scalar_bounded(
// CHECK-SAME:    %[[X:.+]]: tensor<32xi32> {enzymexla.bounds = [-100, 100]})
func.func @reduce_max_mul_scalar_bounded(%x: tensor<32xi32> {enzymexla.bounds = [-100, 100]}) -> tensor<i32> {
  // CHECK: %[[RED:.+]] = stablehlo.reduce(%[[X]] init: %{{.+}}) applies stablehlo.maximum
  // CHECK: stablehlo.multiply %[[RED]]
  %c0 = stablehlo.constant dense<0> : tensor<i32>
  %c4 = stablehlo.constant dense<4> : tensor<i32>
  %bcast = stablehlo.broadcast_in_dim %c4, dims = [] : (tensor<i32>) -> tensor<32xi32>
  %mul = stablehlo.multiply %bcast, %x : tensor<32xi32>
  %red = stablehlo.reduce(%mul init: %c0) applies stablehlo.maximum across dimensions = [0] : (tensor<32xi32>, tensor<i32>) -> tensor<i32>
  return %red : tensor<i32>
}

// [Pattern: NegDivConstSimplify]

// CHECK-LABEL: func.func @neg_div_safe_const(
// CHECK-SAME:    %[[X:.+]]: tensor<4xi8>)
func.func @neg_div_safe_const(%x: tensor<4xi8>) -> tensor<4xi8> {
  // CHECK-NOT: stablehlo.negate
  // CHECK:     %[[CST:.+]] = stablehlo.constant dense<-10>
  // CHECK:     stablehlo.divide %[[CST]], %[[X]]
  %c10 = stablehlo.constant dense<10> : tensor<4xi8>
  %div = stablehlo.divide %c10, %x : tensor<4xi8>
  %neg = stablehlo.negate %div : tensor<4xi8>
  return %neg : tensor<4xi8>
}

// Rewriting -(x / [1, 2, 3, 4]) to x / [-1, -2, -3, -4] introduces INT_MIN / -1
// at lane 0 unless signed wrap is excluded.
// CHECK-LABEL: func.func @neg_div_rhs_one_unbounded(
// CHECK-SAME:    %[[X:.+]]: tensor<4xi8>)
func.func @neg_div_rhs_one_unbounded(%x: tensor<4xi8>) -> tensor<4xi8> {
  // NSW-NOT: stablehlo.negate
  // NSW:     %[[CST:.+]] = stablehlo.constant dense<[-1, -2, -3, -4]>
  // NSW:     stablehlo.divide %[[X]], %[[CST]]
  // WRAP:    %[[DIV:.+]] = stablehlo.divide %[[X]]
  // WRAP:    stablehlo.negate %[[DIV]]
  %cst = stablehlo.constant dense<[1, 2, 3, 4]> : tensor<4xi8>
  %div = stablehlo.divide %x, %cst : tensor<4xi8>
  %neg = stablehlo.negate %div : tensor<4xi8>
  return %neg : tensor<4xi8>
}

// Negating constant INT_MIN (-128 in i8) overflows even when -((-128) / 2) == 64
// has no overflow in the input, so this must not fold even under NSW.
// CHECK-LABEL: func.func @no_fold_neg_div_int_min_const(
// CHECK-SAME:    %[[X:.+]]: tensor<4xi8>)
func.func @no_fold_neg_div_int_min_const(%x: tensor<4xi8>) -> tensor<4xi8> {
  // CHECK: %[[DIV:.+]] = stablehlo.divide %{{.+}}, %[[X]]
  // CHECK: stablehlo.negate %[[DIV]]
  %min = stablehlo.constant dense<-128> : tensor<4xi8>
  %div = stablehlo.divide %min, %x : tensor<4xi8>
  %neg = stablehlo.negate %div : tensor<4xi8>
  return %neg : tensor<4xi8>
}

// [Pattern: BinaryNegatedOperandsSimplify<DivOp>]

// CHECK-LABEL: func.func @div_negated_operands_unbounded(
// CHECK-SAME:    %[[X:.+]]: tensor<4xi32>, %[[Y:.+]]: tensor<4xi32>)
func.func @div_negated_operands_unbounded(%x: tensor<4xi32>, %y: tensor<4xi32>) -> tensor<4xi32> {
  // NSW-NOT: stablehlo.negate
  // NSW:     stablehlo.divide %[[X]], %[[Y]]
  // WRAP:    %[[NX:.+]] = stablehlo.negate %[[X]]
  // WRAP:    %[[NY:.+]] = stablehlo.negate %[[Y]]
  // WRAP:    stablehlo.divide %[[NX]], %[[NY]]
  %nx = stablehlo.negate %x : tensor<4xi32>
  %ny = stablehlo.negate %y : tensor<4xi32>
  %div = stablehlo.divide %nx, %ny : tensor<4xi32>
  return %div : tensor<4xi32>
}

// CHECK-LABEL: func.func @div_negated_operands_bounded(
// CHECK-SAME:    %[[X:.+]]: tensor<4xi32> {enzymexla.bounds = [-100, 100]},
// CHECK-SAME:    %[[Y:.+]]: tensor<4xi32> {enzymexla.bounds = [-100, 100]})
func.func @div_negated_operands_bounded(%x: tensor<4xi32> {enzymexla.bounds = [-100, 100]}, %y: tensor<4xi32> {enzymexla.bounds = [-100, 100]}) -> tensor<4xi32> {
  // CHECK-NOT: stablehlo.negate
  // CHECK:     stablehlo.divide %[[X]], %[[Y]]
  %nx = stablehlo.negate %x : tensor<4xi32>
  %ny = stablehlo.negate %y : tensor<4xi32>
  %div = stablehlo.divide %nx, %ny : tensor<4xi32>
  return %div : tensor<4xi32>
}

// Reversing (x - y) into (y - x) synthesizes a new overflow when (x - y) == -128
// (e.g. -100 - 28 == -128, divided by -2 == 64), so it requires x - y != INT_MIN.
// CHECK-LABEL: func.func @no_fold_div_sub_unbounded(
// CHECK-SAME:    %[[X:.+]]: tensor<4xi8>, %[[Y:.+]]: tensor<4xi8>, %[[Z:.+]]: tensor<4xi8>)
func.func @no_fold_div_sub_unbounded(%x: tensor<4xi8>, %y: tensor<4xi8>, %z: tensor<4xi8>) -> tensor<4xi8> {
  // CHECK: %[[SUB:.+]] = stablehlo.subtract %[[X]], %[[Y]]
  // CHECK: %[[NEG:.+]] = stablehlo.negate %[[Z]]
  // CHECK: stablehlo.divide %[[SUB]], %[[NEG]]
  %sub = stablehlo.subtract %x, %y : tensor<4xi8>
  %neg = stablehlo.negate %z : tensor<4xi8>
  %div = stablehlo.divide %sub, %neg : tensor<4xi8>
  return %div : tensor<4xi8>
}

// CHECK-LABEL: func.func @div_sub_bounded(
// CHECK-SAME:    %[[X:.+]]: tensor<4xi8> {enzymexla.bounds = [-50, 50]},
// CHECK-SAME:    %[[Y:.+]]: tensor<4xi8> {enzymexla.bounds = [-50, 50]},
// CHECK-SAME:    %[[Z:.+]]: tensor<4xi8> {enzymexla.bounds = [-50, 50]})
func.func @div_sub_bounded(%x: tensor<4xi8> {enzymexla.bounds = [-50, 50]}, %y: tensor<4xi8> {enzymexla.bounds = [-50, 50]}, %z: tensor<4xi8> {enzymexla.bounds = [-50, 50]}) -> tensor<4xi8> {
  // CHECK-NOT: stablehlo.negate
  // CHECK:     %[[SUB:.+]] = stablehlo.subtract %[[Y]], %[[X]]
  // CHECK:     stablehlo.divide %[[SUB]], %[[Z]]
  %sub = stablehlo.subtract %x, %y : tensor<4xi8>
  %neg = stablehlo.negate %z : tensor<4xi8>
  %div = stablehlo.divide %sub, %neg : tensor<4xi8>
  return %div : tensor<4xi8>
}
