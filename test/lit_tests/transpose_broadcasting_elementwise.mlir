// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt | FileCheck %s
// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt --inline --canonicalize --symbol-dce --drop-unsupported-attributes | stablehlo-translate --interpret
// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=broadcasting_elementwise_all_transpose_operands_simplify" --transform-interpreter --enzyme-hlo-remove-transform | FileCheck %s --check-prefix=TRANSFORM

// A select with a tensor predicate blocks ordinary elementwise transpose
// factoring: SelectOp has BroadcastingElementwise, not Elementwise. Factor
// its input transposes first so the shared arithmetic can stay in 2x3 layout.
// The selected value and its square are each computed once and reused.
// The transform-only run factors the select, leaving the multiply in 3x2
// layout. This checks the new pattern's transform registration directly.
// TRANSFORM-LABEL: func.func @shared_select(
// TRANSFORM-SAME: %[[P:.*]]: tensor<2x3xi1>, %[[X:.*]]: tensor<2x3xi32>
// TRANSFORM: %[[S:.*]] = stablehlo.select %[[P]], %[[X]], {{.*}} : tensor<2x3xi1>, tensor<2x3xi32>
// TRANSFORM-NEXT: %[[T:.*]] = stablehlo.transpose %[[S]], dims = [1, 0]
// TRANSFORM-NEXT: %[[M:.*]] = stablehlo.multiply %[[T]], %[[T]] : tensor<3x2xi32>
// CHECK-LABEL: func.func @shared_select(
// CHECK-SAME: %[[P:.*]]: tensor<2x3xi1>, %[[X:.*]]: tensor<2x3xi32>
// CHECK-NOT: stablehlo.transpose
// CHECK: %[[S:.*]] = stablehlo.select %[[P]], %[[X]],
// CHECK-NOT: stablehlo.transpose
// CHECK: %[[M:.*]] = stablehlo.multiply %[[S]], %[[S]]
// CHECK: %[[A:.*]] = stablehlo.add
// CHECK-NOT: stablehlo.transpose
// CHECK: return %[[M]], %[[A]]
func.func @shared_select(%pred: tensor<2x3xi1>, %x: tensor<2x3xi32>) -> (tensor<2x3xi32>, tensor<2x3xi32>) {
  %zero = stablehlo.constant dense<0> : tensor<3x2xi32>
  %p = stablehlo.transpose %pred, dims = [1, 0] : (tensor<2x3xi1>) -> tensor<3x2xi1>
  %t = stablehlo.transpose %x, dims = [1, 0] : (tensor<2x3xi32>) -> tensor<3x2xi32>
  %s = stablehlo.select %p, %t, %zero : tensor<3x2xi1>, tensor<3x2xi32>
  %m = stablehlo.multiply %s, %s : tensor<3x2xi32>
  %a = stablehlo.add %s, %m : tensor<3x2xi32>
  %r0 = stablehlo.transpose %m, dims = [1, 0] : (tensor<3x2xi32>) -> tensor<2x3xi32>
  %r1 = stablehlo.transpose %a, dims = [1, 0] : (tensor<3x2xi32>) -> tensor<2x3xi32>
  return %r0, %r1 : tensor<2x3xi32>, tensor<2x3xi32>
}

// Scalar predicates stay scalar. The result shape must come from the select's
// tensor result, not its first operand. The non-splat constant's elements must
// also follow the inverse permutation.
// A sharding attribute must not block factoring and must be preserved.
// TRANSFORM-LABEL: func.func @scalar_select(
// TRANSFORM-SAME: %[[P:.*]]: tensor<i1>, %[[X:.*]]: tensor<2x3xi32>
// TRANSFORM: stablehlo.select %[[P]], %[[X]], {{.*}} {mhlo.sharding = "{replicated}"} : tensor<i1>, tensor<2x3xi32>
// CHECK-LABEL: func.func @scalar_select(
// CHECK-SAME: %[[P:.*]]: tensor<i1>, %[[X:.*]]: tensor<2x3xi32>
// CHECK: stablehlo.constant dense<{{.*}}1, 3, 5{{.*}}2, 4, 6{{.*}}> : tensor<2x3xi32>
// CHECK-NOT: stablehlo.transpose
// CHECK: stablehlo.select %[[P]], %[[X]],
// CHECK-NOT: stablehlo.transpose
// CHECK: return
func.func @scalar_select(%pred: tensor<i1>, %x: tensor<2x3xi32>) -> tensor<2x3xi32> {
  %fallback = stablehlo.constant dense<[[1, 2], [3, 4], [5, 6]]> : tensor<3x2xi32>
  %t = stablehlo.transpose %x, dims = [1, 0] : (tensor<2x3xi32>) -> tensor<3x2xi32>
  %s = stablehlo.select %pred, %t, %fallback {mhlo.sharding = "{replicated}"} : tensor<i1>, tensor<3x2xi32>
  %r = stablehlo.transpose %s, dims = [1, 0] : (tensor<3x2xi32>) -> tensor<2x3xi32>
  return %r : tensor<2x3xi32>
}

// Clamp also has BroadcastingElementwise. Both scalar bounds must remain
// scalars while the tensor operand and result change layout.
// TRANSFORM-LABEL: func.func @scalar_clamp(
// TRANSFORM-SAME: %[[LOW:.*]]: tensor<i32>, %[[X:.*]]: tensor<2x3xi32>, %[[HIGH:.*]]: tensor<i32>
// TRANSFORM: stablehlo.clamp %[[LOW]], %[[X]], %[[HIGH]] : (tensor<i32>, tensor<2x3xi32>, tensor<i32>) -> tensor<2x3xi32>
// CHECK-LABEL: func.func @scalar_clamp(
// CHECK-SAME: %[[LOW:.*]]: tensor<i32>, %[[X:.*]]: tensor<2x3xi32>, %[[HIGH:.*]]: tensor<i32>
// CHECK-NOT: stablehlo.transpose
// CHECK: stablehlo.clamp %[[LOW]], %[[X]], %[[HIGH]] : (tensor<i32>, tensor<2x3xi32>, tensor<i32>) -> tensor<2x3xi32>
// CHECK-NOT: stablehlo.transpose
// CHECK: return
func.func @scalar_clamp(%low: tensor<i32>, %x: tensor<2x3xi32>, %high: tensor<i32>) -> tensor<2x3xi32> {
  %t = stablehlo.transpose %x, dims = [1, 0] : (tensor<2x3xi32>) -> tensor<3x2xi32>
  %c = stablehlo.clamp %low, %t, %high : (tensor<i32>, tensor<3x2xi32>, tensor<i32>) -> tensor<3x2xi32>
  %r = stablehlo.transpose %c, dims = [1, 0] : (tensor<3x2xi32>) -> tensor<2x3xi32>
  return %r : tensor<2x3xi32>
}

// Exercise a non-self-inverse permutation; tensor clamp bounds are permuted
// along with the value. The scalar upper bound still has no axes.
// CHECK-LABEL: func.func @three_axes(
// CHECK-NOT: stablehlo.transpose
// CHECK: stablehlo.clamp
// CHECK-NOT: stablehlo.transpose
// CHECK: return
func.func @three_axes(%low: tensor<2x3x4xi32>, %x: tensor<2x3x4xi32>, %high: tensor<i32>) -> tensor<2x3x4xi32> {
  %l = stablehlo.transpose %low, dims = [2, 0, 1] : (tensor<2x3x4xi32>) -> tensor<4x2x3xi32>
  %t = stablehlo.transpose %x, dims = [2, 0, 1] : (tensor<2x3x4xi32>) -> tensor<4x2x3xi32>
  %c = stablehlo.clamp %l, %t, %high : (tensor<4x2x3xi32>, tensor<4x2x3xi32>, tensor<i32>) -> tensor<4x2x3xi32>
  %r = stablehlo.transpose %c, dims = [1, 2, 0] : (tensor<4x2x3xi32>) -> tensor<2x3x4xi32>
  return %r : tensor<2x3x4xi32>
}

// A common permutation is required. These square tensors have equal shapes
// but different permutations, so their indexing cannot be factored together.
// CHECK-LABEL: func.func @different_permutations(
// CHECK: stablehlo.transpose {{.*}}dims = [1, 2, 0]
// CHECK: stablehlo.transpose {{.*}}dims = [2, 0, 1]
// CHECK: stablehlo.select
// CHECK: return
func.func @different_permutations(%pred: tensor<2x2x2xi1>, %x: tensor<2x2x2xi32>, %y: tensor<2x2x2xi32>) -> tensor<2x2x2xi32> {
  %p = stablehlo.transpose %pred, dims = [1, 2, 0] : (tensor<2x2x2xi1>) -> tensor<2x2x2xi1>
  %t = stablehlo.transpose %x, dims = [2, 0, 1] : (tensor<2x2x2xi32>) -> tensor<2x2x2xi32>
  %u = stablehlo.transpose %y, dims = [2, 0, 1] : (tensor<2x2x2xi32>) -> tensor<2x2x2xi32>
  %r = stablehlo.select %p, %t, %u : tensor<2x2x2xi1>, tensor<2x2x2xi32>
  return %r : tensor<2x2x2xi32>
}

func.func @main() {
  %x = stablehlo.constant dense<[[0, 1, 2], [3, 4, 5]]> : tensor<2x3xi32>
  %p = stablehlo.constant dense<[[true, false, true], [false, true, true]]> : tensor<2x3xi1>
  %expected_square = stablehlo.constant dense<[[0, 0, 4], [0, 16, 25]]> : tensor<2x3xi32>
  %expected_sum = stablehlo.constant dense<[[0, 0, 6], [0, 20, 30]]> : tensor<2x3xi32>
  %shared:2 = func.call @shared_select(%p, %x) : (tensor<2x3xi1>, tensor<2x3xi32>) -> (tensor<2x3xi32>, tensor<2x3xi32>)
  check.expect_eq %shared#0, %expected_square : tensor<2x3xi32>
  check.expect_eq %shared#1, %expected_sum : tensor<2x3xi32>
  %yes = stablehlo.constant dense<true> : tensor<i1>
  %no = stablehlo.constant dense<false> : tensor<i1>
  %fallback = stablehlo.constant dense<[[1, 3, 5], [2, 4, 6]]> : tensor<2x3xi32>
  %chosen = func.call @scalar_select(%yes, %x) : (tensor<i1>, tensor<2x3xi32>) -> tensor<2x3xi32>
  %other = func.call @scalar_select(%no, %x) : (tensor<i1>, tensor<2x3xi32>) -> tensor<2x3xi32>
  check.expect_eq %chosen, %x : tensor<2x3xi32>
  check.expect_eq %other, %fallback : tensor<2x3xi32>
  %low = stablehlo.constant dense<1> : tensor<i32>
  %high = stablehlo.constant dense<4> : tensor<i32>
  %expected_clamp = stablehlo.constant dense<[[1, 1, 2], [3, 4, 4]]> : tensor<2x3xi32>
  %clamped = func.call @scalar_clamp(%low, %x, %high) : (tensor<i32>, tensor<2x3xi32>, tensor<i32>) -> tensor<2x3xi32>
  check.expect_eq %clamped, %expected_clamp : tensor<2x3xi32>
  return
}
