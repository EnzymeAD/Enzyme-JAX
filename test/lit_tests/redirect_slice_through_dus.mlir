// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=redirect_slice_through_dus" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s

// Positive: slice region [0,3) does not overlap DUS update region [5,8) — redirect slice(A) to slice(B)
func.func @basic_redirect(%base: tensor<10xf32>, %update_a: tensor<4xf32>, %update_b: tensor<3xf32>) -> (tensor<3xf32>, tensor<10xf32>) {
    %c2 = stablehlo.constant dense<2> : tensor<i32>
    %c5 = stablehlo.constant dense<5> : tensor<i32>
    %a = stablehlo.dynamic_update_slice %base, %update_a, %c2 : (tensor<10xf32>, tensor<4xf32>, tensor<i32>) -> tensor<10xf32>
    %b = stablehlo.dynamic_update_slice %a, %update_b, %c5 : (tensor<10xf32>, tensor<3xf32>, tensor<i32>) -> tensor<10xf32>
    %s = stablehlo.slice %a [0:3] : (tensor<10xf32>) -> tensor<3xf32>
    return %s, %b : tensor<3xf32>, tensor<10xf32>
}

// CHECK-LABEL:   func.func @basic_redirect(
// CHECK:           %[[A:.*]] = stablehlo.dynamic_update_slice
// CHECK:           %[[B:.*]] = stablehlo.dynamic_update_slice %[[A]]
// CHECK:           %[[S:.*]] = stablehlo.slice %[[B]] [0:3]
// CHECK:           return %[[S]], %[[B]]


// Positive 2D: slice region [0:4, 0:5] does not overlap DUS update region [0:4, 6:10] — redirect
func.func @redirect_2d(%base: tensor<8x10xf32>, %update_a: tensor<3x3xf32>, %update_b: tensor<4x4xf32>) -> (tensor<4x5xf32>, tensor<8x10xf32>) {
    %c0 = stablehlo.constant dense<0> : tensor<i32>
    %c6 = stablehlo.constant dense<6> : tensor<i32>
    %a = stablehlo.dynamic_update_slice %base, %update_a, %c0, %c0 : (tensor<8x10xf32>, tensor<3x3xf32>, tensor<i32>, tensor<i32>) -> tensor<8x10xf32>
    %b = stablehlo.dynamic_update_slice %a, %update_b, %c0, %c6 : (tensor<8x10xf32>, tensor<4x4xf32>, tensor<i32>, tensor<i32>) -> tensor<8x10xf32>
    %s = stablehlo.slice %a [0:4, 0:5] : (tensor<8x10xf32>) -> tensor<4x5xf32>
    return %s, %b : tensor<4x5xf32>, tensor<8x10xf32>
}

// CHECK-LABEL:   func.func @redirect_2d(
// CHECK:           %[[A:.*]] = stablehlo.dynamic_update_slice
// CHECK:           %[[B:.*]] = stablehlo.dynamic_update_slice %[[A]]
// CHECK:           %[[S:.*]] = stablehlo.slice %[[B]] [0:4, 0:5]
// CHECK:           return %[[S]], %[[B]]


// Positive: dynamic_slice with constant indices, no overlap — redirect
func.func @redirect_dynamic_slice(%base: tensor<10xf32>, %update_a: tensor<2xf32>, %update_b: tensor<3xf32>) -> (tensor<2xf32>, tensor<10xf32>) {
    %c0 = stablehlo.constant dense<0> : tensor<i32>
    %c5 = stablehlo.constant dense<5> : tensor<i32>
    %c8 = stablehlo.constant dense<8> : tensor<i32>
    %a = stablehlo.dynamic_update_slice %base, %update_a, %c0 : (tensor<10xf32>, tensor<2xf32>, tensor<i32>) -> tensor<10xf32>
    %b = stablehlo.dynamic_update_slice %a, %update_b, %c5 : (tensor<10xf32>, tensor<3xf32>, tensor<i32>) -> tensor<10xf32>
    %s = "stablehlo.dynamic_slice"(%a, %c8) {slice_sizes = array<i64: 2>} : (tensor<10xf32>, tensor<i32>) -> tensor<2xf32>
    return %s, %b : tensor<2xf32>, tensor<10xf32>
}

// CHECK-LABEL:   func.func @redirect_dynamic_slice(
// CHECK:           %[[A:.*]] = stablehlo.dynamic_update_slice
// CHECK:           %[[B:.*]] = stablehlo.dynamic_update_slice %[[A]]
// CHECK:           %[[S:.*]] = stablehlo.dynamic_slice %[[B]]
// CHECK:           return %[[S]], %[[B]]


// Negative: slice region [4:8) overlaps with DUS update region [5:8) — do NOT redirect
func.func @overlap_no_redirect(%base: tensor<10xf32>, %update_a: tensor<4xf32>, %update_b: tensor<3xf32>) -> (tensor<4xf32>, tensor<10xf32>) {
    %c2 = stablehlo.constant dense<2> : tensor<i32>
    %c5 = stablehlo.constant dense<5> : tensor<i32>
    %a = stablehlo.dynamic_update_slice %base, %update_a, %c2 : (tensor<10xf32>, tensor<4xf32>, tensor<i32>) -> tensor<10xf32>
    %b = stablehlo.dynamic_update_slice %a, %update_b, %c5 : (tensor<10xf32>, tensor<3xf32>, tensor<i32>) -> tensor<10xf32>
    %s = stablehlo.slice %a [4:8] : (tensor<10xf32>) -> tensor<4xf32>
    return %s, %b : tensor<4xf32>, tensor<10xf32>
}

// CHECK-LABEL:   func.func @overlap_no_redirect(
// CHECK:           %[[A:.*]] = stablehlo.dynamic_update_slice
// CHECK:           %[[B:.*]] = stablehlo.dynamic_update_slice %[[A]]
// CHECK:           %[[S:.*]] = stablehlo.slice %[[A]] [4:8]
// CHECK:           return %[[S]], %[[B]]


// Negative: slice occurs BEFORE B — do NOT redirect (dominance)
func.func @slice_before_dus(%base: tensor<10xf32>, %update_a: tensor<4xf32>, %update_b: tensor<3xf32>) -> (tensor<3xf32>, tensor<10xf32>) {
    %c2 = stablehlo.constant dense<2> : tensor<i32>
    %c5 = stablehlo.constant dense<5> : tensor<i32>
    %a = stablehlo.dynamic_update_slice %base, %update_a, %c2 : (tensor<10xf32>, tensor<4xf32>, tensor<i32>) -> tensor<10xf32>
    %s = stablehlo.slice %a [0:3] : (tensor<10xf32>) -> tensor<3xf32>
    %b = stablehlo.dynamic_update_slice %a, %update_b, %c5 : (tensor<10xf32>, tensor<3xf32>, tensor<i32>) -> tensor<10xf32>
    return %s, %b : tensor<3xf32>, tensor<10xf32>
}

// CHECK-LABEL:   func.func @slice_before_dus(
// CHECK:           %[[A:.*]] = stablehlo.dynamic_update_slice
// CHECK:           %[[S:.*]] = stablehlo.slice %[[A]] [0:3]
// CHECK:           %[[B:.*]] = stablehlo.dynamic_update_slice %[[A]]
// CHECK:           return %[[S]], %[[B]]


// Negative: DUS has dynamic (non-constant) start indices — cannot analyze overlap
func.func @dynamic_dus_indices(%base: tensor<10xf32>, %update_a: tensor<4xf32>, %update_b: tensor<3xf32>, %idx: tensor<i32>) -> (tensor<3xf32>, tensor<10xf32>) {
    %c2 = stablehlo.constant dense<2> : tensor<i32>
    %a = stablehlo.dynamic_update_slice %base, %update_a, %c2 : (tensor<10xf32>, tensor<4xf32>, tensor<i32>) -> tensor<10xf32>
    %b = stablehlo.dynamic_update_slice %a, %update_b, %idx : (tensor<10xf32>, tensor<3xf32>, tensor<i32>) -> tensor<10xf32>
    %s = stablehlo.slice %a [0:3] : (tensor<10xf32>) -> tensor<3xf32>
    return %s, %b : tensor<3xf32>, tensor<10xf32>
}

// CHECK-LABEL:   func.func @dynamic_dus_indices(
// CHECK:           %[[A:.*]] = stablehlo.dynamic_update_slice
// CHECK:           %[[B:.*]] = stablehlo.dynamic_update_slice %[[A]]
// CHECK:           %[[S:.*]] = stablehlo.slice %[[A]] [0:3]
// CHECK:           return %[[S]], %[[B]]


// Positive: multiple slices of A, both non-overlapping with B — redirect both
func.func @multiple_slices(%base: tensor<10xf32>, %update_a: tensor<2xf32>, %update_b: tensor<3xf32>) -> (tensor<2xf32>, tensor<2xf32>, tensor<10xf32>) {
    %c0 = stablehlo.constant dense<0> : tensor<i32>
    %c4 = stablehlo.constant dense<4> : tensor<i32>
    %a = stablehlo.dynamic_update_slice %base, %update_a, %c0 : (tensor<10xf32>, tensor<2xf32>, tensor<i32>) -> tensor<10xf32>
    %b = stablehlo.dynamic_update_slice %a, %update_b, %c4 : (tensor<10xf32>, tensor<3xf32>, tensor<i32>) -> tensor<10xf32>
    %s1 = stablehlo.slice %a [2:4] : (tensor<10xf32>) -> tensor<2xf32>
    %s2 = stablehlo.slice %a [8:10] : (tensor<10xf32>) -> tensor<2xf32>
    return %s1, %s2, %b : tensor<2xf32>, tensor<2xf32>, tensor<10xf32>
}

// CHECK-LABEL:   func.func @multiple_slices(
// CHECK:           %[[A:.*]] = stablehlo.dynamic_update_slice
// CHECK:           %[[B:.*]] = stablehlo.dynamic_update_slice %[[A]]
// CHECK:           %[[S1:.*]] = stablehlo.slice %[[B]] [2:4]
// CHECK:           %[[S2:.*]] = stablehlo.slice %[[B]] [8:10]
// CHECK:           return %[[S1]], %[[S2]], %[[B]]


// Positive: one slice overlaps, one doesn't — only redirect the non-overlapping one
func.func @mixed_overlap(%base: tensor<10xf32>, %update_a: tensor<2xf32>, %update_b: tensor<3xf32>) -> (tensor<3xf32>, tensor<2xf32>, tensor<10xf32>) {
    %c0 = stablehlo.constant dense<0> : tensor<i32>
    %c4 = stablehlo.constant dense<4> : tensor<i32>
    %a = stablehlo.dynamic_update_slice %base, %update_a, %c0 : (tensor<10xf32>, tensor<2xf32>, tensor<i32>) -> tensor<10xf32>
    %b = stablehlo.dynamic_update_slice %a, %update_b, %c4 : (tensor<10xf32>, tensor<3xf32>, tensor<i32>) -> tensor<10xf32>
    %s1 = stablehlo.slice %a [3:6] : (tensor<10xf32>) -> tensor<3xf32>
    %s2 = stablehlo.slice %a [8:10] : (tensor<10xf32>) -> tensor<2xf32>
    return %s1, %s2, %b : tensor<3xf32>, tensor<2xf32>, tensor<10xf32>
}

// CHECK-LABEL:   func.func @mixed_overlap(
// CHECK:           %[[A:.*]] = stablehlo.dynamic_update_slice
// CHECK:           %[[B:.*]] = stablehlo.dynamic_update_slice %[[A]]
// CHECK:           %[[S1:.*]] = stablehlo.slice %[[A]] [3:6]
// CHECK:           %[[S2:.*]] = stablehlo.slice %[[B]] [8:10]
// CHECK:           return %[[S1]], %[[S2]], %[[B]]


// Positive: slice ends exactly where DUS update starts (adjacent, no overlap) — redirect
// B updates [5,8), slice reads [3,5) — sliceLimit==dusStart, no overlap
func.func @adjacent_no_overlap(%base: tensor<10xf32>, %update_a: tensor<2xf32>, %update_b: tensor<3xf32>) -> (tensor<2xf32>, tensor<10xf32>) {
    %c0 = stablehlo.constant dense<0> : tensor<i32>
    %c5 = stablehlo.constant dense<5> : tensor<i32>
    %a = stablehlo.dynamic_update_slice %base, %update_a, %c0 : (tensor<10xf32>, tensor<2xf32>, tensor<i32>) -> tensor<10xf32>
    %b = stablehlo.dynamic_update_slice %a, %update_b, %c5 : (tensor<10xf32>, tensor<3xf32>, tensor<i32>) -> tensor<10xf32>
    %s = stablehlo.slice %a [3:5] : (tensor<10xf32>) -> tensor<2xf32>
    return %s, %b : tensor<2xf32>, tensor<10xf32>
}

// CHECK-LABEL:   func.func @adjacent_no_overlap(
// CHECK:           %[[A:.*]] = stablehlo.dynamic_update_slice
// CHECK:           %[[B:.*]] = stablehlo.dynamic_update_slice %[[A]]
// CHECK:           %[[S:.*]] = stablehlo.slice %[[B]] [3:5]
// CHECK:           return %[[S]], %[[B]]


// Negative: slice overlaps DUS update by exactly 1 element — do NOT redirect
// B updates [5,8), slice reads [4,6) — overlap at element 5
func.func @overlap_by_one(%base: tensor<10xf32>, %update_a: tensor<2xf32>, %update_b: tensor<3xf32>) -> (tensor<2xf32>, tensor<10xf32>) {
    %c0 = stablehlo.constant dense<0> : tensor<i32>
    %c5 = stablehlo.constant dense<5> : tensor<i32>
    %a = stablehlo.dynamic_update_slice %base, %update_a, %c0 : (tensor<10xf32>, tensor<2xf32>, tensor<i32>) -> tensor<10xf32>
    %b = stablehlo.dynamic_update_slice %a, %update_b, %c5 : (tensor<10xf32>, tensor<3xf32>, tensor<i32>) -> tensor<10xf32>
    %s = stablehlo.slice %a [4:6] : (tensor<10xf32>) -> tensor<2xf32>
    return %s, %b : tensor<2xf32>, tensor<10xf32>
}

// CHECK-LABEL:   func.func @overlap_by_one(
// CHECK:           %[[A:.*]] = stablehlo.dynamic_update_slice
// CHECK:           %[[B:.*]] = stablehlo.dynamic_update_slice %[[A]]
// CHECK:           %[[S:.*]] = stablehlo.slice %[[A]] [4:6]
// CHECK:           return %[[S]], %[[B]]


// Positive: slice starts exactly where DUS update ends (adjacent, no overlap) — redirect
// B updates [5,8), slice reads [8,10) — sliceStart==dusStart+uSize, no overlap
func.func @adjacent_after_no_overlap(%base: tensor<10xf32>, %update_a: tensor<2xf32>, %update_b: tensor<3xf32>) -> (tensor<2xf32>, tensor<10xf32>) {
    %c0 = stablehlo.constant dense<0> : tensor<i32>
    %c5 = stablehlo.constant dense<5> : tensor<i32>
    %a = stablehlo.dynamic_update_slice %base, %update_a, %c0 : (tensor<10xf32>, tensor<2xf32>, tensor<i32>) -> tensor<10xf32>
    %b = stablehlo.dynamic_update_slice %a, %update_b, %c5 : (tensor<10xf32>, tensor<3xf32>, tensor<i32>) -> tensor<10xf32>
    %s = stablehlo.slice %a [8:10] : (tensor<10xf32>) -> tensor<2xf32>
    return %s, %b : tensor<2xf32>, tensor<10xf32>
}

// CHECK-LABEL:   func.func @adjacent_after_no_overlap(
// CHECK:           %[[A:.*]] = stablehlo.dynamic_update_slice
// CHECK:           %[[B:.*]] = stablehlo.dynamic_update_slice %[[A]]
// CHECK:           %[[S:.*]] = stablehlo.slice %[[B]] [8:10]
// CHECK:           return %[[S]], %[[B]]


// Negative: slice overlaps DUS update by 1 element from the end — do NOT redirect
// B updates [5,8), slice reads [7,9) — overlap at element 7
func.func @overlap_by_one_end(%base: tensor<10xf32>, %update_a: tensor<2xf32>, %update_b: tensor<3xf32>) -> (tensor<2xf32>, tensor<10xf32>) {
    %c0 = stablehlo.constant dense<0> : tensor<i32>
    %c5 = stablehlo.constant dense<5> : tensor<i32>
    %a = stablehlo.dynamic_update_slice %base, %update_a, %c0 : (tensor<10xf32>, tensor<2xf32>, tensor<i32>) -> tensor<10xf32>
    %b = stablehlo.dynamic_update_slice %a, %update_b, %c5 : (tensor<10xf32>, tensor<3xf32>, tensor<i32>) -> tensor<10xf32>
    %s = stablehlo.slice %a [7:9] : (tensor<10xf32>) -> tensor<2xf32>
    return %s, %b : tensor<2xf32>, tensor<10xf32>
}

// CHECK-LABEL:   func.func @overlap_by_one_end(
// CHECK:           %[[A:.*]] = stablehlo.dynamic_update_slice
// CHECK:           %[[B:.*]] = stablehlo.dynamic_update_slice %[[A]]
// CHECK:           %[[S:.*]] = stablehlo.slice %[[A]] [7:9]
// CHECK:           return %[[S]], %[[B]]
