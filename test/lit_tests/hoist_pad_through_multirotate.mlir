// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=hoist_pad_through_multirotate" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s

// Basic case: multi_rotate along dim 0, all results padded along dim 1
func.func @basic_hoist(%arg0: tensor<10x7xf32>) -> (tensor<10x8xf32>, tensor<10x8xf32>, tensor<10x8xf32>) {
    %zero = stablehlo.constant dense<0.0> : tensor<f32>
    %0:3 = "enzymexla.multi_rotate"(%arg0) {dimension = 0 : i32, left_amount = 1 : i32, right_amount = 1 : i32} : (tensor<10x7xf32>) -> (tensor<10x7xf32>, tensor<10x7xf32>, tensor<10x7xf32>)
    %p0 = stablehlo.pad %0#0, %zero, low = [0, 0], high = [0, 1], interior = [0, 0] : (tensor<10x7xf32>, tensor<f32>) -> tensor<10x8xf32>
    %p1 = stablehlo.pad %0#1, %zero, low = [0, 0], high = [0, 1], interior = [0, 0] : (tensor<10x7xf32>, tensor<f32>) -> tensor<10x8xf32>
    %p2 = stablehlo.pad %0#2, %zero, low = [0, 0], high = [0, 1], interior = [0, 0] : (tensor<10x7xf32>, tensor<f32>) -> tensor<10x8xf32>
    return %p0, %p1, %p2 : tensor<10x8xf32>, tensor<10x8xf32>, tensor<10x8xf32>
}

// CHECK-LABEL:   func.func @basic_hoist(
// CHECK-SAME:                           %[[ARG:.*]]: tensor<10x7xf32>) -> (tensor<10x8xf32>, tensor<10x8xf32>, tensor<10x8xf32>) {
// CHECK:           %[[ZERO:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK:           %[[PAD:.*]] = stablehlo.pad %[[ARG]], %[[ZERO]], low = [0, 0], high = [0, 1], interior = [0, 0] : (tensor<10x7xf32>, tensor<f32>) -> tensor<10x8xf32>
// CHECK:           %[[MR:.*]]:3 = "enzymexla.multi_rotate"(%[[PAD]]) <{dimension = 0 : i32, left_amount = 1 : i32, right_amount = 1 : i32}> : (tensor<10x8xf32>) -> (tensor<10x8xf32>, tensor<10x8xf32>, tensor<10x8xf32>)
// CHECK:           return %[[MR]]#0, %[[MR]]#1, %[[MR]]#2 : tensor<10x8xf32>, tensor<10x8xf32>, tensor<10x8xf32>
// CHECK:         }


// 3D tensor: rotate along dim 1, pad along dim 2 (39 -> 40)
func.func @hoist_3d(%arg0: tensor<10x20x39xf32>) -> (tensor<10x20x40xf32>, tensor<10x20x40xf32>, tensor<10x20x40xf32>, tensor<10x20x40xf32>, tensor<10x20x40xf32>) {
    %zero = stablehlo.constant dense<0.0> : tensor<f32>
    %0:5 = "enzymexla.multi_rotate"(%arg0) {dimension = 1 : i32, left_amount = 2 : i32, right_amount = 2 : i32} : (tensor<10x20x39xf32>) -> (tensor<10x20x39xf32>, tensor<10x20x39xf32>, tensor<10x20x39xf32>, tensor<10x20x39xf32>, tensor<10x20x39xf32>)
    %p0 = stablehlo.pad %0#0, %zero, low = [0, 0, 0], high = [0, 0, 1], interior = [0, 0, 0] : (tensor<10x20x39xf32>, tensor<f32>) -> tensor<10x20x40xf32>
    %p1 = stablehlo.pad %0#1, %zero, low = [0, 0, 0], high = [0, 0, 1], interior = [0, 0, 0] : (tensor<10x20x39xf32>, tensor<f32>) -> tensor<10x20x40xf32>
    %p2 = stablehlo.pad %0#2, %zero, low = [0, 0, 0], high = [0, 0, 1], interior = [0, 0, 0] : (tensor<10x20x39xf32>, tensor<f32>) -> tensor<10x20x40xf32>
    %p3 = stablehlo.pad %0#3, %zero, low = [0, 0, 0], high = [0, 0, 1], interior = [0, 0, 0] : (tensor<10x20x39xf32>, tensor<f32>) -> tensor<10x20x40xf32>
    %p4 = stablehlo.pad %0#4, %zero, low = [0, 0, 0], high = [0, 0, 1], interior = [0, 0, 0] : (tensor<10x20x39xf32>, tensor<f32>) -> tensor<10x20x40xf32>
    return %p0, %p1, %p2, %p3, %p4 : tensor<10x20x40xf32>, tensor<10x20x40xf32>, tensor<10x20x40xf32>, tensor<10x20x40xf32>, tensor<10x20x40xf32>
}

// CHECK-LABEL:   func.func @hoist_3d(
// CHECK-SAME:                        %[[ARG:.*]]: tensor<10x20x39xf32>) -> (tensor<10x20x40xf32>, tensor<10x20x40xf32>, tensor<10x20x40xf32>, tensor<10x20x40xf32>, tensor<10x20x40xf32>) {
// CHECK:           %[[ZERO:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK:           %[[PAD:.*]] = stablehlo.pad %[[ARG]], %[[ZERO]], low = [0, 0, 0], high = [0, 0, 1], interior = [0, 0, 0] : (tensor<10x20x39xf32>, tensor<f32>) -> tensor<10x20x40xf32>
// CHECK:           %[[MR:.*]]:5 = "enzymexla.multi_rotate"(%[[PAD]]) <{dimension = 1 : i32, left_amount = 2 : i32, right_amount = 2 : i32}> : (tensor<10x20x40xf32>) -> (tensor<10x20x40xf32>, tensor<10x20x40xf32>, tensor<10x20x40xf32>, tensor<10x20x40xf32>, tensor<10x20x40xf32>)
// CHECK:           return %[[MR]]#0, %[[MR]]#1, %[[MR]]#2, %[[MR]]#3, %[[MR]]#4 : tensor<10x20x40xf32>, tensor<10x20x40xf32>, tensor<10x20x40xf32>, tensor<10x20x40xf32>, tensor<10x20x40xf32>
// CHECK:         }


// Some results unused — should still hoist
func.func @unused_results(%arg0: tensor<10x7xf32>) -> (tensor<10x8xf32>, tensor<10x8xf32>) {
    %zero = stablehlo.constant dense<0.0> : tensor<f32>
    %0:3 = "enzymexla.multi_rotate"(%arg0) {dimension = 0 : i32, left_amount = 1 : i32, right_amount = 1 : i32} : (tensor<10x7xf32>) -> (tensor<10x7xf32>, tensor<10x7xf32>, tensor<10x7xf32>)
    %p0 = stablehlo.pad %0#0, %zero, low = [0, 0], high = [0, 1], interior = [0, 0] : (tensor<10x7xf32>, tensor<f32>) -> tensor<10x8xf32>
    %p2 = stablehlo.pad %0#2, %zero, low = [0, 0], high = [0, 1], interior = [0, 0] : (tensor<10x7xf32>, tensor<f32>) -> tensor<10x8xf32>
    return %p0, %p2 : tensor<10x8xf32>, tensor<10x8xf32>
}

// CHECK-LABEL:   func.func @unused_results(
// CHECK-SAME:                              %[[ARG:.*]]: tensor<10x7xf32>) -> (tensor<10x8xf32>, tensor<10x8xf32>) {
// CHECK:           %[[ZERO:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK:           %[[PAD:.*]] = stablehlo.pad %[[ARG]], %[[ZERO]], low = [0, 0], high = [0, 1], interior = [0, 0] : (tensor<10x7xf32>, tensor<f32>) -> tensor<10x8xf32>
// CHECK:           %[[MR:.*]]:3 = "enzymexla.multi_rotate"(%[[PAD]]) <{dimension = 0 : i32, left_amount = 1 : i32, right_amount = 1 : i32}> : (tensor<10x8xf32>) -> (tensor<10x8xf32>, tensor<10x8xf32>, tensor<10x8xf32>)
// CHECK:           return %[[MR]]#0, %[[MR]]#2 : tensor<10x8xf32>, tensor<10x8xf32>
// CHECK:         }


// Negative: pad along rotate dimension — should NOT hoist
func.func @pad_along_rotate_dim(%arg0: tensor<10x8xf32>) -> (tensor<12x8xf32>, tensor<12x8xf32>, tensor<12x8xf32>) {
    %zero = stablehlo.constant dense<0.0> : tensor<f32>
    %0:3 = "enzymexla.multi_rotate"(%arg0) {dimension = 0 : i32, left_amount = 1 : i32, right_amount = 1 : i32} : (tensor<10x8xf32>) -> (tensor<10x8xf32>, tensor<10x8xf32>, tensor<10x8xf32>)
    %p0 = stablehlo.pad %0#0, %zero, low = [1, 0], high = [1, 0], interior = [0, 0] : (tensor<10x8xf32>, tensor<f32>) -> tensor<12x8xf32>
    %p1 = stablehlo.pad %0#1, %zero, low = [1, 0], high = [1, 0], interior = [0, 0] : (tensor<10x8xf32>, tensor<f32>) -> tensor<12x8xf32>
    %p2 = stablehlo.pad %0#2, %zero, low = [1, 0], high = [1, 0], interior = [0, 0] : (tensor<10x8xf32>, tensor<f32>) -> tensor<12x8xf32>
    return %p0, %p1, %p2 : tensor<12x8xf32>, tensor<12x8xf32>, tensor<12x8xf32>
}

// CHECK-LABEL:   func.func @pad_along_rotate_dim(
// CHECK:           "enzymexla.multi_rotate"
// CHECK:           stablehlo.pad
// CHECK:           stablehlo.pad
// CHECK:           stablehlo.pad


// Negative: interior padding — should NOT hoist
func.func @interior_padding(%arg0: tensor<10x7xf32>) -> (tensor<10x14xf32>, tensor<10x14xf32>, tensor<10x14xf32>) {
    %zero = stablehlo.constant dense<0.0> : tensor<f32>
    %0:3 = "enzymexla.multi_rotate"(%arg0) {dimension = 0 : i32, left_amount = 1 : i32, right_amount = 1 : i32} : (tensor<10x7xf32>) -> (tensor<10x7xf32>, tensor<10x7xf32>, tensor<10x7xf32>)
    %p0 = stablehlo.pad %0#0, %zero, low = [0, 0], high = [0, 1], interior = [0, 1] : (tensor<10x7xf32>, tensor<f32>) -> tensor<10x14xf32>
    %p1 = stablehlo.pad %0#1, %zero, low = [0, 0], high = [0, 1], interior = [0, 1] : (tensor<10x7xf32>, tensor<f32>) -> tensor<10x14xf32>
    %p2 = stablehlo.pad %0#2, %zero, low = [0, 0], high = [0, 1], interior = [0, 1] : (tensor<10x7xf32>, tensor<f32>) -> tensor<10x14xf32>
    return %p0, %p1, %p2 : tensor<10x14xf32>, tensor<10x14xf32>, tensor<10x14xf32>
}

// CHECK-LABEL:   func.func @interior_padding(
// CHECK:           "enzymexla.multi_rotate"
// CHECK:           stablehlo.pad
// CHECK:           stablehlo.pad
// CHECK:           stablehlo.pad


// Negative: inconsistent pad amounts — should NOT hoist
func.func @inconsistent_pads(%arg0: tensor<10x7xf32>) -> (tensor<10x8xf32>, tensor<10x9xf32>, tensor<10x8xf32>) {
    %zero = stablehlo.constant dense<0.0> : tensor<f32>
    %0:3 = "enzymexla.multi_rotate"(%arg0) {dimension = 0 : i32, left_amount = 1 : i32, right_amount = 1 : i32} : (tensor<10x7xf32>) -> (tensor<10x7xf32>, tensor<10x7xf32>, tensor<10x7xf32>)
    %p0 = stablehlo.pad %0#0, %zero, low = [0, 0], high = [0, 1], interior = [0, 0] : (tensor<10x7xf32>, tensor<f32>) -> tensor<10x8xf32>
    %p1 = stablehlo.pad %0#1, %zero, low = [0, 0], high = [0, 2], interior = [0, 0] : (tensor<10x7xf32>, tensor<f32>) -> tensor<10x9xf32>
    %p2 = stablehlo.pad %0#2, %zero, low = [0, 0], high = [0, 1], interior = [0, 0] : (tensor<10x7xf32>, tensor<f32>) -> tensor<10x8xf32>
    return %p0, %p1, %p2 : tensor<10x8xf32>, tensor<10x9xf32>, tensor<10x8xf32>
}

// CHECK-LABEL:   func.func @inconsistent_pads(
// CHECK:           "enzymexla.multi_rotate"
// CHECK:           stablehlo.pad
// CHECK:           stablehlo.pad
// CHECK:           stablehlo.pad


// Negative: result has non-pad user — should NOT hoist
func.func @non_pad_user(%arg0: tensor<10x7xf32>) -> (tensor<10x8xf32>, tensor<10x7xf32>, tensor<10x8xf32>) {
    %zero = stablehlo.constant dense<0.0> : tensor<f32>
    %0:3 = "enzymexla.multi_rotate"(%arg0) {dimension = 0 : i32, left_amount = 1 : i32, right_amount = 1 : i32} : (tensor<10x7xf32>) -> (tensor<10x7xf32>, tensor<10x7xf32>, tensor<10x7xf32>)
    %p0 = stablehlo.pad %0#0, %zero, low = [0, 0], high = [0, 1], interior = [0, 0] : (tensor<10x7xf32>, tensor<f32>) -> tensor<10x8xf32>
    %p2 = stablehlo.pad %0#2, %zero, low = [0, 0], high = [0, 1], interior = [0, 0] : (tensor<10x7xf32>, tensor<f32>) -> tensor<10x8xf32>
    return %p0, %0#1, %p2 : tensor<10x8xf32>, tensor<10x7xf32>, tensor<10x8xf32>
}

// CHECK-LABEL:   func.func @non_pad_user(
// CHECK:           "enzymexla.multi_rotate"
// CHECK:           stablehlo.pad
// CHECK:           stablehlo.pad


// Negative: one result consumed by a non-pad op (negate) — should NOT hoist
func.func @non_pad_op_user(%arg0: tensor<10x7xf32>) -> (tensor<10x8xf32>, tensor<10x7xf32>, tensor<10x8xf32>) {
    %zero = stablehlo.constant dense<0.0> : tensor<f32>
    %0:3 = "enzymexla.multi_rotate"(%arg0) {dimension = 0 : i32, left_amount = 1 : i32, right_amount = 1 : i32} : (tensor<10x7xf32>) -> (tensor<10x7xf32>, tensor<10x7xf32>, tensor<10x7xf32>)
    %p0 = stablehlo.pad %0#0, %zero, low = [0, 0], high = [0, 1], interior = [0, 0] : (tensor<10x7xf32>, tensor<f32>) -> tensor<10x8xf32>
    %neg = stablehlo.negate %0#1 : tensor<10x7xf32>
    %p2 = stablehlo.pad %0#2, %zero, low = [0, 0], high = [0, 1], interior = [0, 0] : (tensor<10x7xf32>, tensor<f32>) -> tensor<10x8xf32>
    return %p0, %neg, %p2 : tensor<10x8xf32>, tensor<10x7xf32>, tensor<10x8xf32>
}

// CHECK-LABEL:   func.func @non_pad_op_user(
// CHECK:           "enzymexla.multi_rotate"
// CHECK:           stablehlo.pad
// CHECK:           stablehlo.negate
// CHECK:           stablehlo.pad


// Negative: no padding at all — should NOT hoist
func.func @no_padding(%arg0: tensor<10x8xf32>) -> (tensor<10x8xf32>, tensor<10x8xf32>, tensor<10x8xf32>) {
    %0:3 = "enzymexla.multi_rotate"(%arg0) {dimension = 0 : i32, left_amount = 1 : i32, right_amount = 1 : i32} : (tensor<10x8xf32>) -> (tensor<10x8xf32>, tensor<10x8xf32>, tensor<10x8xf32>)
    return %0#0, %0#1, %0#2 : tensor<10x8xf32>, tensor<10x8xf32>, tensor<10x8xf32>
}

// CHECK-LABEL:   func.func @no_padding(
// CHECK:           "enzymexla.multi_rotate"
// CHECK-NOT:       stablehlo.pad


// Negative: result has multiple users (one pad, one negate) — should NOT hoist
func.func @multi_user_mixed(%arg0: tensor<10x7xf32>) -> (tensor<10x8xf32>, tensor<10x7xf32>, tensor<10x8xf32>) {
    %zero = stablehlo.constant dense<0.0> : tensor<f32>
    %0:3 = "enzymexla.multi_rotate"(%arg0) {dimension = 0 : i32, left_amount = 1 : i32, right_amount = 1 : i32} : (tensor<10x7xf32>) -> (tensor<10x7xf32>, tensor<10x7xf32>, tensor<10x7xf32>)
    %p0 = stablehlo.pad %0#0, %zero, low = [0, 0], high = [0, 1], interior = [0, 0] : (tensor<10x7xf32>, tensor<f32>) -> tensor<10x8xf32>
    %neg = stablehlo.negate %0#0 : tensor<10x7xf32>
    %p2 = stablehlo.pad %0#2, %zero, low = [0, 0], high = [0, 1], interior = [0, 0] : (tensor<10x7xf32>, tensor<f32>) -> tensor<10x8xf32>
    return %p0, %neg, %p2 : tensor<10x8xf32>, tensor<10x7xf32>, tensor<10x8xf32>
}

// CHECK-LABEL:   func.func @multi_user_mixed(
// CHECK:           "enzymexla.multi_rotate"
// CHECK:           stablehlo.pad
// CHECK:           stablehlo.negate
// CHECK:           stablehlo.pad


// Positive: multi_rotate and pads inside a while body — should hoist within the body
func.func @same_region_while(%arg0: tensor<10x7xf32>, %arg1: tensor<10x8xf32>) -> tensor<10x8xf32> {
    %zero = stablehlo.constant dense<0.0> : tensor<f32>
    %pred_init = stablehlo.constant dense<true> : tensor<i1>
    %result = stablehlo.while(%iter = %arg1) : tensor<10x8xf32>
      cond {
        %cond = stablehlo.constant dense<true> : tensor<i1>
        stablehlo.return %cond : tensor<i1>
      } do {
        %mr:3 = "enzymexla.multi_rotate"(%arg0) {dimension = 0 : i32, left_amount = 1 : i32, right_amount = 1 : i32} : (tensor<10x7xf32>) -> (tensor<10x7xf32>, tensor<10x7xf32>, tensor<10x7xf32>)
        %p0 = stablehlo.pad %mr#0, %zero, low = [0, 0], high = [0, 1], interior = [0, 0] : (tensor<10x7xf32>, tensor<f32>) -> tensor<10x8xf32>
        %p1 = stablehlo.pad %mr#1, %zero, low = [0, 0], high = [0, 1], interior = [0, 0] : (tensor<10x7xf32>, tensor<f32>) -> tensor<10x8xf32>
        %p2 = stablehlo.pad %mr#2, %zero, low = [0, 0], high = [0, 1], interior = [0, 0] : (tensor<10x7xf32>, tensor<f32>) -> tensor<10x8xf32>
        %sum = stablehlo.add %p0, %p1 : tensor<10x8xf32>
        stablehlo.return %sum : tensor<10x8xf32>
      }
    return %result : tensor<10x8xf32>
}

// CHECK-LABEL:   func.func @same_region_while(
// CHECK:           stablehlo.while
// CHECK:             stablehlo.pad {{.*}} low = [0, 0], high = [0, 1]
// CHECK:             "enzymexla.multi_rotate"({{.*}}) <{dimension = 0 : i32{{.*}}}> : (tensor<10x8xf32>)
// CHECK-NOT:         stablehlo.pad {{.*}} low = [0, 0], high = [0, 1]
// CHECK:             stablehlo.add


// Positive: multi_rotate in outer region, pads use results from outer scope — should hoist
func.func @cross_region_use(%arg0: tensor<10x7xf32>, %arg1: tensor<10x8xf32>) -> tensor<10x8xf32> {
    %zero = stablehlo.constant dense<0.0> : tensor<f32>
    %mr:3 = "enzymexla.multi_rotate"(%arg0) {dimension = 0 : i32, left_amount = 1 : i32, right_amount = 1 : i32} : (tensor<10x7xf32>) -> (tensor<10x7xf32>, tensor<10x7xf32>, tensor<10x7xf32>)
    %p0 = stablehlo.pad %mr#0, %zero, low = [0, 0], high = [0, 1], interior = [0, 0] : (tensor<10x7xf32>, tensor<f32>) -> tensor<10x8xf32>
    %p1 = stablehlo.pad %mr#1, %zero, low = [0, 0], high = [0, 1], interior = [0, 0] : (tensor<10x7xf32>, tensor<f32>) -> tensor<10x8xf32>
    %p2 = stablehlo.pad %mr#2, %zero, low = [0, 0], high = [0, 1], interior = [0, 0] : (tensor<10x7xf32>, tensor<f32>) -> tensor<10x8xf32>
    %result = stablehlo.while(%iter = %arg1) : tensor<10x8xf32>
      cond {
        %cond = stablehlo.constant dense<true> : tensor<i1>
        stablehlo.return %cond : tensor<i1>
      } do {
        %sum = stablehlo.add %iter, %p0 : tensor<10x8xf32>
        stablehlo.return %sum : tensor<10x8xf32>
      }
    return %result : tensor<10x8xf32>
}

// CHECK-LABEL:   func.func @cross_region_use(
// CHECK:           %[[ZERO:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK:           %[[PAD:.*]] = stablehlo.pad {{.*}}, %[[ZERO]], low = [0, 0], high = [0, 1], interior = [0, 0] : (tensor<10x7xf32>, tensor<f32>) -> tensor<10x8xf32>
// CHECK:           %[[MR:.*]]:3 = "enzymexla.multi_rotate"(%[[PAD]])
// CHECK-NOT:       stablehlo.pad
// CHECK:           stablehlo.while


// Positive: only one result used (via pad), other unused — should hoist
func.func @single_used_result(%arg0: tensor<10x7xf32>, %arg1: tensor<10x7xf32>) -> (tensor<10x8xf32>, tensor<10x8xf32>) {
    %zero = stablehlo.constant dense<0.0> : tensor<f32>
    %0:2 = "enzymexla.multi_rotate"(%arg0) {dimension = 0 : i32, left_amount = 0 : i32, right_amount = 1 : i32} : (tensor<10x7xf32>) -> (tensor<10x7xf32>, tensor<10x7xf32>)
    %p0 = stablehlo.pad %0#0, %zero, low = [0, 0], high = [0, 1], interior = [0, 0] : (tensor<10x7xf32>, tensor<f32>) -> tensor<10x8xf32>
    %p1 = stablehlo.pad %arg1, %zero, low = [0, 0], high = [0, 1], interior = [0, 0] : (tensor<10x7xf32>, tensor<f32>) -> tensor<10x8xf32>
    return %p0, %p1 : tensor<10x8xf32>, tensor<10x8xf32>
}

// CHECK-LABEL:   func.func @single_used_result(
// CHECK:           %[[ZERO:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK:           %[[PAD:.*]] = stablehlo.pad %arg0, %[[ZERO]], low = [0, 0], high = [0, 1], interior = [0, 0] : (tensor<10x7xf32>, tensor<f32>) -> tensor<10x8xf32>
// CHECK:           %[[MR:.*]]:2 = "enzymexla.multi_rotate"(%[[PAD]]) <{dimension = 0 : i32, left_amount = 0 : i32, right_amount = 1 : i32}> : (tensor<10x8xf32>) -> (tensor<10x8xf32>, tensor<10x8xf32>)
// CHECK:           %[[P1:.*]] = stablehlo.pad %arg1, %[[ZERO]], low = [0, 0], high = [0, 1], interior = [0, 0] : (tensor<10x7xf32>, tensor<f32>) -> tensor<10x8xf32>
// CHECK:           return %[[MR]]#0, %[[P1]] : tensor<10x8xf32>, tensor<10x8xf32>
