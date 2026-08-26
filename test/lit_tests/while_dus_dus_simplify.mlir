// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=while_dus_dus_simplify" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s

// Test 1: Complete overlap with same indices (same SSA value)
// Outer DUS completely covers inner DUS - skip the inner DUS
func.func @while_dus_dus_same_indices_complete_cover(%arg0: tensor<10x10xf32>, %arg1: tensor<4x4xf32>, %arg2: tensor<2x2xf32>) -> tensor<10x10xf32> {
  %c = stablehlo.constant dense<0> : tensor<i64>
  %c_0 = stablehlo.constant dense<5> : tensor<i64>
  %c_1 = stablehlo.constant dense<1> : tensor<i64>
  %0:2 = stablehlo.while(%iterArg = %c, %iterArg_2 = %arg0) : tensor<i64>, tensor<10x10xf32>
  cond {
    %1 = stablehlo.compare LT, %iterArg, %c_0 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %1 : tensor<i1>
  } do {
    // Inner DUS writes 2x2 at [iterArg, 0]
    %1 = stablehlo.dynamic_update_slice %iterArg_2, %arg2, %iterArg, %c : (tensor<10x10xf32>, tensor<2x2xf32>, tensor<i64>, tensor<i64>) -> tensor<10x10xf32>
    // Outer DUS writes 4x4 at [iterArg, 0] - completely covers the inner 2x2
    %2 = stablehlo.dynamic_update_slice %1, %arg1, %iterArg, %c : (tensor<10x10xf32>, tensor<4x4xf32>, tensor<i64>, tensor<i64>) -> tensor<10x10xf32>
    %3 = stablehlo.add %iterArg, %c_1 : tensor<i64>
    stablehlo.return %3, %2 : tensor<i64>, tensor<10x10xf32>
  }
  return %0#1 : tensor<10x10xf32>
}

// CHECK: func.func @while_dus_dus_same_indices_complete_cover(%arg0: tensor<10x10xf32>, %arg1: tensor<4x4xf32>, %arg2: tensor<2x2xf32>) -> tensor<10x10xf32> {
// CHECK-NEXT:   %c = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:   %c_0 = stablehlo.constant dense<5> : tensor<i64>
// CHECK-NEXT:   %c_1 = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:   %0:2 = stablehlo.while(%iterArg = %c, %iterArg_2 = %arg0) : tensor<i64>, tensor<10x10xf32>
// CHECK-NEXT:   cond {
// CHECK-NEXT:     %1 = stablehlo.compare  LT, %iterArg, %c_0 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:     stablehlo.return %1 : tensor<i1>
// CHECK-NEXT:   } do {
// CHECK-NEXT:     %1 = stablehlo.dynamic_update_slice %iterArg_2, %arg1, %iterArg, %c : (tensor<10x10xf32>, tensor<4x4xf32>, tensor<i64>, tensor<i64>) -> tensor<10x10xf32>
// CHECK-NEXT:     %2 = stablehlo.add %iterArg, %c_1 : tensor<i64>
// CHECK-NEXT:     stablehlo.return %2, %1 : tensor<i64>, tensor<10x10xf32>
// CHECK-NEXT:   }
// CHECK-NEXT:   return %0#1 : tensor<10x10xf32>
// CHECK-NEXT: }


// Test 2: Complete overlap with constant indices
// Outer DUS at [0,0] with 4x4 completely covers inner DUS at [1,1] with 2x2
func.func @while_dus_dus_const_indices_complete_cover(%arg0: tensor<10x10xf32>, %arg1: tensor<4x4xf32>, %arg2: tensor<2x2xf32>) -> tensor<10x10xf32> {
  %c = stablehlo.constant dense<0> : tensor<i64>
  %c_0 = stablehlo.constant dense<5> : tensor<i64>
  %c_1 = stablehlo.constant dense<1> : tensor<i64>
  %0:2 = stablehlo.while(%iterArg = %c, %iterArg_2 = %arg0) : tensor<i64>, tensor<10x10xf32>
  cond {
    %1 = stablehlo.compare LT, %iterArg, %c_0 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %1 : tensor<i1>
  } do {
    // Inner DUS writes 2x2 at [1, 1]
    %1 = stablehlo.dynamic_update_slice %iterArg_2, %arg2, %c_1, %c_1 : (tensor<10x10xf32>, tensor<2x2xf32>, tensor<i64>, tensor<i64>) -> tensor<10x10xf32>
    // Outer DUS writes 4x4 at [0, 0] - completely covers inner at [1,1] with size 2x2
    %2 = stablehlo.dynamic_update_slice %1, %arg1, %c, %c : (tensor<10x10xf32>, tensor<4x4xf32>, tensor<i64>, tensor<i64>) -> tensor<10x10xf32>
    %3 = stablehlo.add %iterArg, %c_1 : tensor<i64>
    stablehlo.return %3, %2 : tensor<i64>, tensor<10x10xf32>
  }
  return %0#1 : tensor<10x10xf32>
}

// CHECK: func.func @while_dus_dus_const_indices_complete_cover(%arg0: tensor<10x10xf32>, %arg1: tensor<4x4xf32>, %arg2: tensor<2x2xf32>) -> tensor<10x10xf32> {
// CHECK-NEXT:   %c = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:   %c_0 = stablehlo.constant dense<5> : tensor<i64>
// CHECK-NEXT:   %c_1 = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:   %0:2 = stablehlo.while(%iterArg = %c, %iterArg_2 = %arg0) : tensor<i64>, tensor<10x10xf32>
// CHECK-NEXT:   cond {
// CHECK-NEXT:     %1 = stablehlo.compare  LT, %iterArg, %c_0 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:     stablehlo.return %1 : tensor<i1>
// CHECK-NEXT:   } do {
// CHECK-NEXT:     %1 = stablehlo.dynamic_update_slice %iterArg_2, %arg1, %c, %c : (tensor<10x10xf32>, tensor<4x4xf32>, tensor<i64>, tensor<i64>) -> tensor<10x10xf32>
// CHECK-NEXT:     %2 = stablehlo.add %iterArg, %c_1 : tensor<i64>
// CHECK-NEXT:     stablehlo.return %2, %1 : tensor<i64>, tensor<10x10xf32>
// CHECK-NEXT:   }
// CHECK-NEXT:   return %0#1 : tensor<10x10xf32>
// CHECK-NEXT: }


// Test 3: Partial overlap - outer update is within inner update (full dimension)
// Inner DUS does a full update on first dimension, outer is within
func.func @while_dus_dus_outer_within_inner_full_dim(%arg0: tensor<10x10xf32>, %arg1: tensor<10x4xf32>, %arg2: tensor<2x2xf32>) -> tensor<10x10xf32> {
  %c = stablehlo.constant dense<0> : tensor<i64>
  %c_0 = stablehlo.constant dense<5> : tensor<i64>
  %c_1 = stablehlo.constant dense<1> : tensor<i64>
  %0:2 = stablehlo.while(%iterArg = %c, %iterArg_2 = %arg0) : tensor<i64>, tensor<10x10xf32>
  cond {
    %1 = stablehlo.compare LT, %iterArg, %c_0 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %1 : tensor<i1>
  } do {
    // Inner DUS writes 10x4 at [0, 0] - full update on first dimension
    %1 = stablehlo.dynamic_update_slice %iterArg_2, %arg1, %c, %c : (tensor<10x10xf32>, tensor<10x4xf32>, tensor<i64>, tensor<i64>) -> tensor<10x10xf32>
    // Outer DUS writes 2x2 at [iterArg, 1] - outer is within inner since inner covers full first dim
    %2 = stablehlo.dynamic_update_slice %1, %arg2, %iterArg, %c_1 : (tensor<10x10xf32>, tensor<2x2xf32>, tensor<i64>, tensor<i64>) -> tensor<10x10xf32>
    %3 = stablehlo.add %iterArg, %c_1 : tensor<i64>
    stablehlo.return %3, %2 : tensor<i64>, tensor<10x10xf32>
  }
  return %0#1 : tensor<10x10xf32>
}

// CHECK: func.func @while_dus_dus_outer_within_inner_full_dim(%arg0: tensor<10x10xf32>, %arg1: tensor<10x4xf32>, %arg2: tensor<2x2xf32>) -> tensor<10x10xf32> {
// CHECK-NEXT:   %c = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:   %c_0 = stablehlo.constant dense<5> : tensor<i64>
// CHECK-NEXT:   %c_1 = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:   %0:2 = stablehlo.while(%iterArg = %c, %iterArg_2 = %arg0) : tensor<i64>, tensor<10x10xf32>
// CHECK-NEXT:   cond {
// CHECK-NEXT:     %1 = stablehlo.compare  LT, %iterArg, %c_0 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:     stablehlo.return %1 : tensor<i1>
// CHECK-NEXT:   } do {
// CHECK-NEXT:     %1 = stablehlo.dynamic_update_slice %arg1, %arg2, %iterArg, %c_1 : (tensor<10x4xf32>, tensor<2x2xf32>, tensor<i64>, tensor<i64>) -> tensor<10x4xf32>
// CHECK-NEXT:     %2 = stablehlo.dynamic_update_slice %iterArg_2, %1, %c, %c : (tensor<10x10xf32>, tensor<10x4xf32>, tensor<i64>, tensor<i64>) -> tensor<10x10xf32>
// CHECK-NEXT:     %3 = stablehlo.add %iterArg, %c_1 : tensor<i64>
// CHECK-NEXT:     stablehlo.return %3, %2 : tensor<i64>, tensor<10x10xf32>
// CHECK-NEXT:   }
// CHECK-NEXT:   return %0#1 : tensor<10x10xf32>
// CHECK-NEXT: }


// Test 4: Partial overlap - outer is within inner with constant indices
func.func @while_dus_dus_outer_within_inner_const(%arg0: tensor<10x10xf32>, %arg1: tensor<6x6xf32>, %arg2: tensor<2x2xf32>) -> tensor<10x10xf32> {
  %c = stablehlo.constant dense<0> : tensor<i64>
  %c_0 = stablehlo.constant dense<5> : tensor<i64>
  %c_1 = stablehlo.constant dense<1> : tensor<i64>
  %c_2 = stablehlo.constant dense<2> : tensor<i64>
  %c_3 = stablehlo.constant dense<3> : tensor<i64>
  %0:2 = stablehlo.while(%iterArg = %c, %iterArg_2 = %arg0) : tensor<i64>, tensor<10x10xf32>
  cond {
    %1 = stablehlo.compare LT, %iterArg, %c_0 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %1 : tensor<i1>
  } do {
    // Inner DUS writes 6x6 at [1, 1]
    %1 = stablehlo.dynamic_update_slice %iterArg_2, %arg1, %c_1, %c_1 : (tensor<10x10xf32>, tensor<6x6xf32>, tensor<i64>, tensor<i64>) -> tensor<10x10xf32>
    // Outer DUS writes 2x2 at [3, 3] - within inner's [1,1] to [7,7] region
    %2 = stablehlo.dynamic_update_slice %1, %arg2, %c_3, %c_3 : (tensor<10x10xf32>, tensor<2x2xf32>, tensor<i64>, tensor<i64>) -> tensor<10x10xf32>
    %3 = stablehlo.add %iterArg, %c_1 : tensor<i64>
    stablehlo.return %3, %2 : tensor<i64>, tensor<10x10xf32>
  }
  return %0#1 : tensor<10x10xf32>
}

// CHECK: func.func @while_dus_dus_outer_within_inner_const(%arg0: tensor<10x10xf32>, %arg1: tensor<6x6xf32>, %arg2: tensor<2x2xf32>) -> tensor<10x10xf32> {
// CHECK-NEXT:   %c = stablehlo.constant dense<2> : tensor<i64>
// CHECK-NEXT:   %c_0 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:   %c_1 = stablehlo.constant dense<5> : tensor<i64>
// CHECK-NEXT:   %c_2 = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:   %0:2 = stablehlo.while(%iterArg = %c_0, %iterArg_3 = %arg0) : tensor<i64>, tensor<10x10xf32>
// CHECK-NEXT:   cond {
// CHECK-NEXT:     %1 = stablehlo.compare  LT, %iterArg, %c_1 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:     stablehlo.return %1 : tensor<i1>
// CHECK-NEXT:   } do {
// CHECK-NEXT:     %1 = stablehlo.dynamic_update_slice %arg1, %arg2, %c, %c : (tensor<6x6xf32>, tensor<2x2xf32>, tensor<i64>, tensor<i64>) -> tensor<6x6xf32>
// CHECK-NEXT:     %2 = stablehlo.dynamic_update_slice %iterArg_3, %1, %c_2, %c_2 : (tensor<10x10xf32>, tensor<6x6xf32>, tensor<i64>, tensor<i64>) -> tensor<10x10xf32>
// CHECK-NEXT:     %3 = stablehlo.add %iterArg, %c_2 : tensor<i64>
// CHECK-NEXT:     stablehlo.return %3, %2 : tensor<i64>, tensor<10x10xf32>
// CHECK-NEXT:   }
// CHECK-NEXT:   return %0#1 : tensor<10x10xf32>
// CHECK-NEXT: }

// Test 5: No simplification - outer and inner don't overlap (different regions)
func.func @while_dus_dus_no_overlap(%arg0: tensor<20x20xf32>, %arg1: tensor<4x4xf32>, %arg2: tensor<4x4xf32>) -> tensor<20x20xf32> {
  %c = stablehlo.constant dense<0> : tensor<i64>
  %c_0 = stablehlo.constant dense<5> : tensor<i64>
  %c_1 = stablehlo.constant dense<1> : tensor<i64>
  %c_10 = stablehlo.constant dense<10> : tensor<i64>
  %0:2 = stablehlo.while(%iterArg = %c, %iterArg_2 = %arg0) : tensor<i64>, tensor<20x20xf32>
  cond {
    %1 = stablehlo.compare LT, %iterArg, %c_0 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %1 : tensor<i1>
  } do {
    // Inner DUS writes 4x4 at [0, 0]
    %1 = stablehlo.dynamic_update_slice %iterArg_2, %arg2, %c, %c : (tensor<20x20xf32>, tensor<4x4xf32>, tensor<i64>, tensor<i64>) -> tensor<20x20xf32>
    // Outer DUS writes 4x4 at [10, 10] - no overlap with inner
    %2 = stablehlo.dynamic_update_slice %1, %arg1, %c_10, %c_10 : (tensor<20x20xf32>, tensor<4x4xf32>, tensor<i64>, tensor<i64>) -> tensor<20x20xf32>
    %3 = stablehlo.add %iterArg, %c_1 : tensor<i64>
    stablehlo.return %3, %2 : tensor<i64>, tensor<20x20xf32>
  }
  return %0#1 : tensor<20x20xf32>
}

// CHECK: func.func @while_dus_dus_no_overlap(%arg0: tensor<20x20xf32>, %arg1: tensor<4x4xf32>, %arg2: tensor<4x4xf32>) -> tensor<20x20xf32> {
// CHECK-NEXT:   %c = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:   %c_0 = stablehlo.constant dense<5> : tensor<i64>
// CHECK-NEXT:   %c_1 = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:   %c_2 = stablehlo.constant dense<10> : tensor<i64>
// CHECK-NEXT:   %0:2 = stablehlo.while(%iterArg = %c, %iterArg_3 = %arg0) : tensor<i64>, tensor<20x20xf32>
// CHECK-NEXT:   cond {
// CHECK-NEXT:     %1 = stablehlo.compare  LT, %iterArg, %c_0 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:     stablehlo.return %1 : tensor<i1>
// CHECK-NEXT:   } do {
// CHECK-NEXT:     %1 = stablehlo.dynamic_update_slice %iterArg_3, %arg2, %c, %c : (tensor<20x20xf32>, tensor<4x4xf32>, tensor<i64>, tensor<i64>) -> tensor<20x20xf32>
// CHECK-NEXT:     %2 = stablehlo.dynamic_update_slice %1, %arg1, %c_2, %c_2 : (tensor<20x20xf32>, tensor<4x4xf32>, tensor<i64>, tensor<i64>) -> tensor<20x20xf32>
// CHECK-NEXT:     %3 = stablehlo.add %iterArg, %c_1 : tensor<i64>
// CHECK-NEXT:     stablehlo.return %3, %2 : tensor<i64>, tensor<20x20xf32>
// CHECK-NEXT:   }
// CHECK-NEXT:   return %0#1 : tensor<20x20xf32>
// CHECK-NEXT: }


// Test 6: Outer DUS has covers a full dimension where inner dus is dynamic, we can
//         simplify
func.func @while_dus_dus_outer_full_inner_dynamic(%arg0: tensor<20x20xf32>, %arg1: tensor<4x20xf32>, %arg2: tensor<4x4xf32>) -> tensor<20x20xf32> {
  %c = stablehlo.constant dense<0> : tensor<i64>
  %c_0 = stablehlo.constant dense<5> : tensor<i64>
  %c_1 = stablehlo.constant dense<1> : tensor<i64>
  %c_10 = stablehlo.constant dense<10> : tensor<i64>
  %0:2 = stablehlo.while(%iterArg = %c, %iterArg_2 = %arg0) : tensor<i64>, tensor<20x20xf32>
  cond {
    %1 = stablehlo.compare LT, %iterArg, %c_0 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %1 : tensor<i1>
  } do {
    // Inner DUS writes 4x4 at [i:(i + 4), i:(i + 4)]
    %1 = stablehlo.dynamic_update_slice %iterArg_2, %arg2, %iterArg, %iterArg : (tensor<20x20xf32>, tensor<4x4xf32>, tensor<i64>, tensor<i64>) -> tensor<20x20xf32>
    // Outer DUS writes 4x20 at [i:(i + 4), :]
    %2 = stablehlo.dynamic_update_slice %1, %arg1, %iterArg, %c : (tensor<20x20xf32>, tensor<4x20xf32>, tensor<i64>, tensor<i64>) -> tensor<20x20xf32>
    // CHECK: %1 = stablehlo.dynamic_update_slice %iterArg_2, %arg1, %iterArg, %c : (tensor<20x20xf32>, tensor<4x20xf32>, tensor<i64>, tensor<i64>) -> tensor<20x20xf32>
    %3 = stablehlo.add %iterArg, %c_1 : tensor<i64>
    stablehlo.return %3, %2 : tensor<i64>, tensor<20x20xf32>
    // CHECK: stablehlo.return %2, %1 : tensor<i64>, tensor<20x20xf32>
  }
  return %0#1 : tensor<20x20xf32>
}
