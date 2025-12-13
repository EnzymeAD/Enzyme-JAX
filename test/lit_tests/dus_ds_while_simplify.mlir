// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

// Test 1: Basic case - DS reads the exact slice that DUS wrote (same indices)
// The DS should be replaced with the update value directly
func.func @while_dus_ds_same_indices(%arg0: tensor<10x10xf32>, %arg1: tensor<2x2xf32>) -> (tensor<2x2xf32>, tensor<2x2xf32>) {
  %c = stablehlo.constant dense<0> : tensor<i64>
  %c_0 = stablehlo.constant dense<5> : tensor<i64>
  %c_1 = stablehlo.constant dense<1> : tensor<i64>
  %cst = stablehlo.constant dense<0.000000e+00> : tensor<2x2xf32>
  %0:3 = stablehlo.while(%iterArg = %c, %iterArg_2 = %arg0, %iterArg_3 = %cst) : tensor<i64>, tensor<10x10xf32>, tensor<2x2xf32>
  cond {
    %2 = stablehlo.compare  LT, %iterArg, %c_0 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %2 : tensor<i1>
  } do {
    %2 = stablehlo.dynamic_update_slice %iterArg_2, %arg1, %iterArg, %c : (tensor<10x10xf32>, tensor<2x2xf32>, tensor<i64>, tensor<i64>) -> tensor<10x10xf32>
    %3 = stablehlo.dynamic_slice %2, %iterArg, %c, sizes = [2, 2] : (tensor<10x10xf32>, tensor<i64>, tensor<i64>) -> tensor<2x2xf32>
    %4 = stablehlo.add %iterArg, %c_1 : tensor<i64>
    stablehlo.return %4, %2, %3 : tensor<i64>, tensor<10x10xf32>, tensor<2x2xf32>
  }
  %1 = stablehlo.dynamic_slice %0#1, %c, %c, sizes = [2, 2] : (tensor<10x10xf32>, tensor<i64>, tensor<i64>) -> tensor<2x2xf32>
  return %1, %0#2 : tensor<2x2xf32>, tensor<2x2xf32>
}

// CHECK: func.func @while_dus_ds_same_indices(%arg0: tensor<10x10xf32>, %arg1: tensor<2x2xf32>) -> (tensor<2x2xf32>, tensor<2x2xf32>) {
// CHECK-NEXT:    %c = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %c_0 = stablehlo.constant dense<5> : tensor<i64>
// CHECK-NEXT:    %c_1 = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:    %0:2 = stablehlo.while(%iterArg = %c, %iterArg_2 = %arg0) : tensor<i64>, tensor<10x10xf32>
// CHECK-NEXT:    cond {
// CHECK-NEXT:      %2 = stablehlo.compare  LT, %iterArg, %c_0 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:      stablehlo.return %2 : tensor<i1>
// CHECK-NEXT:    } do {
// CHECK-NEXT:      %2 = stablehlo.dynamic_update_slice %iterArg_2, %arg1, %iterArg, %c : (tensor<10x10xf32>, tensor<2x2xf32>, tensor<i64>, tensor<i64>) -> tensor<10x10xf32>
// CHECK-NEXT:      %3 = stablehlo.add %iterArg, %c_1 {enzymexla.bounds = {{\[}}[1, 5]{{\]}}} : tensor<i64>
// CHECK-NEXT:      stablehlo.return %3, %2 : tensor<i64>, tensor<10x10xf32>
// CHECK-NEXT:    }
// CHECK-NEXT:    %1 = stablehlo.slice %0#1 [0:2, 0:2] : (tensor<10x10xf32>) -> tensor<2x2xf32>
// CHECK-NEXT:    return %1, %arg1 : tensor<2x2xf32>, tensor<2x2xf32>
// CHECK-NEXT:  }


// Test 2: DS reads from non-overlapping region (different constant offset)
// The DS should read from original tensor, not the DUS result
func.func @while_dus_ds_no_overlap(%arg0: tensor<10x10xf32>, %arg1: tensor<2x2xf32>) -> (tensor<2x2xf32>, tensor<2x2xf32>) {
  %c = stablehlo.constant dense<0> : tensor<i64>
  %c_0 = stablehlo.constant dense<5> : tensor<i64>
  %c_1 = stablehlo.constant dense<1> : tensor<i64>
  %c_2 = stablehlo.constant dense<8> : tensor<i64>
  %cst = stablehlo.constant dense<0.000000e+00> : tensor<2x2xf32>
  %0:3 = stablehlo.while(%iterArg = %c, %iterArg_3 = %arg0, %iterArg_4 = %cst) : tensor<i64>, tensor<10x10xf32>, tensor<2x2xf32>
  cond {
    %2 = stablehlo.compare  LT, %iterArg, %c_0 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %2 : tensor<i1>
  } do {
    %2 = stablehlo.add %iterArg, %c_0 : tensor<i64>
    %3 = stablehlo.dynamic_update_slice %iterArg_3, %arg1, %c, %c : (tensor<10x10xf32>, tensor<2x2xf32>, tensor<i64>, tensor<i64>) -> tensor<10x10xf32>
    %4 = stablehlo.dynamic_slice %3, %c_2, %2, sizes = [2, 2] : (tensor<10x10xf32>, tensor<i64>, tensor<i64>) -> tensor<2x2xf32>
    %5 = stablehlo.add %iterArg, %c_1 : tensor<i64>
    %6 = stablehlo.add %iterArg_4, %4 : tensor<2x2xf32>
    stablehlo.return %5, %3, %6 : tensor<i64>, tensor<10x10xf32>, tensor<2x2xf32>
  }
  %1 = stablehlo.dynamic_slice %0#1, %c, %c, sizes = [2, 2] : (tensor<10x10xf32>, tensor<i64>, tensor<i64>) -> tensor<2x2xf32>
  return %1, %0#2 : tensor<2x2xf32>, tensor<2x2xf32>
}

// CHECK: func.func @while_dus_ds_no_overlap(%arg0: tensor<10x10xf32>, %arg1: tensor<2x2xf32>) -> (tensor<2x2xf32>, tensor<2x2xf32>) {
// CHECK-NEXT:   %c = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:   %c_0 = stablehlo.constant dense<5> : tensor<i64>
// CHECK-NEXT:   %c_1 = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:   %c_2 = stablehlo.constant dense<8> : tensor<i64>
// CHECK-NEXT:   %cst = stablehlo.constant dense<0.000000e+00> : tensor<2x2xf32>
// CHECK-NEXT:   %0:3 = stablehlo.while(%iterArg = %c, %iterArg_3 = %arg0, %iterArg_4 = %cst) : tensor<i64>, tensor<10x10xf32>, tensor<2x2xf32>
// CHECK-NEXT:   cond {
// CHECK-NEXT:     %2 = stablehlo.compare  LT, %iterArg, %c_0 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:     stablehlo.return %2 : tensor<i1>
// CHECK-NEXT:   } do {
// CHECK-NEXT:     %2 = stablehlo.add %iterArg, %c_0 {enzymexla.bounds = {{\[}}[5, 9]{{\]}}} : tensor<i64>
// CHECK-NEXT:     %3 = stablehlo.dynamic_update_slice %iterArg_3, %arg1, %c, %c : (tensor<10x10xf32>, tensor<2x2xf32>, tensor<i64>, tensor<i64>) -> tensor<10x10xf32>
// CHECK-NEXT:     %4 = stablehlo.dynamic_slice %iterArg_3, %c_2, %2, sizes = [2, 2] : (tensor<10x10xf32>, tensor<i64>, tensor<i64>) -> tensor<2x2xf32>
// CHECK-NEXT:     %5 = stablehlo.add %iterArg, %c_1 {enzymexla.bounds = {{\[}}[1, 5]{{\]}}} : tensor<i64>
// CHECK-NEXT:     %6 = stablehlo.add %iterArg_4, %4 : tensor<2x2xf32>
// CHECK-NEXT:     stablehlo.return %5, %3, %6 : tensor<i64>, tensor<10x10xf32>, tensor<2x2xf32>
// CHECK-NEXT:   }
// CHECK-NEXT:   %1 = stablehlo.slice %0#1 [0:2, 0:2] : (tensor<10x10xf32>) -> tensor<2x2xf32>
// CHECK-NEXT:   return %1, %0#2 : tensor<2x2xf32>, tensor<2x2xf32>
// CHECK-NEXT: }


// Test 3: Affine index - DS at iterArg+offset reads from region written by DUS at iterArg
// When offset makes them non-overlapping
func.func @while_dus_ds_affine_no_overlap(%arg0: tensor<20x10xf32>, %arg1: tensor<2x2xf32>) -> (tensor<20x10xf32>, tensor<2x2xf32>) {
  %c = stablehlo.constant dense<0> : tensor<i64>
  %c_0 = stablehlo.constant dense<5> : tensor<i64>
  %c_1 = stablehlo.constant dense<1> : tensor<i64>
  %c_2 = stablehlo.constant dense<10> : tensor<i64>
  %cst = stablehlo.constant dense<0.000000e+00> : tensor<2x2xf32>

  %0:3 = stablehlo.while(%iterArg = %c, %iterArg_3 = %arg0, %iterArg_4 = %cst) : tensor<i64>, tensor<20x10xf32>, tensor<2x2xf32>
  cond {
    %1 = stablehlo.compare LT, %iterArg, %c_0 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %1 : tensor<i1>
  } do {
    // DUS writes at position [iterArg, 0]
    %1 = stablehlo.dynamic_update_slice %iterArg_3, %arg1, %iterArg, %c : (tensor<20x10xf32>, tensor<2x2xf32>, tensor<i64>, tensor<i64>) -> tensor<20x10xf32>
    // DS reads from [iterArg + 10, 0] - no overlap since update is only 2x2
    %2 = stablehlo.add %iterArg, %c_2 : tensor<i64>
    %3 = stablehlo.dynamic_slice %1, %2, %c, sizes = [2, 2] : (tensor<20x10xf32>, tensor<i64>, tensor<i64>) -> tensor<2x2xf32>
    %4 = stablehlo.add %iterArg, %c_1 : tensor<i64>
    %5 = stablehlo.add %iterArg_4, %3 : tensor<2x2xf32>
    stablehlo.return %4, %1, %5 : tensor<i64>, tensor<20x10xf32>, tensor<2x2xf32>
  }
  return %0#1, %0#2 : tensor<20x10xf32>, tensor<2x2xf32>
}

// CHECK: func.func @while_dus_ds_affine_no_overlap(%arg0: tensor<20x10xf32>, %arg1: tensor<2x2xf32>) -> (tensor<20x10xf32>, tensor<2x2xf32>) {
// CHECK-NEXT:   %c = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:   %c_0 = stablehlo.constant dense<5> : tensor<i64>
// CHECK-NEXT:   %c_1 = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:   %c_2 = stablehlo.constant dense<10> : tensor<i64>
// CHECK-NEXT:   %cst = stablehlo.constant dense<0.000000e+00> : tensor<2x2xf32>
// CHECK-NEXT:   %0:3 = stablehlo.while(%iterArg = %c, %iterArg_3 = %arg0, %iterArg_4 = %cst) : tensor<i64>, tensor<20x10xf32>, tensor<2x2xf32>
// CHECK-NEXT:   cond {
// CHECK-NEXT:     %1 = stablehlo.compare  LT, %iterArg, %c_0 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:     stablehlo.return %1 : tensor<i1>
// CHECK-NEXT:   } do {
// CHECK-NEXT:     %1 = stablehlo.dynamic_update_slice %iterArg_3, %arg1, %iterArg, %c : (tensor<20x10xf32>, tensor<2x2xf32>, tensor<i64>, tensor<i64>) -> tensor<20x10xf32>
// CHECK-NEXT:     %2 = stablehlo.add %iterArg, %c_2 {enzymexla.bounds = {{\[}}[10, 14]{{\]}}} : tensor<i64>
// DS should read from original tensor since there's no overlap (affine analysis)
// CHECK-NEXT:     %3 = stablehlo.dynamic_slice %iterArg_3, %2, %c, sizes = [2, 2] : (tensor<20x10xf32>, tensor<i64>, tensor<i64>) -> tensor<2x2xf32>
// CHECK-NEXT:     %4 = stablehlo.add %iterArg, %c_1 {enzymexla.bounds = {{\[}}[1, 5]{{\]}}} : tensor<i64>
// CHECK-NEXT:     %5 = stablehlo.add %iterArg_4, %3 : tensor<2x2xf32>
// CHECK-NEXT:     stablehlo.return %4, %1, %5 : tensor<i64>, tensor<20x10xf32>, tensor<2x2xf32>
// CHECK-NEXT:   }
// CHECK-NEXT:   return %0#1, %0#2 : tensor<20x10xf32>, tensor<2x2xf32>
// CHECK-NEXT: }


// Test 4: Affine index with same scale but different offset - overlap case
// DS reads exactly what DUS wrote (offset difference is 0)
func.func @while_dus_ds_affine_same_offset(%arg0: tensor<20x10xf32>, %arg1: tensor<2x2xf32>) -> (tensor<20x10xf32>, tensor<2x2xf32>) {
  %c = stablehlo.constant dense<0> : tensor<i64>
  %c_0 = stablehlo.constant dense<5> : tensor<i64>
  %c_1 = stablehlo.constant dense<1> : tensor<i64>
  %c_2 = stablehlo.constant dense<2> : tensor<i64>
  %cst = stablehlo.constant dense<0.000000e+00> : tensor<2x2xf32>

  %0:3 = stablehlo.while(%iterArg = %c, %iterArg_3 = %arg0, %iterArg_4 = %cst) : tensor<i64>, tensor<20x10xf32>, tensor<2x2xf32>
  cond {
    %1 = stablehlo.compare LT, %iterArg, %c_0 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %1 : tensor<i1>
  } do {
    // Both DUS and DS use the same affine index: iterArg + 2
    %1 = stablehlo.add %iterArg, %c_2 : tensor<i64>
    %2 = stablehlo.dynamic_update_slice %iterArg_3, %arg1, %1, %c : (tensor<20x10xf32>, tensor<2x2xf32>, tensor<i64>, tensor<i64>) -> tensor<20x10xf32>
    %3 = stablehlo.dynamic_slice %2, %1, %c, sizes = [2, 2] : (tensor<20x10xf32>, tensor<i64>, tensor<i64>) -> tensor<2x2xf32>
    %4 = stablehlo.add %iterArg, %c_1 : tensor<i64>
    %5 = stablehlo.add %iterArg_4, %3 : tensor<2x2xf32>
    stablehlo.return %4, %2, %5 : tensor<i64>, tensor<20x10xf32>, tensor<2x2xf32>
  }

  return %0#1, %0#2 : tensor<20x10xf32>, tensor<2x2xf32>
}

// CHECK: func.func @while_dus_ds_affine_same_offset(%arg0: tensor<20x10xf32>, %arg1: tensor<2x2xf32>) -> (tensor<20x10xf32>, tensor<2x2xf32>) {
// CHECK-NEXT:   %c = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:   %c_0 = stablehlo.constant dense<5> : tensor<i64>
// CHECK-NEXT:   %c_1 = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:   %c_2 = stablehlo.constant dense<2> : tensor<i64>
// CHECK-NEXT:   %cst = stablehlo.constant dense<0.000000e+00> : tensor<2x2xf32>
// CHECK-NEXT:   %0:3 = stablehlo.while(%iterArg = %c, %iterArg_3 = %arg0, %iterArg_4 = %cst) : tensor<i64>, tensor<20x10xf32>, tensor<2x2xf32>
// CHECK-NEXT:   cond {
// CHECK-NEXT:     %1 = stablehlo.compare  LT, %iterArg, %c_0 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:     stablehlo.return %1 : tensor<i1>
// CHECK-NEXT:   } do {
// CHECK-NEXT:     %1 = stablehlo.add %iterArg, %c_2 {enzymexla.bounds = {{\[}}[2, 6]{{\]}}} : tensor<i64>
// CHECK-NEXT:     %2 = stablehlo.dynamic_update_slice %iterArg_3, %arg1, %1, %c : (tensor<20x10xf32>, tensor<2x2xf32>, tensor<i64>, tensor<i64>) -> tensor<20x10xf32>
// DS should be replaced with the update value since indices match
// CHECK-NEXT:     %3 = stablehlo.add %iterArg, %c_1 {enzymexla.bounds = {{\[}}[1, 5]{{\]}}} : tensor<i64>
// CHECK-NEXT:     %4 = stablehlo.add %iterArg_4, %arg1 : tensor<2x2xf32>
// CHECK-NEXT:     stablehlo.return %3, %2, %4 : tensor<i64>, tensor<20x10xf32>, tensor<2x2xf32>
// CHECK-NEXT:   }
// CHECK-NEXT:   return %0#1, %0#2 : tensor<20x10xf32>, tensor<2x2xf32>
// CHECK-NEXT: }


// Test 5: Multiple DUS-DS pairs in the same while loop
func.func @while_multiple_dus_ds(%arg0: tensor<10x10xf32>, %arg1: tensor<2x2xf32>, %arg2: tensor<3x3xf32>) -> (tensor<10x10xf32>, tensor<2x2xf32>, tensor<3x3xf32>) {
  %c = stablehlo.constant dense<0> : tensor<i64>
  %c_0 = stablehlo.constant dense<5> : tensor<i64>
  %c_1 = stablehlo.constant dense<1> : tensor<i64>
  %c_2 = stablehlo.constant dense<2> : tensor<i64>
  %cst = stablehlo.constant dense<0.000000e+00> : tensor<2x2xf32>
  %cst_3 = stablehlo.constant dense<0.000000e+00> : tensor<3x3xf32>

  %0:4 = stablehlo.while(%iterArg = %c, %iterArg_4 = %arg0, %iterArg_5 = %cst, %iterArg_6 = %cst_3) : tensor<i64>, tensor<10x10xf32>, tensor<2x2xf32>, tensor<3x3xf32>
  cond {
    %1 = stablehlo.compare LT, %iterArg, %c_0 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %1 : tensor<i1>
  } do {
    // First DUS-DS pair at [iterArg, 0]
    %1 = stablehlo.dynamic_update_slice %iterArg_4, %arg1, %iterArg, %c : (tensor<10x10xf32>, tensor<2x2xf32>, tensor<i64>, tensor<i64>) -> tensor<10x10xf32>
    %2 = stablehlo.dynamic_slice %1, %iterArg, %c, sizes = [2, 2] : (tensor<10x10xf32>, tensor<i64>, tensor<i64>) -> tensor<2x2xf32>

    // Second DUS at [iterArg+2, 5] - DS reads from same position
    %3 = stablehlo.add %iterArg, %c_2 : tensor<i64>
    %4 = stablehlo.dynamic_update_slice %1, %arg2, %3, %c_0 : (tensor<10x10xf32>, tensor<3x3xf32>, tensor<i64>, tensor<i64>) -> tensor<10x10xf32>
    %5 = stablehlo.dynamic_slice %4, %3, %c_0, sizes = [3, 3] : (tensor<10x10xf32>, tensor<i64>, tensor<i64>) -> tensor<3x3xf32>

    %6 = stablehlo.add %iterArg, %c_1 : tensor<i64>
    %7 = stablehlo.add %iterArg_5, %2 : tensor<2x2xf32>
    %8 = stablehlo.add %iterArg_6, %5 : tensor<3x3xf32>
    stablehlo.return %6, %4, %7, %8 : tensor<i64>, tensor<10x10xf32>, tensor<2x2xf32>, tensor<3x3xf32>
  }

  return %0#1, %0#2, %0#3 : tensor<10x10xf32>, tensor<2x2xf32>, tensor<3x3xf32>
}

// CHECK: func.func @while_multiple_dus_ds(%arg0: tensor<10x10xf32>, %arg1: tensor<2x2xf32>, %arg2: tensor<3x3xf32>) -> (tensor<10x10xf32>, tensor<2x2xf32>, tensor<3x3xf32>) {
// CHECK-NEXT:   %c = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:   %c_0 = stablehlo.constant dense<5> : tensor<i64>
// CHECK-NEXT:   %c_1 = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:   %c_2 = stablehlo.constant dense<2> : tensor<i64>
// CHECK-NEXT:   %cst = stablehlo.constant dense<0.000000e+00> : tensor<2x2xf32>
// CHECK-NEXT:   %cst_3 = stablehlo.constant dense<0.000000e+00> : tensor<3x3xf32>
// CHECK-NEXT:   %0:4 = stablehlo.while(%iterArg = %c, %iterArg_4 = %arg0, %iterArg_5 = %cst, %iterArg_6 = %cst_3) : tensor<i64>, tensor<10x10xf32>, tensor<2x2xf32>, tensor<3x3xf32>
// CHECK-NEXT:   cond {
// CHECK-NEXT:     %1 = stablehlo.compare  LT, %iterArg, %c_0 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:     stablehlo.return %1 : tensor<i1>
// CHECK-NEXT:   } do {
// Both DS ops should be simplified - replaced with the update values
// CHECK-NEXT:     %1 = stablehlo.dynamic_update_slice %iterArg_4, %arg1, %iterArg, %c : (tensor<10x10xf32>, tensor<2x2xf32>, tensor<i64>, tensor<i64>) -> tensor<10x10xf32>
// CHECK-NEXT:     %2 = stablehlo.add %iterArg, %c_2 {enzymexla.bounds = {{\[}}[2, 6]{{\]}}} : tensor<i64>
// CHECK-NEXT:     %3 = stablehlo.dynamic_update_slice %1, %arg2, %2, %c_0 : (tensor<10x10xf32>, tensor<3x3xf32>, tensor<i64>, tensor<i64>) -> tensor<10x10xf32>
// CHECK-NEXT:     %4 = stablehlo.add %iterArg, %c_1 {enzymexla.bounds = {{\[}}[1, 5]{{\]}}} : tensor<i64>
// CHECK-NEXT:     %5 = stablehlo.add %iterArg_5, %arg1 : tensor<2x2xf32>
// CHECK-NEXT:     %6 = stablehlo.add %iterArg_6, %arg2 : tensor<3x3xf32>
// CHECK-NEXT:     stablehlo.return %4, %3, %5, %6 : tensor<i64>, tensor<10x10xf32>, tensor<2x2xf32>, tensor<3x3xf32>
// CHECK-NEXT:   }
// CHECK-NEXT:   return %0#1, %0#2, %0#3 : tensor<10x10xf32>, tensor<2x2xf32>, tensor<3x3xf32>
// CHECK-NEXT: }


// Test 6: DS with smaller slice than DUS update (partial read)
func.func @while_dus_ds_partial_read(%arg0: tensor<10x10xf32>, %arg1: tensor<4x4xf32>) -> (tensor<10x10xf32>, tensor<2x2xf32>) {
  %c = stablehlo.constant dense<0> : tensor<i64>
  %c_0 = stablehlo.constant dense<5> : tensor<i64>
  %c_1 = stablehlo.constant dense<1> : tensor<i64>
  %cst = stablehlo.constant dense<0.000000e+00> : tensor<2x2xf32>

  %0:3 = stablehlo.while(%iterArg = %c, %iterArg_3 = %arg0, %iterArg_4 = %cst) : tensor<i64>, tensor<10x10xf32>, tensor<2x2xf32>
  cond {
    %1 = stablehlo.compare LT, %iterArg, %c_0 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %1 : tensor<i1>
  } do {
    // DUS writes 4x4 at [iterArg, 0]
    %1 = stablehlo.dynamic_update_slice %iterArg_3, %arg1, %iterArg, %c : (tensor<10x10xf32>, tensor<4x4xf32>, tensor<i64>, tensor<i64>) -> tensor<10x10xf32>
    // DS reads only 2x2 from [iterArg, 0] - this is a partial read of the update
    %2 = stablehlo.dynamic_slice %1, %iterArg, %c, sizes = [2, 2] : (tensor<10x10xf32>, tensor<i64>, tensor<i64>) -> tensor<2x2xf32>
    %3 = stablehlo.add %iterArg, %c_1 : tensor<i64>
    %4 = stablehlo.add %iterArg_4, %2 : tensor<2x2xf32>
    stablehlo.return %3, %1, %4 : tensor<i64>, tensor<10x10xf32>, tensor<2x2xf32>
  }

  return %0#1, %0#2 : tensor<10x10xf32>, tensor<2x2xf32>
}

// CHECK: func.func @while_dus_ds_partial_read(%arg0: tensor<10x10xf32>, %arg1: tensor<4x4xf32>) -> (tensor<10x10xf32>, tensor<2x2xf32>) {
// CHECK-NEXT:   %c = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:   %c_0 = stablehlo.constant dense<5> : tensor<i64>
// CHECK-NEXT:   %c_1 = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:   %cst = stablehlo.constant dense<0.000000e+00> : tensor<2x2xf32>
// CHECK-NEXT:   %0:3 = stablehlo.while(%iterArg = %c, %iterArg_2 = %arg0, %iterArg_3 = %cst) : tensor<i64>, tensor<10x10xf32>, tensor<2x2xf32>
// CHECK-NEXT:   cond {
// CHECK-NEXT:     %1 = stablehlo.compare  LT, %iterArg, %c_0 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:     stablehlo.return %1 : tensor<i1>
// CHECK-NEXT:   } do {
// CHECK-NEXT:     %1 = stablehlo.dynamic_update_slice %iterArg_2, %arg1, %iterArg, %c : (tensor<10x10xf32>, tensor<4x4xf32>, tensor<i64>, tensor<i64>) -> tensor<10x10xf32>
// DS should be replaced with a slice of the update
// CHECK-NEXT:     %2 = stablehlo.slice %arg1 [0:2, 0:2] : (tensor<4x4xf32>) -> tensor<2x2xf32>
// CHECK-NEXT:     %3 = stablehlo.add %iterArg, %c_1 {enzymexla.bounds = {{\[}}[1, 5]{{\]}}} : tensor<i64>
// CHECK-NEXT:     %4 = stablehlo.add %iterArg_3, %2 : tensor<2x2xf32>
// CHECK-NEXT:     stablehlo.return %3, %1, %4 : tensor<i64>, tensor<10x10xf32>, tensor<2x2xf32>
// CHECK-NEXT:   }
// CHECK-NEXT:   return %0#1, %0#2 : tensor<10x10xf32>, tensor<2x2xf32>
// CHECK-NEXT: }
