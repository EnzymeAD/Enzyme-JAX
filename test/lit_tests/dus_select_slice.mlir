// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(enzyme-hlo-generate-td{patterns=dus_select_slice},transform-interpreter,enzyme-hlo-remove-transform)" | FileCheck %s

// Test case 1: DUS with select where true branch is slice of operand
func.func @dus_select_slice_true(%arg0: tensor<20x6144x12288xf64>, %arg1: tensor<6x6130x12272xf64>, %pred: tensor<i1>, %c0: tensor<i32>, %c1: tensor<i32>, %c2: tensor<i32>) -> tensor<20x6144x12272xf64> {
  %0 = stablehlo.slice %arg0 [0:20, 0:6144, 8:12280] : (tensor<20x6144x12288xf64>) -> tensor<20x6144x12272xf64>
  %1 = stablehlo.slice %arg0 [7:13, 7:6137, 8:12280] : (tensor<20x6144x12288xf64>) -> tensor<6x6130x12272xf64>
  %2 = stablehlo.select %pred, %1, %arg1 : tensor<i1>, tensor<6x6130x12272xf64>
  %3 = stablehlo.dynamic_update_slice %0, %2, %c0, %c1, %c2 : (tensor<20x6144x12272xf64>, tensor<6x6130x12272xf64>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<20x6144x12272xf64>
  return %3 : tensor<20x6144x12272xf64>
}

// CHECK-LABEL: func.func @dus_select_slice_true
// CHECK-SAME: (%[[ARG0:.*]]: tensor<20x6144x12288xf64>, %[[ARG1:.*]]: tensor<6x6130x12272xf64>, %[[PRED:.*]]: tensor<i1>, %[[C0:.*]]: tensor<i32>, %[[C1:.*]]: tensor<i32>, %[[C2:.*]]: tensor<i32>)
// CHECK: %[[SLICE:.*]] = stablehlo.slice %[[ARG0]] [0:20, 0:6144, 8:12280]
// CHECK: %[[DUS:.*]] = stablehlo.dynamic_update_slice %[[SLICE]], %[[ARG1]], %[[C0]], %[[C1]], %[[C2]]
// CHECK: %[[SELECT:.*]] = stablehlo.select %[[PRED]], %[[SLICE]], %[[DUS]]
// CHECK: return %[[SELECT]]

// -----

// Test case 2: DUS with select where false branch is slice of operand
func.func @dus_select_slice_false(%arg0: tensor<20x6144x12288xf64>, %arg1: tensor<6x6130x12272xf64>, %pred: tensor<i1>, %c0: tensor<i32>, %c1: tensor<i32>, %c2: tensor<i32>) -> tensor<20x6144x12272xf64> {
  %0 = stablehlo.slice %arg0 [0:20, 0:6144, 8:12280] : (tensor<20x6144x12288xf64>) -> tensor<20x6144x12272xf64>
  %1 = stablehlo.slice %arg0 [7:13, 7:6137, 8:12280] : (tensor<20x6144x12288xf64>) -> tensor<6x6130x12272xf64>
  %2 = stablehlo.select %pred, %arg1, %1 : tensor<i1>, tensor<6x6130x12272xf64>
  %3 = stablehlo.dynamic_update_slice %0, %2, %c0, %c1, %c2 : (tensor<20x6144x12272xf64>, tensor<6x6130x12272xf64>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<20x6144x12272xf64>
  return %3 : tensor<20x6144x12272xf64>
}

// CHECK-LABEL: func.func @dus_select_slice_false
// CHECK-SAME: (%[[ARG0:.*]]: tensor<20x6144x12288xf64>, %[[ARG1:.*]]: tensor<6x6130x12272xf64>, %[[PRED:.*]]: tensor<i1>, %[[C0:.*]]: tensor<i32>, %[[C1:.*]]: tensor<i32>, %[[C2:.*]]: tensor<i32>)
// CHECK: %[[SLICE:.*]] = stablehlo.slice %[[ARG0]] [0:20, 0:6144, 8:12280]
// CHECK: %[[DUS:.*]] = stablehlo.dynamic_update_slice %[[SLICE]], %[[ARG1]], %[[C0]], %[[C1]], %[[C2]]
// CHECK: %[[SELECT:.*]] = stablehlo.select %[[PRED]], %[[DUS]], %[[SLICE]]
// CHECK: return %[[SELECT]]

// -----

// Test case 3: No transformation when update is not a select
func.func @dus_no_select(%arg0: tensor<20x6144x12288xf64>, %arg1: tensor<6x6130x12272xf64>, %c0: tensor<i32>, %c1: tensor<i32>, %c2: tensor<i32>) -> tensor<20x6144x12272xf64> {
  %0 = stablehlo.slice %arg0 [0:20, 0:6144, 8:12280] : (tensor<20x6144x12288xf64>) -> tensor<20x6144x12272xf64>
  %1 = stablehlo.dynamic_update_slice %0, %arg1, %c0, %c1, %c2 : (tensor<20x6144x12272xf64>, tensor<6x6130x12272xf64>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<20x6144x12272xf64>
  return %1 : tensor<20x6144x12272xf64>
}

// CHECK-LABEL: func.func @dus_no_select
// CHECK: %[[SLICE:.*]] = stablehlo.slice
// CHECK: %[[DUS:.*]] = stablehlo.dynamic_update_slice %[[SLICE]]
// CHECK: return %[[DUS]]

// -----

// Test case 4: No transformation when slice is not from DUS operand
func.func @dus_select_slice_different_operand(%arg0: tensor<20x6144x12288xf64>, %arg1: tensor<20x6144x12288xf64>, %arg2: tensor<6x6130x12272xf64>, %pred: tensor<i1>, %c0: tensor<i32>, %c1: tensor<i32>, %c2: tensor<i32>) -> tensor<20x6144x12272xf64> {
  %0 = stablehlo.slice %arg0 [0:20, 0:6144, 8:12280] : (tensor<20x6144x12288xf64>) -> tensor<20x6144x12272xf64>
  %1 = stablehlo.slice %arg1 [7:13, 7:6137, 8:12280] : (tensor<20x6144x12288xf64>) -> tensor<6x6130x12272xf64>
  %2 = stablehlo.select %pred, %1, %arg2 : tensor<i1>, tensor<6x6130x12272xf64>
  %3 = stablehlo.dynamic_update_slice %0, %2, %c0, %c1, %c2 : (tensor<20x6144x12272xf64>, tensor<6x6130x12272xf64>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<20x6144x12272xf64>
  return %3 : tensor<20x6144x12272xf64>
}

// CHECK-LABEL: func.func @dus_select_slice_different_operand
// CHECK: %[[SLICE0:.*]] = stablehlo.slice %arg0
// CHECK: %[[SLICE1:.*]] = stablehlo.slice %arg1
// CHECK: %[[SELECT:.*]] = stablehlo.select
// CHECK: %[[DUS:.*]] = stablehlo.dynamic_update_slice %[[SLICE0]], %[[SELECT]]
// CHECK: return %[[DUS]]

// -----

// Test case 5: Complex pattern from the issue description
func.func @complex_pattern(%arg20: tensor<20x6144x12288xf64>, %arg_update: tensor<4x1x12272xf64>, %pred: tensor<i1>, %c_6: tensor<i32>, %c_3: tensor<i32>, %c_5: tensor<i32>, %c_2: tensor<i32>) -> tensor<20x6144x12272xf64> {
  // Initial slice of arg20
  %0 = stablehlo.slice %arg20 [0:20, 0:6144, 8:12280] : (tensor<20x6144x12288xf64>) -> tensor<20x6144x12272xf64>
  
  // Another slice of arg20 (smaller region that will be updated)
  %16 = stablehlo.slice %arg20 [7:13, 7:6137, 8:12280] : (tensor<20x6144x12288xf64>) -> tensor<6x6130x12272xf64>
  
  // Create a base tensor to update (using a constant for simplicity in test)
  %cst = stablehlo.constant dense<0.0> : tensor<6x6130x12272xf64>
  
  // DUS on the base tensor
  %32 = stablehlo.dynamic_update_slice %cst, %arg_update, %c_6, %c_3, %c_5 : (tensor<6x6130x12272xf64>, tensor<4x1x12272xf64>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<6x6130x12272xf64>
  
  // Select between original slice and updated tensor
  %40 = stablehlo.select %pred, %16, %32 : tensor<i1>, tensor<6x6130x12272xf64>
  
  // Final DUS into the initial slice - this should be optimized
  %46 = stablehlo.dynamic_update_slice %0, %40, %c_2, %c_2, %c_5 : (tensor<20x6144x12272xf64>, tensor<6x6130x12272xf64>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<20x6144x12272xf64>
  
  return %46 : tensor<20x6144x12272xf64>
}

// CHECK-LABEL: func.func @complex_pattern
// CHECK: %[[SLICE0:.*]] = stablehlo.slice %arg0 [0:20, 0:6144, 8:12280]
// CHECK: %[[CST:.*]] = stablehlo.constant
// CHECK: %[[DUS1:.*]] = stablehlo.dynamic_update_slice %[[CST]], %arg1
// CHECK: %[[DUS2:.*]] = stablehlo.dynamic_update_slice %[[SLICE0]], %[[DUS1]]
// CHECK: %[[SELECT:.*]] = stablehlo.select %arg2, %[[SLICE0]], %[[DUS2]]
// CHECK: return %[[SELECT]]
