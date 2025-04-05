// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(enzyme-hlo-generate-td{patterns=slice_select_to_select_slice},transform-interpreter,enzyme-hlo-remove-transform)" | FileCheck %s

// Basic test case: slice of select with tensor predicate
func.func @slice_select_tensor_predicate(%pred: tensor<8x8xi1>, %on_true: tensor<8x8xf32>, %on_false: tensor<8x8xf32>) -> tensor<3x3xf32> {
  
  %0 = stablehlo.select %pred, %on_true, %on_false : (tensor<8x8xi1>, tensor<8x8xf32>, tensor<8x8xf32>) -> tensor<8x8xf32>
  %1 = stablehlo.slice %0 [2:5:1, 3:6:1] : (tensor<8x8xf32>) -> tensor<3x3xf32>
  return %1 : tensor<3x3xf32>
}


// CHECK: func.func @slice_select_tensor_predicate(%arg0: tensor<8x8xi1>, %arg1: tensor<8x8xf32>, %arg2: tensor<8x8xf32>) -> tensor<3x3xf32> {
// CHECK-NEXT: %0 = stablehlo.slice %arg0 [2:5, 3:6] : (tensor<8x8xi1>) -> tensor<3x3xi1>
// CHECK-NEXT: %1 = stablehlo.slice %arg1 [2:5, 3:6] : (tensor<8x8xf32>) -> tensor<3x3xf32>
// CHECK-NEXT: %2 = stablehlo.slice %arg2 [2:5, 3:6] : (tensor<8x8xf32>) -> tensor<3x3xf32>
// CHECK-NEXT: %3 = stablehlo.select %0, %1, %2 : tensor<3x3xi1>, tensor<3x3xf32>
// CHECK-NEXT: return %3 : tensor<3x3xf32>
// CHECK-NEXT: }

// -----

// Test case: slice of select with scalar predicate
func.func @slice_select_scalar_predicate(%pred: tensor<i1>, %on_true: tensor<8x8xf32>, %on_false: tensor<8x8xf32>) -> tensor<3x3xf32> { 
  %0 = stablehlo.select %pred, %on_true, %on_false : (tensor<i1>, tensor<8x8xf32>, tensor<8x8xf32>) -> tensor<8x8xf32>
  %1 = stablehlo.slice %0 [2:5:1, 3:6:1] : (tensor<8x8xf32>) -> tensor<3x3xf32>
  return %1 : tensor<3x3xf32>
}


// CHECK: func.func @slice_select_scalar_predicate(%arg0: tensor<i1>, %arg1: tensor<8x8xf32>, %arg2: tensor<8x8xf32>) -> tensor<3x3xf32> {
// CHECK-NEXT: %0 = stablehlo.slice %arg1 [2:5, 3:6] : (tensor<8x8xf32>) -> tensor<3x3xf32>
// CHECK-NEXT: %1 = stablehlo.slice %arg2 [2:5, 3:6] : (tensor<8x8xf32>) -> tensor<3x3xf32>
// CHECK-NEXT: %2 = stablehlo.select %arg0, %0, %1 : tensor<i1>, tensor<3x3xf32>
// CHECK-NEXT: return %2 : tensor<3x3xf32>
// CHECK-NEXT: }

// -----

// Test case: no transformation when input is not a select
func.func @no_select_input(%input: tensor<8x8xf32>) -> tensor<3x3xf32> { 
  %0 = stablehlo.slice %input [2:5:1, 3:6:1] : (tensor<8x8xf32>) -> tensor<3x3xf32>
  return %0 : tensor<3x3xf32>
}

//CHECK: func.func @no_select_input(%arg0: tensor<8x8xf32>) -> tensor<3x3xf32> {
//CHECK-NEXT: %0 = stablehlo.slice %arg0 [2:5, 3:6] : (tensor<8x8xf32>) -> tensor<3x3xf32>
//CHECK-NEXT: return %0 : tensor<3x3xf32>
//CHECK-NEXT: }

// -----

// Test case: multi-user select operation
// We don't transform if the select has multiple users
func.func @multi_user_select(%pred: tensor<8x8xi1>, %on_true: tensor<8x8xf32>, %on_false: tensor<8x8xf32>) -> (tensor<3x3xf32>, tensor<8x8xf32>) {
  
  %0 = stablehlo.select %pred, %on_true, %on_false : (tensor<8x8xi1>, tensor<8x8xf32>, tensor<8x8xf32>) -> tensor<8x8xf32>
  %1 = stablehlo.slice %0 [2:5:1, 3:6:1] : (tensor<8x8xf32>) -> tensor<3x3xf32>
  return %1, %0 : tensor<3x3xf32>, tensor<8x8xf32>
}


// CHECK: func.func @multi_user_select(%arg0: tensor<8x8xi1>, %arg1: tensor<8x8xf32>, %arg2: tensor<8x8xf32>) -> (tensor<3x3xf32>, tensor<8x8xf32>) {
// CHECK-NEXT: %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<8x8xi1>, tensor<8x8xf32>
// CHECK-NEXT: %1 = stablehlo.slice %0 [2:5, 3:6] : (tensor<8x8xf32>) -> tensor<3x3xf32>
// CHECK-NEXT: return %1, %0 : tensor<3x3xf32>, tensor<8x8xf32>
// CHECK-NEXT: }
