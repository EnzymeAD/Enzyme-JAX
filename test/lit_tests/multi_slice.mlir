// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

// Test basic MultiSliceOp lowering
func.func @test_multi_slice_lower(%arg0: tensor<10x20xf32>) -> (tensor<10x5xf32>, tensor<10x5xf32>, tensor<10x5xf32>) {
  %0:3 = "enzymexla.multi_slice"(%arg0) <{dimension = 1 : si32, left_amount = 1 : si32, right_amount = 1 : si32, start_indices = array<i64: 0, 7>, limit_indices = array<i64: 10, 12>, strides = array<i64: 1, 1>}> : (tensor<10x20xf32>) -> (tensor<10x5xf32>, tensor<10x5xf32>, tensor<10x5xf32>)
  return %0#0, %0#1, %0#2 : tensor<10x5xf32>, tensor<10x5xf32>, tensor<10x5xf32>
}

// CHECK-LABEL: func.func @test_multi_slice_lower
// CHECK-NEXT:    %0 = stablehlo.slice %arg0 [0:10, 6:11] : (tensor<10x20xf32>) -> tensor<10x5xf32>
// CHECK-NEXT:    %1 = stablehlo.slice %arg0 [0:10, 7:12] : (tensor<10x20xf32>) -> tensor<10x5xf32>
// CHECK-NEXT:    %2 = stablehlo.slice %arg0 [0:10, 8:13] : (tensor<10x20xf32>) -> tensor<10x5xf32>
// CHECK-NEXT:    return %0, %1, %2

// Test MultiSliceOp with unused results
func.func @test_multi_slice_unused(%arg0: tensor<10x20xf32>) -> tensor<10x5xf32> {
  %0:5 = "enzymexla.multi_slice"(%arg0) <{dimension = 1 : si32, left_amount = 2 : si32, right_amount = 2 : si32, start_indices = array<i64: 0, 7>, limit_indices = array<i64: 10, 12>, strides = array<i64: 1, 1>}> : (tensor<10x20xf32>) -> (tensor<10x5xf32>, tensor<10x5xf32>, tensor<10x5xf32>, tensor<10x5xf32>, tensor<10x5xf32>)
  return %0#2 : tensor<10x5xf32>
}

// CHECK-LABEL: func.func @test_multi_slice_unused
// CHECK-NEXT:    %0 = stablehlo.slice %arg0 [0:10, 7:12] : (tensor<10x20xf32>) -> tensor<10x5xf32>
// CHECK-NEXT:    return %0

// Test recognizing multiple slices and creating MultiSliceOp
func.func @test_recognize_multi_slice(%arg0: tensor<10x20xf32>) -> (tensor<10x5xf32>, tensor<10x5xf32>, tensor<10x5xf32>) {
  %0 = stablehlo.slice %arg0 [0:10, 6:11] : (tensor<10x20xf32>) -> tensor<10x5xf32>
  %1 = stablehlo.slice %arg0 [0:10, 7:12] : (tensor<10x20xf32>) -> tensor<10x5xf32>
  %2 = stablehlo.slice %arg0 [0:10, 8:13] : (tensor<10x20xf32>) -> tensor<10x5xf32>
  return %0, %1, %2 : tensor<10x5xf32>, tensor<10x5xf32>, tensor<10x5xf32>
}

// CHECK-LABEL: func.func @test_recognize_multi_slice
// CHECK-NEXT:    %0:3 = "enzymexla.multi_slice"(%arg0) <{dimension = 1 : si32, left_amount = 1 : si32, right_amount = 1 : si32, start_indices = array<i64: 0, 7>, limit_indices = array<i64: 10, 12>, strides = array<i64: 1, 1>}> : (tensor<10x20xf32>) -> (tensor<10x5xf32>, tensor<10x5xf32>, tensor<10x5xf32>)
// CHECK-NEXT:    return %0#0, %0#1, %0#2

// Test CSE: reusing MultiSliceOp result
func.func @test_reuse_multi_slice(%arg0: tensor<10x20xf32>) -> (tensor<10x5xf32>, tensor<10x5xf32>) {
  %0:3 = "enzymexla.multi_slice"(%arg0) <{dimension = 1 : si32, left_amount = 1 : si32, right_amount = 1 : si32, start_indices = array<i64: 0, 7>, limit_indices = array<i64: 10, 12>, strides = array<i64: 1, 1>}> : (tensor<10x20xf32>) -> (tensor<10x5xf32>, tensor<10x5xf32>, tensor<10x5xf32>)
  %1 = stablehlo.slice %arg0 [0:10, 6:11] : (tensor<10x20xf32>) -> tensor<10x5xf32>
  return %0#0, %1 : tensor<10x5xf32>, tensor<10x5xf32>
}

// CHECK-LABEL: func.func @test_reuse_multi_slice
// CHECK-NEXT:    %0:3 = "enzymexla.multi_slice"(%arg0) <{dimension = 1 : si32, left_amount = 1 : si32, right_amount = 1 : si32, start_indices = array<i64: 0, 7>, limit_indices = array<i64: 10, 12>, strides = array<i64: 1, 1>}> : (tensor<10x20xf32>) -> (tensor<10x5xf32>, tensor<10x5xf32>, tensor<10x5xf32>)
// CHECK-NEXT:    return %0#0, %0#0
