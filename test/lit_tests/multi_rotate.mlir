// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

// Test basic MultiRotateOp lowering
func.func @test_multi_rotate_lower(%arg0: tensor<10x20xf32>) -> (tensor<10x20xf32>, tensor<10x20xf32>, tensor<10x20xf32>) {
  %0:3 = "enzymexla.multi_rotate"(%arg0) <{dimension = 1 : si32, left_amount = 1 : si32, right_amount = 1 : si32}> : (tensor<10x20xf32>) -> (tensor<10x20xf32>, tensor<10x20xf32>, tensor<10x20xf32>)
  return %0#0, %0#1, %0#2 : tensor<10x20xf32>, tensor<10x20xf32>, tensor<10x20xf32>
}

// CHECK-LABEL: func.func @test_multi_rotate_lower
// CHECK-NEXT:    %0 = "enzymexla.rotate"(%arg0) <{amount = 1 : si32, dimension = 1 : si32}> : (tensor<10x20xf32>) -> tensor<10x20xf32>
// CHECK-NEXT:    %1 = "enzymexla.rotate"(%arg0) <{amount = 0 : si32, dimension = 1 : si32}> : (tensor<10x20xf32>) -> tensor<10x20xf32>
// CHECK-NEXT:    %2 = "enzymexla.rotate"(%arg0) <{amount = -1 : si32, dimension = 1 : si32}> : (tensor<10x20xf32>) -> tensor<10x20xf32>
// CHECK-NEXT:    return %0, %1, %2

// Test MultiRotateOp with unused results
func.func @test_multi_rotate_unused(%arg0: tensor<10x20xf32>) -> tensor<10x20xf32> {
  %0:5 = "enzymexla.multi_rotate"(%arg0) <{dimension = 1 : si32, left_amount = 2 : si32, right_amount = 2 : si32}> : (tensor<10x20xf32>) -> (tensor<10x20xf32>, tensor<10x20xf32>, tensor<10x20xf32>, tensor<10x20xf32>, tensor<10x20xf32>)
  return %0#2 : tensor<10x20xf32>
}

// CHECK-LABEL: func.func @test_multi_rotate_unused
// CHECK-NEXT:    %0 = "enzymexla.rotate"(%arg0) <{amount = 0 : si32, dimension = 1 : si32}> : (tensor<10x20xf32>) -> tensor<10x20xf32>
// CHECK-NEXT:    return %0

// Test recognizing multiple rotates and creating MultiRotateOp
func.func @test_recognize_multi_rotate(%arg0: tensor<10x20xf32>) -> (tensor<10x20xf32>, tensor<10x20xf32>, tensor<10x20xf32>) {
  %0 = "enzymexla.rotate"(%arg0) <{amount = -1 : si32, dimension = 1 : si32}> : (tensor<10x20xf32>) -> tensor<10x20xf32>
  %1 = "enzymexla.rotate"(%arg0) <{amount = 0 : si32, dimension = 1 : si32}> : (tensor<10x20xf32>) -> tensor<10x20xf32>
  %2 = "enzymexla.rotate"(%arg0) <{amount = 1 : si32, dimension = 1 : si32}> : (tensor<10x20xf32>) -> tensor<10x20xf32>
  return %0, %1, %2 : tensor<10x20xf32>, tensor<10x20xf32>, tensor<10x20xf32>
}

// CHECK-LABEL: func.func @test_recognize_multi_rotate
// CHECK-NEXT:    %0:3 = "enzymexla.multi_rotate"(%arg0) <{dimension = 1 : si32, left_amount = 1 : si32, right_amount = 1 : si32}> : (tensor<10x20xf32>) -> (tensor<10x20xf32>, tensor<10x20xf32>, tensor<10x20xf32>)
// CHECK-NEXT:    return %0#2, %0#1, %0#0

// Test CSE: reusing MultiRotateOp result
func.func @test_reuse_multi_rotate(%arg0: tensor<10x20xf32>) -> (tensor<10x20xf32>, tensor<10x20xf32>) {
  %0:3 = "enzymexla.multi_rotate"(%arg0) <{dimension = 1 : si32, left_amount = 1 : si32, right_amount = 1 : si32}> : (tensor<10x20xf32>) -> (tensor<10x20xf32>, tensor<10x20xf32>, tensor<10x20xf32>)
  %1 = "enzymexla.rotate"(%arg0) <{amount = 1 : si32, dimension = 1 : si32}> : (tensor<10x20xf32>) -> tensor<10x20xf32>
  return %0#0, %1 : tensor<10x20xf32>, tensor<10x20xf32>
}

// CHECK-LABEL: func.func @test_reuse_multi_rotate
// CHECK-NEXT:    %0:3 = "enzymexla.multi_rotate"(%arg0) <{dimension = 1 : si32, left_amount = 1 : si32, right_amount = 1 : si32}> : (tensor<10x20xf32>) -> (tensor<10x20xf32>, tensor<10x20xf32>, tensor<10x20xf32>)
// CHECK-NEXT:    return %0#0, %0#0
