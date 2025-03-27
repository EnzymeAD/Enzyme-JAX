// RUN: enzymexlamlir-opt %s -affine-cfg | FileCheck %s

// Define a simple affine set
#set0 = affine_set<(d0) : (d0 >= 0)>

// Basic test case - single result with index_cast
// CHECK-LABEL: func @test_affine_if_index_cast
func.func @test_affine_if_index_cast(%arg0: index) -> index {
  %c1_i64 = arith.constant 1 : i64
  %c2_i64 = arith.constant 2 : i64
  
  // CHECK: %[[RESULT:.*]] = affine.if
  // CHECK:   %[[CAST1:.*]] = arith.index_cast %c1_i64 : i64 to index
  // CHECK:   affine.yield %[[CAST1]]
  // CHECK: else
  // CHECK:   %[[CAST2:.*]] = arith.index_cast %c2_i64 : i64 to index
  // CHECK:   affine.yield %[[CAST2]]
  // CHECK-NOT: arith.index_cast
  
  %0 = affine.if #set0(%arg0) -> (i64) {
    affine.yield %c1_i64 : i64
  } else {
    affine.yield %c2_i64 : i64
  }
  
  %1 = arith.index_cast %0 : i64 to index
  return %1 : index
}

// -----

// Multiple results with only one used in index_cast
// CHECK-LABEL: func @test_multi_result_single_cast
func.func @test_multi_result_single_cast(%arg0: index) -> (i64, index) {
  %c1_i64 = arith.constant 1 : i64
  %c2_i64 = arith.constant 2 : i64
  %c3_i64 = arith.constant 3 : i64
  %c4_i64 = arith.constant 4 : i64
  
  // CHECK: %[[RESULT:.*]]:2 = affine.if
  // CHECK:   affine.yield %c1_i64, %{{.*}}
  // CHECK: else
  // CHECK:   %[[CAST:.*]] = arith.index_cast %c4_i64 : i64 to index
  // CHECK:   affine.yield %c3_i64, %[[CAST]]
  // CHECK-NOT: arith.index_cast %{{.*}}#1
  
  %0:2 = affine.if #set0(%arg0) -> (i64, i64) {
    affine.yield %c1_i64, %c2_i64 : i64, i64
  } else {
    affine.yield %c3_i64, %c4_i64 : i64, i64
  }
  
  %1 = arith.index_cast %0#1 : i64 to index
  return %0#0, %1 : i64, index
}

// -----

// Multiple results with all used in index_cast
// CHECK-LABEL: func @test_multi_result_all_cast
func.func @test_multi_result_all_cast(%arg0: index) -> (index, index) {
  %c1_i64 = arith.constant 1 : i64
  %c2_i64 = arith.constant 2 : i64
  %c3_i64 = arith.constant 3 : i64
  %c4_i64 = arith.constant 4 : i64
  
  // CHECK: %[[RESULT:.*]]:2 = affine.if
  // CHECK:   %[[CAST1:.*]] = arith.index_cast %c1_i64 : i64 to index
  // CHECK:   %[[CAST2:.*]] = arith.index_cast %c2_i64 : i64 to index
  // CHECK:   affine.yield %[[CAST1]], %[[CAST2]]
  // CHECK: else
  // CHECK:   %[[CAST3:.*]] = arith.index_cast %c3_i64 : i64 to index
  // CHECK:   %[[CAST4:.*]] = arith.index_cast %c4_i64 : i64 to index
  // CHECK:   affine.yield %[[CAST3]], %[[CAST4]]
  
  %0:2 = affine.if #set0(%arg0) -> (i64, i64) {
    affine.yield %c1_i64, %c2_i64 : i64, i64
  } else {
    affine.yield %c3_i64, %c4_i64 : i64, i64
  }
  
  %1 = arith.index_cast %0#0 : i64 to index
  %2 = arith.index_cast %0#1 : i64 to index
  return %1, %2 : index, index
}

// -----

// Case where result is used in multiple places - should not optimize
// CHECK-LABEL: func @test_multiple_uses
func.func @test_multiple_uses(%arg0: index) -> (index, i64) {
  %c1_i64 = arith.constant 1 : i64
  %c2_i64 = arith.constant 2 : i64
  
  // CHECK: %[[RESULT:.*]] = affine.if
  // CHECK-NOT: arith.index_cast
  // CHECK:   affine.yield %c1_i64
  // CHECK: else
  // CHECK-NOT: arith.index_cast
  // CHECK:   affine.yield %c2_i64
  // CHECK: arith.index_cast %[[RESULT]] : i64 to index
  
  %0 = affine.if #set0(%arg0) -> (i64) {
    affine.yield %c1_i64 : i64
  } else {
    affine.yield %c2_i64 : i64
  }
  
  %1 = arith.index_cast %0 : i64 to index
  return %1, %0 : index, i64
}

// -----

// Case with multiple index_casts applied to the same result - should not optimize
// CHECK-LABEL: func @test_multiple_casts
func.func @test_multiple_casts(%arg0: index) -> (index, index) {
  %c1_i64 = arith.constant 1 : i64
  %c2_i64 = arith.constant 2 : i64
  
  // CHECK: %[[RESULT:.*]] = affine.if
  // CHECK-NOT: arith.index_cast
  // CHECK:   affine.yield %c1_i64
  // CHECK: else
  // CHECK-NOT: arith.index_cast
  // CHECK:   affine.yield %c2_i64
  // CHECK: %[[CAST1:.*]] = arith.index_cast %[[RESULT]] : i64 to index
  // CHECK: %[[CAST2:.*]] = arith.index_cast %[[RESULT]] : i64 to index
  
  %0 = affine.if #set0(%arg0) -> (i64) {
    affine.yield %c1_i64 : i64
  } else {
    affine.yield %c2_i64 : i64
  }
  
  %1 = arith.index_cast %0 : i64 to index
  %2 = arith.index_cast %0 : i64 to index
  return %1, %2 : index, index
}

// -----

// Case where the result is already an index type - should not optimize
// CHECK-LABEL: func @test_already_index
func.func @test_already_index(%arg0: index) -> index {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  
  // CHECK: %[[RESULT:.*]] = affine.if
  // CHECK-NOT: arith.index_cast
  // CHECK:   affine.yield %c1
  // CHECK: else
  // CHECK-NOT: arith.index_cast
  // CHECK:   affine.yield %c2
  
  %0 = affine.if #set0(%arg0) -> (index) {
    affine.yield %c1 : index
  } else {
    affine.yield %c2 : index
  }
  
  return %0 : index
} 