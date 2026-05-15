// RUN: enzymexlamlir-opt %s --pass-pipeline="any(canonicalize-scf-for)" | FileCheck %s

module {
  func.func private @usevar1(i64) -> ()
  func.func @test_invalid_step(%offset: i64, %bound: i64) -> i64 {
    %c0 = arith.constant 0 : i64
    %c1 = arith.constant 1 : i64
    
    %res = scf.while (%j = %c0) : (i64) -> (i64) {
      // This is the flattened index (induction var + invariant offset)
      %idx = arith.addi %j, %offset : i64
      
      func.call @usevar1(%idx) : (i64) -> ()

      // The condition compares the flattened index with the bound
      %cond = arith.cmpi slt, %idx, %bound : i64
      
      // If true, continue with the current induction variable
      scf.condition(%cond) %j : i64
    } do {
    ^bb0(%j: i64):
      // The actual induction variable is incremented by 1
      %j_next = arith.addi %j, %c1 : i64
      scf.yield %j_next : i64
    }
    return %res : i64
  }
}

// TODO we should support raising this loop.
// However it previously was broken as it used the offset in the before
// as the step size, whereas the actual inductive step is in %1 in after
// CHECK:  func.func @test_invalid_step(%arg0: i64, %arg1: i64) -> i64 {
// CHECK-NEXT:    %c0_i64 = arith.constant 0 : i64
// CHECK-NEXT:    %c1_i64 = arith.constant 1 : i64
// CHECK-NEXT:    %0 = scf.while (%arg2 = %c0_i64) : (i64) -> i64 {
// CHECK-NEXT:      %1 = arith.addi %arg2, %arg0 : i64
// CHECK-NEXT:      func.call @usevar1(%1) : (i64) -> ()
// CHECK-NEXT:      %2 = arith.cmpi slt, %1, %arg1 : i64
// CHECK-NEXT:      scf.condition(%2) %arg2 : i64
// CHECK-NEXT:    } do {
// CHECK-NEXT:    ^bb0(%arg2: i64):
// CHECK-NEXT:      %1 = arith.addi %arg2, %c1_i64 : i64
// CHECK-NEXT:      scf.yield %1 : i64
// CHECK-NEXT:    }
// CHECK-NEXT:    return %0 : i64
// CHECK-NEXT:  }
