// RUN: enzymexlamlir-opt %s -split-input-file -allow-unregistered-dialect --transform-interpreter | FileCheck %s

module {
  func.func @test_remove_ivs(%arg0: index, %arg1: index, %arg2: index) -> index {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %result = scf.for %iv = %c0 to %arg1 step %c1 iter_args(%iter_arg = %arg0) -> (index) {
      %next = arith.addi %iter_arg, %arg2 : index
      scf.yield %next : index
    }
    return %result : index
  }

  builtin.module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg2: !transform.any_op) {
      %4 = transform.structured.match ops{["func.func"]} in %arg2 : (!transform.any_op) -> !transform.any_op
      transform.apply_patterns to %4 {
        transform.apply_patterns.raising.remove_ivs
      } : !transform.any_op
      transform.yield
    }
  }
}

// CHECK-LABEL: func @test_remove_ivs(
// CHECK-SAME:  %[[START:.*]]: index, %[[BOUND:.*]]: index, %[[STEP:.*]]: index
// CHECK: %[[MUL:.*]] = arith.muli %[[BOUND]], %[[STEP]]
// CHECK: %[[RET:.*]] = arith.addi %[[MUL]], %[[START]]
// CHECK: return %[[RET]]
