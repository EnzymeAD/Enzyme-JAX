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

// -----

// An iter arg whose yield is an addi of two other values entirely -- here an
// outer accumulator plus the inner loop's IV, the flattened counter of a
// nested tensor-product loop -- is not an induction variable, and its final
// value is not init + count * step. Treating it as one replaced MFEM's
// H1_QuadrilateralElement::CalcShape counter with poison: every row of shape
// values landed in row zero's slots and the rest stayed unwritten, so every
// element mass matrix went singular and CholeskyFactors::Factor aborted.

module {
  func.func @flat_counter(%n: index, %m: index) -> index {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %flat = scf.for %j = %c0 to %n step %c1 iter_args(%o = %c0) -> (index) {
      %inner = scf.for %i = %c0 to %m step %c1 iter_args(%u = %o) -> (index) {
        %next = arith.addi %o, %i : index
        scf.yield %next : index
      }
      scf.yield %inner : index
    }
    return %flat : index
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

// CHECK-LABEL: func @flat_counter(
// CHECK: %[[OUT:.*]] = scf.for %{{.*}} iter_args(%[[O:.*]] = %{{.*}}) -> (index) {
// CHECK:   %[[IN:.*]] = scf.for %[[I:.*]] = %{{.*}} iter_args(%{{.*}} = %[[O]]) -> (index) {
// CHECK:     %[[NEXT:.*]] = arith.addi %[[O]], %[[I]]
// CHECK:     scf.yield %[[NEXT]]
// CHECK:   scf.yield %[[IN]]
// CHECK: return %[[OUT]]
