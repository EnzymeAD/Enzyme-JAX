// RUN: enzymexlamlir-opt %s -split-input-file -allow-unregistered-dialect --transform-interpreter | FileCheck %s

module {
  func.func @test_normalize_loop() {
    %c5 = arith.constant 5 : index
    %c20 = arith.constant 20 : index
    %c3 = arith.constant 3 : index
    scf.parallel (%arg0) = (%c5) to (%c20) step (%c3) {
      scf.for %i = %c5 to %c20 step %c3 {
        "test.test"() : () -> ()
        "test.test1"(%i) : (index) -> ()
      }
    }
    return
  }

  builtin.module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg2: !transform.any_op) {
      %4 = transform.structured.match ops{["func.func"]} in %arg2 : (!transform.any_op) -> !transform.any_op
      transform.apply_patterns to %4 {
        transform.apply_patterns.raising.normalize_loop
      } : !transform.any_op
      transform.yield
    }
  }
}

// CHECK-LABEL: func @test_normalize_loop(
// CHECK-DAG: %[[C0:.*]] = arith.constant 0
// CHECK-DAG: %[[C1:.*]] = arith.constant 1
// CHECK-DAG: %[[C3:.*]] = arith.constant 3
// CHECK-DAG: %[[C5:.*]] = arith.constant 5
// CHECK: scf.for %[[I:.*]] = %[[C0]] to %[[C5]] step %[[C1]]
// CHECK: %[[MUL:.*]] = arith.muli %[[I]], %[[C3]]
// CHECK: %[[ADD:.*]] = arith.addi %[[MUL]], %[[C5]]
// CHECK: "test.test"
// CHECK: "test.test1"(%[[ADD]])
// CHECK: scf.reduce
