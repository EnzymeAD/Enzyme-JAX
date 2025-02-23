// RUN: enzymexlamlir-opt --allow-unregistered-dialect --canonicalize-scf-for -split-input-file %s | FileCheck %s
module @simple{
  func.func @do_while(%arg0 : f64) -> (index, f64) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c5 = arith.constant 5 : index

    %result:2 = scf.while (%i = %c0) : (index) -> (index, f64) {
      "before.keepalive"(%i) : (index) -> ()
      %updated = arith.addi %i, %c1 : index
      %cond = arith.cmpi slt, %updated, %c5 : index
      scf.condition(%cond) %updated, %arg0 : index, f64
    } do {
    ^bb0(%new_i: index, %new_2 : f64):
      scf.yield %new_i : index
    }
    
    return %result#0, %result#1 : index, f64
  }
}

// CHECK:  func.func @do_while(%arg0: f64) -> (index, f64) {
// CHECK-NEXT:    %c0 = arith.constant 0 : index
// CHECK-NEXT:    %c1 = arith.constant 1 : index
// CHECK-NEXT:    %c5 = arith.constant 5 : index
// CHECK-NEXT:    %0 = ub.poison : index
// CHECK-NEXT:    %1 = scf.for %arg1 = %c0 to %c5 step %c1 iter_args(%arg2 = %0) -> (index) {
// CHECK-NEXT:      "before.keepalive"(%arg1) : (index) -> ()
// CHECK-NEXT:      %2 = arith.addi %arg1, %c1 : index
// CHECK-NEXT:      scf.yield %2 : index
// CHECK-NEXT:    }
// CHECK-NEXT:    return %1, %arg0 : index, f64
// CHECK-NEXT:  }
