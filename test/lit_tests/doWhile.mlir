// RUN: enzymexlamlir-opt --canonicalize-scf-for %s | FileCheck %s
module {
  func.func @do_while_example() -> index {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c5 = arith.constant 5 : index

    %result = scf.while (%i = %c0) : (index) -> index {
      // "Before" region (executes first)
      %updated = arith.addi %i, %c1 : index
      %cond = arith.cmpi slt, %updated, %c5 : index
      scf.condition(%cond) %updated : index
    } do {
    ^bb0(%new_i: index):
      // "After" region (simple pass-through)
      scf.yield %new_i : index
    }
    
    return %result : index
  }
}
// CHECK-LABEL:   func.func @do_while_example() -> index {
// CHECK:           %[[VAL_0:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_1:.*]] = arith.constant 5 : index
// CHECK:           %[[VAL_2:.*]] = scf.for %[[VAL_3:.*]] = %[[VAL_0]] to %[[VAL_1]] step %[[VAL_0]] iter_args(%[[VAL_4:.*]] = %[[VAL_0]]) -> (index) {
// CHECK:             %[[VAL_5:.*]] = arith.addi %[[VAL_3]], %[[VAL_0]] : index
// CHECK:             scf.yield %[[VAL_5]] : index
// CHECK:           }
// CHECK:           return %[[VAL_2]] : index
// CHECK:         }