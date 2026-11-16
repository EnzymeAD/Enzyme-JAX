// RUN: enzymexlamlir-opt --canonicalize-scf-for --split-input-file %s | FileCheck %s

module {
func.func @foo(%arg0: i32) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c1_i32 = arith.constant 1 : i32
  %c0_i32 = arith.constant 0 : i32
  %1 = scf.for %arg1 = %c0 to %c1 step %c1 iter_args(%arg2 = %c0_i32) -> (i32) {
    %2 = arith.index_cast %arg1 : index to i32
    %4:3 = scf.while (%arg3 = %2, %arg4 = %2) : (i32, i32) -> (i32, i32, i32) {
      %5 = arith.cmpi slt, %arg3, %arg4 : i32
      scf.condition(%5) %arg4, %arg3, %arg4 : i32, i32, i32
    } do {
    ^bb0(%arg3: i32, %arg4: i32, %arg5: i32):
      %6 = arith.addi %arg4, %c1_i32 : i32
      scf.yield %6, %arg5 : i32, i32
    }
    scf.yield %4#0 : i32
  }
  return %1 : i32
}
}


// The while's second argument is carried back unchanged, so it is %2
// throughout; the loop then never iterates and yields %2, which is 0.

// CHECK-LABEL:   func.func @foo(
// CHECK-SAME:                   %[[VAL_0:.*]]: i32) -> i32
// CHECK-NEXT:      %[[VAL_1:.*]] = arith.constant 0 : i32
// CHECK-NEXT:      return %[[VAL_1]] : i32
// CHECK-NEXT:    }
