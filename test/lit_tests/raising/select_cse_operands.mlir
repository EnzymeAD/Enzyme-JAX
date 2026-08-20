// RUN: enzymexlamlir-opt %s --llvm-to-affine-access | FileCheck %s

// select(c, a+b, x+y) distributes to select(c,a,x) + select(c,b,y) -- both
// selects, not one: taking the second addend from the false side whatever
// the condition rewrote MFEM's Hilbert-curve ordering into a different
// curve on every odd-sized grid, and the elements it permuted with it.

module {
  llvm.func @distribute(%c: i1, %a: i32, %b: i32, %x: i32, %y: i32) -> i32 {
    %s1 = arith.addi %a, %b : i32
    %s2 = arith.addi %x, %y : i32
    %r = arith.select %c, %s1, %s2 : i32
    llvm.return %r : i32
  }
}

// CHECK-LABEL: llvm.func @distribute
// CHECK-DAG:     %[[S0:.+]] = arith.select %arg0, %arg1, %arg3 : i32
// CHECK-DAG:     %[[S1:.+]] = arith.select %arg0, %arg2, %arg4 : i32
// CHECK:         %[[R:.+]] = arith.addi %[[S0]], %[[S1]] : i32
// CHECK:         llvm.return %[[R]]
