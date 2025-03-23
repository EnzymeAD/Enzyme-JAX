// RUN: enzymexlamlir-opt --canonicalize-scf-for %s | FileCheck %s

func.func @test_if_yield_movement(%cond: i1, %a: i32, %b: i32) -> (i32, i32) {
  %0:2 = scf.if %cond -> (i32, i32) {
    %1 = arith.addi %a, %b : i32
    scf.yield %1, %1  : i32, i32
  } else {
    %1 = arith.addi %a, %b : i32
    %2 = arith.addi %1, %b : i32
    scf.yield %1, %2 : i32, i32
  }
  
  return %0#0, %0#1 : i32, i32
}

// CHECK:  func.func @test_if_yield_movement(%arg0: i1, %arg1: i32, %arg2: i32) -> (i32, i32) {
// CHECK-NEXT:    %0 = scf.if %arg0 -> (i32) {
// CHECK-NEXT:      scf.yield %arg1 : i32
// CHECK-NEXT:    } else {
// CHECK-NEXT:      %3 = arith.addi %arg1, %arg2 : i32
// CHECK-NEXT:      scf.yield %3 : i32
// CHECK-NEXT:    }
// CHECK-NEXT:    %1 = arith.addi %0, %arg2 : i32
// CHECK-NEXT:    %2 = arith.addi %arg1, %arg2 : i32
// CHECK-NEXT:    return %2, %1 : i32, i32
// CHECK-NEXT:  }