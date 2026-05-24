// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(func.func(canonicalize-loops))" %s | FileCheck %s

func.func @reproducer(%cond: i1, %ptr: !llvm.ptr, %cst: i32) -> i32 {
  %c0_i32 = arith.constant 0 : i32
  %c0_index = arith.constant 0 : index
  %c1_index = arith.constant 1 : index
  %c10_index = arith.constant 10 : index
  
  // CHECK: scf.if
  // CHECK: llvm.load
  // CHECK: scf.for
  %0 = scf.if %cond -> (i32) {
    scf.yield %c0_i32 : i32
  } else {
    %val = llvm.load %ptr : !llvm.ptr -> i32
    %res = scf.for %i = %c0_index to %c10_index step %c1_index iter_args(%acc = %c0_i32) -> (i32) {
      %cmp = arith.cmpi slt, %acc, %val : i32
      %next = arith.select %cmp, %acc, %cst : i32
      scf.yield %next : i32
    }
    scf.yield %res : i32
  }
  return %0 : i32
}
