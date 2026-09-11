// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(simplify-affine-exprs)" --split-input-file %s | FileCheck %s

// A rotated do-while remainder whose condition is false on its first
// evaluation is one execution of its before region.
func.func @dowhile(%out: memref<?xf64>, %v: f64) {
  %c2_i32 = arith.constant 2 : i32
  %c2 = arith.constant 2 : index
  affine.parallel (%t) = (0) to (2) {
    %t2 = arith.addi %t, %c2 : index
    %init = arith.index_castui %t2 : index to i32
    scf.while (%j = %init) : (i32) -> i32 {
      %ji = arith.index_castui %j : i32 to index
      memref.store %v, %out[%ji] : memref<?xf64>
      %cond = arith.cmpi ult, %j, %c2_i32 : i32
      scf.condition(%cond) %j : i32
    } do {
    ^bb0(%k: i32):
      %next = arith.addi %k, %c2_i32 : i32
      scf.yield %next : i32
    }
  }
  return
}

// CHECK-LABEL: func.func @dowhile(
// CHECK-NOT: scf.while
// CHECK: %[[init:.+]] = arith.index_castui
// CHECK: %[[ji:.+]] = arith.index_castui %[[init]]
// CHECK: memref.store %{{.+}}, %{{.+}}[%[[ji]]]

// -----

// A condition that can hold on the first evaluation keeps the loop.
func.func @keep(%n: i32, %out: memref<?xf64>, %v: f64) {
  %c2_i32 = arith.constant 2 : i32
  affine.parallel (%t) = (0) to (2) {
    %init = arith.index_castui %t : index to i32
    scf.while (%j = %init) : (i32) -> i32 {
      %ji = arith.index_castui %j : i32 to index
      memref.store %v, %out[%ji] : memref<?xf64>
      %cond = arith.cmpi slt, %j, %n : i32
      scf.condition(%cond) %j : i32
    } do {
    ^bb0(%k: i32):
      %next = arith.addi %k, %c2_i32 : i32
      scf.yield %next : i32
    }
  }
  return
}

// CHECK-LABEL: func.func @keep(
// CHECK: scf.while
