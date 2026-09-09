// RUN: enzymexlamlir-opt %s --canonicalize-scf-for --split-input-file | FileCheck %s
// RUN: enzymexlamlir-opt %s --affine-cfg --split-input-file | FileCheck %s

// A while whose condition is false runs its before region once and never its
// after region: the condition args, as of the inits, are its results.
func.func @once(%init: i32, %out: memref<?xf64>, %v: f64) -> (i32, i32) {
  %false = arith.constant false
  %c2_i32 = arith.constant 2 : i32
  %r:2 = scf.while (%j = %init) : (i32) -> (i32, i32) {
    %ji = arith.index_castui %j : i32 to index
    memref.store %v, %out[%ji] : memref<?xf64>
    %next = arith.addi %j, %c2_i32 : i32
    scf.condition(%false) %j, %next : i32, i32
  } do {
  ^bb0(%k: i32, %n: i32):
    scf.yield %n : i32
  }
  return %r#0, %r#1 : i32, i32
}

// CHECK-LABEL: func.func @once(
// CHECK-SAME:    %[[INIT:.+]]: i32, %[[OUT:.+]]: memref<?xf64>, %[[V:.+]]: f64
// CHECK-NOT:     scf.while
// CHECK:         %[[JI:.+]] = arith.index_castui %[[INIT]] : i32 to index
// CHECK:         {{memref|affine}}.store %[[V]], %[[OUT]][{{.*}}%[[JI]]
// CHECK:         %[[NEXT:.+]] = arith.addi %[[INIT]], %{{.+}} : i32
// CHECK:         return %[[INIT]], %[[NEXT]] : i32, i32

// -----

// A condition that is not a constant keeps the loop (canonicalize-scf-for
// makes it a for).
func.func @keep(%init: i32, %n: i32, %out: memref<?xf64>, %v: f64) -> i32 {
  %c2_i32 = arith.constant 2 : i32
  %r = scf.while (%j = %init) : (i32) -> i32 {
    %ji = arith.index_castui %j : i32 to index
    memref.store %v, %out[%ji] : memref<?xf64>
    %cond = arith.cmpi slt, %j, %n : i32
    scf.condition(%cond) %j : i32
  } do {
  ^bb0(%k: i32):
    %next = arith.addi %k, %c2_i32 : i32
    scf.yield %next : i32
  }
  return %r : i32
}

// CHECK-LABEL: func.func @keep(
// CHECK:         scf.{{while|for}}
// CHECK:         memref.store
