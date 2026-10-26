// RUN: enzymexlamlir-opt %s --canonicalize-scf-for --split-input-file | FileCheck %s

// Inside the body, an iter arg whose yield is the induction variable is the
// previous iteration's IV -- or the init on the first one. With the init at
// the lower bound the two meet in max(iv - step, lb) with no
// first-iteration test, and the iter arg itself then goes away.

func.func @body_prev(%n: index, %m: memref<?xindex>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %r = scf.for %i = %c0 to %n step %c1 iter_args(%prev = %c0) -> (index) {
    memref.store %prev, %m[%i] : memref<?xindex>
    scf.yield %i : index
  }
  return
}

// CHECK-LABEL: func.func @body_prev
// CHECK:         scf.for %[[IV:.+]] = %c0 to %arg0 step %c1 {
// CHECK-NEXT:      %[[PREV:.+]] = arith.subi %[[IV]], %c1 : index
// CHECK-NEXT:      %[[CLAMP:.+]] = arith.maxsi %[[PREV]], %c0 : index
// CHECK-NEXT:      memref.store %[[CLAMP]], %{{.+}}[%[[IV]]] : memref<?xindex>
// CHECK-NEXT:    }

// -----

// An init away from the lower bound keeps a first-iteration select.

func.func @body_prev_offset_init(%n: index, %init: index, %m: memref<?xindex>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %r = scf.for %i = %c0 to %n step %c1 iter_args(%prev = %init) -> (index) {
    memref.store %prev, %m[%i] : memref<?xindex>
    scf.yield %i : index
  }
  return
}

// CHECK-LABEL: func.func @body_prev_offset_init
// CHECK:         scf.for %[[IV:.+]] = %c0 to %arg0 step %c1 {
// CHECK-DAG:       %[[FIRST:.+]] = arith.cmpi eq, %[[IV]], %c0 : index
// CHECK-DAG:       %[[PREV:.+]] = arith.subi %[[IV]], %c1 : index
// CHECK:           %[[SEL:.+]] = arith.select %[[FIRST]], %arg1, %[[PREV]] : index
// CHECK-NEXT:      memref.store %[[SEL]], %{{.+}}[%[[IV]]] : memref<?xindex>
// CHECK-NEXT:    }
