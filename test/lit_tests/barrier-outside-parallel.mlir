// RUN: enzymexlamlir-opt %s --canonicalize | FileCheck %s

// A barrier's effects are read by walking outwards from it, and the walk stops
// at an enclosing parallel op. A barrier outside any parallel op has no such
// stopping point, so the climb runs to the function boundary, where what came
// before is whatever called it.

module {
  func.func @f(%m : memref<10xf32>, %v: f32, %i: index) {
    memref.store %v, %m[%i] : memref<10xf32>
    "enzymexla.barrier"(%i) : (index) -> ()
    return
  }
}

// The effects are unknown rather than empty, so the barrier is not dead.
// CHECK-LABEL: func.func @f
// CHECK-NEXT: memref.store
// CHECK-NEXT: "enzymexla.barrier"
