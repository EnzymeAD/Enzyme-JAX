// RUN: enzymexlamlir-opt --affine-cfg --split-input-file %s | FileCheck %s

// Turning `icmp ne %s, 0` into a constraint needs %s known non-negative, which
// the pass asks by composing an affine map over it with no rewriter. Making an
// operand a valid symbol can mean hoisting the op that defines it, and with no
// rewriter there is nothing to hoist it with, so the question has no answer.
// %s lives inside the loop and is one of those: the pass has to take that for a
// no rather than for a value it may go on to use.

func.func @ne_probe(%a: i32, %b: i32, %m: memref<?xi32>) {
  %c0_i32 = arith.constant 0 : i32
  affine.for %i = 0 to 16 {
    %p = arith.cmpi sgt, %a, %c0_i32 : i32
    %q = arith.cmpi sgt, %b, %c0_i32 : i32
    %pi = arith.extui %p : i1 to i32
    %qi = arith.extui %q : i1 to i32
    %s = arith.select %p, %pi, %qi : i32
    %ne = arith.cmpi ne, %s, %c0_i32 : i32
    scf.if %ne {
      memref.store %a, %m[%i] : memref<?xi32>
    }
  }
  return
}

// The answer taken for a no, the select is hoisted on a later round and the
// conditional does become affine over it.

// CHECK:       #[[SET:.+]] = affine_set<()[s0] : (s0 == 0)>
// CHECK-LABEL: func.func @ne_probe
// CHECK:         %[[S:.+]] = arith.select
// CHECK:         %[[I:.+]] = arith.index_cast %[[S]] : i32 to index
// CHECK:         affine.parallel (%[[IV:.+]]) = (0) to (16)
// CHECK-NEXT:      affine.if #[[SET]]()[%[[I]]]
// CHECK:           } else {
// CHECK-NEXT:        affine.store %arg0, %arg2[%[[IV]]]
