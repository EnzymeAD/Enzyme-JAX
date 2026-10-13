// RUN: enzymexlamlir-opt %s -polygeist-mem2reg | FileCheck %s

// An access at a select between two constant indices names one of two slots
// without saying which; once every other user of the allocation is an access
// at a constant, it becomes a branch on the condition around an access at
// each, and the forwarding resolves both. This is a lambda capture read through
// `const auto &J = shared ? J_shared : J_loc` in MFEM's DG diffusion
// kernel, where the select between the two captures' byte offsets ends up as
// the index of the loads.

module {
  func.func @load_at_select(%c: i1, %a: f64, %b: f64) -> f64 {
    %c1 = arith.constant 1 : index
    %c5 = arith.constant 5 : index
    %cap = memref.alloca() : memref<8xf64>
    affine.store %a, %cap[1] : memref<8xf64>
    affine.store %b, %cap[5] : memref<8xf64>
    %i = arith.select %c, %c1, %c5 : index
    %r = memref.load %cap[%i] : memref<8xf64>
    return %r : f64
  }

  func.func @store_at_select(%c: i1, %a: f64, %v: f64) -> (f64, f64) {
    %c1 = arith.constant 1 : index
    %c5 = arith.constant 5 : index
    %cap = memref.alloca() : memref<8xf64>
    affine.store %a, %cap[1] : memref<8xf64>
    affine.store %a, %cap[5] : memref<8xf64>
    %i = arith.select %c, %c1, %c5 : index
    memref.store %v, %cap[%i] : memref<8xf64>
    %r1 = affine.load %cap[1] : memref<8xf64>
    %r5 = affine.load %cap[5] : memref<8xf64>
    return %r1, %r5 : f64, f64
  }

  // one more access at an index that is neither a constant nor a choice
  // between constants: nothing is split
  func.func @dynamic_access(%c: i1, %a: f64, %b: f64, %j: index) -> (f64, f64) {
    %c1 = arith.constant 1 : index
    %c5 = arith.constant 5 : index
    %cap = memref.alloca() : memref<8xf64>
    affine.store %a, %cap[1] : memref<8xf64>
    affine.store %b, %cap[5] : memref<8xf64>
    %i = arith.select %c, %c1, %c5 : index
    %r = memref.load %cap[%i] : memref<8xf64>
    %s = memref.load %cap[%j] : memref<8xf64>
    return %r, %s : f64, f64
  }

  // a user that is not an access at all, even one the promotion can see
  // past: nothing is split either
  func.func @other_user(%c: i1, %a: f64, %b: f64) -> f64 {
    %c1 = arith.constant 1 : index
    %c5 = arith.constant 5 : index
    %cap = memref.alloca() : memref<8xf64>
    affine.store %a, %cap[1] : memref<8xf64>
    affine.store %b, %cap[5] : memref<8xf64>
    %i = arith.select %c, %c1, %c5 : index
    %r = memref.load %cap[%i] : memref<8xf64>
    func.call @use(%cap) : (memref<8xf64>) -> ()
    return %r : f64
  }
  func.func private @use(memref<8xf64> {llvm.nocapture})
}

// CHECK-LABEL: func.func @load_at_select(
// CHECK-NOT:     memref.alloca
// CHECK:         %[[r:.+]] = scf.if %arg0 -> (f64) {
// CHECK-NEXT:      scf.yield %arg1 : f64
// CHECK-NEXT:    } else {
// CHECK-NEXT:      scf.yield %arg2 : f64
// CHECK-NEXT:    }
// CHECK-NEXT:    return %[[r]] : f64

// CHECK-LABEL: func.func @store_at_select(
// CHECK-NOT:     memref.alloca
// CHECK:         %[[r:.+]]:2 = scf.if %arg0 -> (f64, f64) {
// CHECK-NEXT:      scf.yield %arg2, %arg1 : f64, f64
// CHECK-NEXT:    } else {
// CHECK-NEXT:      scf.yield %arg1, %arg2 : f64, f64
// CHECK-NEXT:    }
// CHECK-NEXT:    return %[[r]]#0, %[[r]]#1 : f64, f64

// CHECK-LABEL: func.func @dynamic_access(
// CHECK:         memref.alloca
// CHECK-NOT:     scf.if
// CHECK:         memref.load
// CHECK:         memref.load

// CHECK-LABEL: func.func @other_user(
// CHECK:         memref.alloca
// CHECK-NOT:     scf.if
// CHECK:         memref.load
// CHECK:         call @use
