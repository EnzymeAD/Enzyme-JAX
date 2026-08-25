// RUN: enzymexlamlir-opt --affine-cfg --split-input-file %s | FileCheck %s

#set = affine_set<(d0)[s0] : (-d0 + s0 - 1 >= 0)>

// A guard covering the whole body folds into the bound even past a local
// allocation: an iteration the tightened bound drops only ever allocated.
func.func @guard_past_alloca(%grid: index, %n: index, %out: memref<?xf64>) {
  %cst = arith.constant 1.0 : f64
  affine.parallel (%i) = (0) to (symbol(%grid) * 256) {
    %alloca = memref.alloca() : memref<196xf64>
    affine.if #set(%i)[%n] {
      affine.store %cst, %alloca[0] : memref<196xf64>
      %v = affine.load %alloca[0] : memref<196xf64>
      affine.store %v, %out[%i] : memref<?xf64>
    }
  }
  return
}

// CHECK-LABEL: func.func @guard_past_alloca(
// CHECK: affine.parallel (%{{.+}}) = (0) to (min(symbol(%{{.+}}), symbol(%{{.+}}) * 256)) {
// CHECK-NEXT: memref.alloca
// CHECK-NOT: affine.if

// -----

#set = affine_set<(d0)[s0] : (-d0 + s0 - 1 >= 0)>

// A store outside the guard is an effect every iteration must keep.
func.func @store_outside_guard(%grid: index, %n: index, %out: memref<?xf64>) {
  %cst = arith.constant 1.0 : f64
  affine.parallel (%i) = (0) to (symbol(%grid) * 256) {
    affine.store %cst, %out[%i] : memref<?xf64>
    affine.if #set(%i)[%n] {
      affine.store %cst, %out[%i + 1] : memref<?xf64>
    }
  }
  return
}

// CHECK-LABEL: func.func @store_outside_guard(
// CHECK: affine.parallel (%{{.+}}) = (0) to (symbol(%{{.+}}) * 256) {
// CHECK: affine.if
