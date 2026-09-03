// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(simplify-affine-exprs)" --split-input-file %s | FileCheck %s

// A strided remainder loop whose extent the bounds prove to be exactly one
// trip inlines its body at the lower bound.
func.func @remainder(%out: memref<?xf64>, %v: f64) {
  %c2_i32 = arith.constant 2 : i32
  %c2 = arith.constant 2 : index
  affine.parallel (%t) = (0) to (2) {
    %t2 = arith.addi %t, %c2 : index
    %lb = arith.index_castui %t2 : index to i32
    %ub = arith.addi %lb, %c2_i32 : i32
    scf.for %j = %lb to %ub step %c2_i32 : i32 {
      %ji = arith.index_castui %j : i32 to index
      memref.store %v, %out[%ji] : memref<?xf64>
    }
  }
  return
}

// CHECK-LABEL: func.func @remainder(
// CHECK-NOT: scf.for
// CHECK: %[[lb:.+]] = arith.index_castui
// CHECK: %[[ji:.+]] = arith.index_castui %[[lb]]
// CHECK: memref.store %{{.+}}, %{{.+}}[%[[ji]]]

// -----

// A trip count the bounds do not decide keeps the loop.
func.func @keep(%n: i32, %out: memref<?xf64>, %v: f64) {
  %c1_i32 = arith.constant 1 : i32
  affine.parallel (%t) = (0) to (2) {
    %lb = arith.index_castui %t : index to i32
    scf.for %j = %lb to %n step %c1_i32 : i32 {
      %ji = arith.index_castui %j : i32 to index
      memref.store %v, %out[%ji] : memref<?xf64>
    }
  }
  return
}

// CHECK-LABEL: func.func @keep(
// CHECK: scf.for

// -----

// An affine.for whose min upper bound the enclosing bounds decide to one
// trip inlines at its lower bound.
func.func @affine_single_trip(%n: index, %out: memref<?xf64>, %v: f64) {
  affine.parallel (%t) = (0) to (symbol(%n)) {
    affine.for %j = affine_map<(d0) -> (d0)>(%t) to min affine_map<(d0)[s0] -> (d0 + 1, s0)>(%t)[%n] {
      memref.store %v, %out[%j] : memref<?xf64>
    }
  }
  return
}

// CHECK-LABEL: func.func @affine_single_trip(
// CHECK: affine.parallel (%[[t:.+]]) =
// CHECK-NOT: affine.for
// CHECK: memref.store %{{.+}}, %{{.+}}[%[[t]]]

// -----

// A lower bound that is not a bare dimension or symbol is materialized as an
// affine.apply.
func.func @affine_apply_lb(%n: index, %out: memref<?xf64>, %v: f64) {
  affine.parallel (%t) = (0) to (symbol(%n)) {
    affine.for %j = affine_map<(d0) -> (d0 * 2)>(%t) to affine_map<(d0) -> (d0 * 2 + 1)>(%t) {
      memref.store %v, %out[%j] : memref<?xf64>
    }
  }
  return
}

// CHECK-LABEL: func.func @affine_apply_lb(
// CHECK: affine.parallel (%[[t:.+]]) =
// CHECK-NOT: affine.for
// CHECK: %[[j:.+]] = affine.apply #{{.+}}(%[[t]])
// CHECK: memref.store %{{.+}}, %{{.+}}[%[[j]]]

// -----

// A provably zero-trip loop folds to its inits.
func.func @zerotrip(%n: index, %out: memref<?xf64>, %v: f64) -> f64 {
  %c1_i32 = arith.constant 1 : i32
  %r = affine.parallel (%b) = (0) to (symbol(%n)) reduce ("addf") -> f64 {
    %bi = arith.index_castui %b : index to i32
    %ni = arith.index_castui %n : index to i32
    %s = scf.for %j = %ni to %bi step %c1_i32 iter_args(%acc = %v) -> f64 : i32 {
      %a = arith.addf %acc, %acc : f64
      scf.yield %a : f64
    }
    affine.yield %s : f64
  }
  return %r : f64
}

// CHECK-LABEL: func.func @zerotrip(
// CHECK-NOT: scf.for
// CHECK: affine.yield %arg2
