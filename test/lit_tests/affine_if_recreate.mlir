// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(simplify-affine-exprs)" --split-input-file | FileCheck %s

// An affine.if condition the domain does not decide (the symbol is unrelated
// to the loop) is still rebuilt in canonical form: even terms leave a mod 2,
// multiples of the divisor leave a floordiv, and sums are ordered.
func.func @under_loop(%n: index, %m: index, %out: memref<?xf64>, %v: f64) {
  affine.parallel (%i) = (0) to (symbol(%n)) {
    affine.if affine_set<(d0)[s0] : ((d0 * 2 + s0 * 5 + 3) mod 2 - 1 >= 0)>(%i)[%m] {
      memref.store %v, %out[%i] : memref<?xf64>
    }
    affine.if affine_set<(d0)[s0] : (s0 * 3 + d0 - 1 >= 0)>(%i)[%m] {
      memref.store %v, %out[%i] : memref<?xf64>
    }
    affine.if affine_set<(d0)[s0, s1] : ((d0 + s1 * 8 + 9) floordiv 4 - s0 == 0)>(%i)[%m, %n] {
      memref.store %v, %out[%i] : memref<?xf64>
    }
  }
  return
}

// CHECK-DAG: #[[MOD:.+]] = affine_set<(d0)[s0] : ((s0 * 5 + 3) mod 2 - 1 >= 0)>
// CHECK-DAG: #[[SUM:.+]] = affine_set<(d0)[s0] : (d0 + s0 * 3 - 1 >= 0)>
// CHECK-DAG: #[[DIV:.+]] = affine_set<(d0)[s0, s1] : ((d0 + 1) floordiv 4 + -s0 + s1 * 2 + 2 == 0)>
// CHECK-LABEL: func.func @under_loop(
// CHECK: affine.if #[[MOD]](
// CHECK: affine.if #[[SUM]](
// CHECK: affine.if #[[DIV]](

// -----

// The same without any enclosing affine operation, where no domain exists.
func.func @top(%n: index, %i: index, %out: memref<?xf64>, %v: f64) {
  affine.if affine_set<(d0)[s0] : ((d0 * 4 + s0 + 8) floordiv 4 - 3 >= 0)>(%i)[%n] {
    memref.store %v, %out[%i] : memref<?xf64>
  }
  return
}

// CHECK: #[[TOP:.+]] = affine_set<(d0)[s0] : (d0 + s0 floordiv 4 - 1 >= 0)>
// CHECK-LABEL: func.func @top(
// CHECK: affine.if #[[TOP]](
