// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(simplify-affine-exprs)" --split-input-file | FileCheck %s

// A size rounded up to whole blocks is never below the size: the min a
// launch guard folded into the bound keeps only the size.
func.func @onesym(%n: index, %m: memref<?xf64>) {
  %cst = arith.constant 1.0 : f64
  affine.parallel (%i) = (0) to (min(symbol(%n), ((symbol(%n) + 255) floordiv 256) * 256)) {
    affine.store %cst, %m[%i] : memref<?xf64>
  }
  return
}

// CHECK-LABEL: func.func @onesym(
// CHECK: affine.parallel (%{{.+}}) = (0) to (symbol(%{{.+}})) {

// -----

// The same relation said through two operands: the grid arrives as its own
// value, computed from the size by arithmetic the pruner expands.
func.func @twosym(%n: i32, %m: memref<?xf64>) {
  %cst = arith.constant 1.0 : f64
  %c255 = arith.constant 255 : i32
  %c256 = arith.constant 256 : i32
  %a = arith.addi %n, %c255 overflow<nsw> : i32
  %d = arith.divsi %a, %c256 : i32
  %grid = arith.index_cast %d : i32 to index
  %nn = arith.index_cast %n : i32 to index
  affine.parallel (%i) = (0) to (min(symbol(%nn), symbol(%grid) * 256)) {
    affine.store %cst, %m[%i] : memref<?xf64>
  }
  return
}

// CHECK-LABEL: func.func @twosym(
// CHECK: affine.parallel (%{{.+}}) = (0) to (symbol(%{{.+}})) {

// -----

// Unrelated sizes stay a min.
func.func @unrelated(%n: index, %k: index, %m: memref<?xf64>) {
  %cst = arith.constant 1.0 : f64
  affine.parallel (%i) = (0) to (min(symbol(%n), symbol(%k))) {
    affine.store %cst, %m[%i] : memref<?xf64>
  }
  return
}

// CHECK-LABEL: func.func @unrelated(
// CHECK: affine.parallel (%{{.+}}) = (0) to (min(symbol(%{{.+}}), symbol(%{{.+}}))) {

// -----

// The dual on a lower bound: a max never above the other operand keeps only
// the tighter one.
func.func @lowmax(%n: index, %m: memref<?xf64>) {
  %cst = arith.constant 1.0 : f64
  affine.parallel (%i) = (max(symbol(%n), (symbol(%n) floordiv 4) * 4)) to (symbol(%n) + 128) {
    affine.store %cst, %m[%i] : memref<?xf64>
  }
  return
}

// CHECK-LABEL: func.func @lowmax(
// CHECK: affine.parallel (%{{.+}}) = (symbol(%{{.+}})) to (symbol(%{{.+}}) + 128) {

// -----

// Sign-extending an i1 is not value-preserving (true becomes -1), so the two
// extensions of the same flag are different operands and the min stays.
func.func @i1ext(%n: index, %b: i1, %m: memref<?xf64>) {
  %cst = arith.constant 1.0 : f64
  %s = arith.extsi %b : i1 to i32
  %u = arith.extui %b : i1 to i32
  %x = arith.index_cast %s : i32 to index
  %y = arith.index_cast %u : i32 to index
  affine.parallel (%i) = (0) to (min(symbol(%n) + symbol(%x), symbol(%n) + symbol(%y))) {
    affine.store %cst, %m[%i] : memref<?xf64>
  }
  return
}

// CHECK-LABEL: func.func @i1ext(
// CHECK: affine.parallel (%{{.+}}) = (0) to (min(symbol(%{{.+}}) + symbol(%{{.+}}), symbol(%{{.+}}) + symbol(%{{.+}}))) {
