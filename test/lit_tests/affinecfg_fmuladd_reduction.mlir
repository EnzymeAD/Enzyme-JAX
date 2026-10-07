// RUN: enzymexlamlir-opt %s --affine-cfg | FileCheck %s

// A dot product accumulated by multiply-adds: fmuladd permits but does not
// require fusing, so it is the sum of the products, and the loop is an addf
// reduction over them.
func.func @dot(%n: index, %x: memref<?xf64>, %y: memref<?xf64>, %out: memref<1xf64>) {
  %zero = arith.constant 0.0 : f64
  %r = affine.for %i = 0 to %n iter_args(%acc = %zero) -> (f64) {
    %a = affine.load %x[%i] : memref<?xf64>
    %b = affine.load %y[%i] : memref<?xf64>
    %s = enzymexla.math.fmuladd %a, %b, %acc : f64
    affine.yield %s : f64
  }
  affine.store %r, %out[0] : memref<1xf64>
  return
}

// CHECK:  func.func @dot(%arg0: index, %arg1: memref<?xf64>, %arg2: memref<?xf64>, %arg3: memref<1xf64>) {
// CHECK-NEXT:   %0 = affine.parallel (%arg4) = (0) to (symbol(%arg0)) reduce ("addf") -> (f64) {
// CHECK-NEXT:     %1 = affine.load %arg1[%arg4] : memref<?xf64>
// CHECK-NEXT:     %2 = affine.load %arg2[%arg4] : memref<?xf64>
// CHECK-NEXT:     %3 = arith.mulf %1, %2 fastmath<contract> : f64
// CHECK-NEXT:     affine.yield %3 : f64
// CHECK-NEXT:   }
// CHECK-NEXT:   affine.store %0, %arg3[0] : memref<1xf64>
// CHECK-NEXT:   return
// CHECK-NEXT: }

// The carried value is also read elsewhere: not an accumulation, left alone.
func.func @read_elsewhere(%n: index, %x: memref<?xf64>, %y: memref<?xf64>, %out: memref<1xf64>) {
  %zero = arith.constant 0.0 : f64
  %r = affine.for %i = 0 to %n iter_args(%acc = %zero) -> (f64) {
    %a = affine.load %x[%i] : memref<?xf64>
    affine.store %acc, %y[%i] : memref<?xf64>
    %s = enzymexla.math.fmuladd %a, %a, %acc : f64
    affine.yield %s : f64
  }
  affine.store %r, %out[0] : memref<1xf64>
  return
}

// CHECK:  func.func @read_elsewhere(%arg0: index, %arg1: memref<?xf64>, %arg2: memref<?xf64>, %arg3: memref<1xf64>) {
// CHECK-NEXT:   %cst = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:   %0 = affine.for %arg4 = 0 to %arg0 iter_args(%arg5 = %cst) -> (f64) {
// CHECK-NEXT:     %1 = affine.load %arg1[%arg4] : memref<?xf64>
// CHECK-NEXT:     affine.store %arg5, %arg2[%arg4] : memref<?xf64>
// CHECK-NEXT:     %2 = enzymexla.math.fmuladd %1, %1, %arg5 : f64
// CHECK-NEXT:     affine.yield %2 : f64
// CHECK-NEXT:   }
// CHECK-NEXT:   affine.store %0, %arg3[0] : memref<1xf64>
// CHECK-NEXT:   return
// CHECK-NEXT: }

// A multiply-add of values the loop does not carry is left fused.
func.func @not_carried(%n: index, %x: memref<?xf64>, %y: memref<?xf64>) {
  affine.for %i = 0 to %n {
    %a = affine.load %x[%i] : memref<?xf64>
    %b = affine.load %y[%i] : memref<?xf64>
    %s = enzymexla.math.fmuladd %a, %b, %b : f64
    affine.store %s, %y[%i] : memref<?xf64>
  }
  return
}

// CHECK:  func.func @not_carried(%arg0: index, %arg1: memref<?xf64>, %arg2: memref<?xf64>) {
// CHECK-NEXT:   affine.parallel (%arg3) = (0) to (symbol(%arg0)) {
// CHECK-NEXT:     %0 = affine.load %arg1[%arg3] : memref<?xf64>
// CHECK-NEXT:     %1 = affine.load %arg2[%arg3] : memref<?xf64>
// CHECK-NEXT:     %2 = enzymexla.math.fmuladd %0, %1, %1 : f64
// CHECK-NEXT:     affine.store %2, %arg2[%arg3] : memref<?xf64>
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }

// An accumulation in a loop that is not parallel otherwise (each iteration
// reads what the previous one wrote): nothing to reduce in parallel, the
// multiply-add stays fused.
func.func @serial(%n: index, %x: memref<?xf64>, %out: memref<1xf64>) {
  %zero = arith.constant 0.0 : f64
  %r = affine.for %i = 1 to %n iter_args(%acc = %zero) -> (f64) {
    %a = affine.load %x[%i - 1] : memref<?xf64>
    affine.store %a, %x[%i] : memref<?xf64>
    %s = enzymexla.math.fmuladd %a, %a, %acc : f64
    affine.yield %s : f64
  }
  affine.store %r, %out[0] : memref<1xf64>
  return
}

// CHECK:  func.func @serial(%arg0: index, %arg1: memref<?xf64>, %arg2: memref<1xf64>) {
// CHECK-NEXT:   %cst = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:   %0 = affine.for %arg3 = 1 to %arg0 iter_args(%arg4 = %cst) -> (f64) {
// CHECK-NEXT:     %1 = affine.load %arg1[%arg3 - 1] : memref<?xf64>
// CHECK-NEXT:     affine.store %1, %arg1[%arg3] : memref<?xf64>
// CHECK-NEXT:     %2 = enzymexla.math.fmuladd %1, %1, %arg4 : f64
// CHECK-NEXT:     affine.yield %2 : f64
// CHECK-NEXT:   }
// CHECK-NEXT:   affine.store %0, %arg2[0] : memref<1xf64>
// CHECK-NEXT:   return
// CHECK-NEXT: }
