// RUN: enzymexlamlir-opt %s --affine-cfg | FileCheck %s

// y(i, e) += y(i, e + 1) over rows of a runtime width nd: for the inner loop
// the offset e * nd of a row is invariant, though the dependence analysis
// cannot flatten it, so the analysis reads it as a symbol of its own and
// proves the iterations independent. The rows themselves depend on each
// other. The parallel loop keeps the loop's attributes.
func.func @add_rows(%y: memref<?xf64>, %ne: index, %nd: index) {
  affine.for %e = 0 to %ne {
    affine.for %i = 0 to %nd {
      %a = affine.load %y[%i + (%e + 1) * symbol(%nd)] : memref<?xf64>
      %b = affine.load %y[%i + %e * symbol(%nd)] : memref<?xf64>
      %c = arith.addf %a, %b : f64
      affine.store %c, %y[%i + %e * symbol(%nd)] : memref<?xf64>
    } {test.kept}
  }
  return
}

// CHECK:  func.func @add_rows(%arg0: memref<?xf64>, %arg1: index, %arg2: index) {
// CHECK-NEXT:   affine.for %arg3 = 0 to %arg1 {
// CHECK-NEXT:     affine.parallel (%arg4) = (0) to (symbol(%arg2)) {
// CHECK-NEXT:       %0 = affine.load %arg0[%arg4 + (%arg3 + 1) * symbol(%arg2)] : memref<?xf64>
// CHECK-NEXT:       %1 = affine.load %arg0[%arg4 + %arg3 * symbol(%arg2)] : memref<?xf64>
// CHECK-NEXT:       %2 = arith.addf %0, %1 : f64
// CHECK-NEXT:       affine.store %2, %arg0[%arg4 + %arg3 * symbol(%arg2)] : memref<?xf64>
// CHECK-NEXT:     } {test.kept}
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }

// The dot product of a row: a reduction over the same invariant offset.
func.func @row_dot(%x: memref<?xf64>, %y: memref<?xf64>, %out: memref<?xf64>, %ne: index, %nd: index) {
  %z = arith.constant 0.0 : f64
  affine.for %e = 0 to %ne {
    %r = affine.for %i = 0 to %nd iter_args(%s = %z) -> (f64) {
      %a = affine.load %x[%i + %e * symbol(%nd)] : memref<?xf64>
      %b = affine.load %y[%i + %e * symbol(%nd)] : memref<?xf64>
      %m = arith.mulf %a, %b : f64
      %t = arith.addf %s, %m : f64
      affine.yield %t : f64
    }
    affine.store %r, %out[%e] : memref<?xf64>
  }
  return
}

// CHECK:  func.func @row_dot(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: memref<?xf64>, %arg3: index, %arg4: index) {
// CHECK-NEXT:   affine.parallel (%arg5) = (0) to (symbol(%arg3)) {
// CHECK-NEXT:     %0 = affine.parallel (%arg6) = (0) to (symbol(%arg4)) reduce ("addf") -> (f64) {
// CHECK-NEXT:       %1 = affine.load %arg0[%arg6 + %arg5 * symbol(%arg4)] : memref<?xf64>
// CHECK-NEXT:       %2 = affine.load %arg1[%arg6 + %arg5 * symbol(%arg4)] : memref<?xf64>
// CHECK-NEXT:       %3 = arith.mulf %1, %2 : f64
// CHECK-NEXT:       affine.yield %3 : f64
// CHECK-NEXT:     }
// CHECK-NEXT:     affine.store %0, %arg2[%arg5] : memref<?xf64>
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }

// Each iteration reads what the one before wrote: not parallel.
func.func @shift_row(%y: memref<?xf64>, %ne: index, %nd: index) {
  affine.for %e = 0 to %ne {
    affine.for %i = 0 to %nd {
      %b = affine.load %y[%i + %e * symbol(%nd)] : memref<?xf64>
      affine.store %b, %y[%i + %e * symbol(%nd) + 1] : memref<?xf64>
    }
  }
  return
}

// CHECK:  func.func @shift_row(%arg0: memref<?xf64>, %arg1: index, %arg2: index) {
// CHECK-NEXT:   affine.for %arg3 = 0 to %arg1 {
// CHECK-NEXT:     affine.for %arg4 = 0 to %arg2 {
// CHECK-NEXT:       %0 = affine.load %arg0[%arg4 + %arg3 * symbol(%arg2)] : memref<?xf64>
// CHECK-NEXT:       affine.store %0, %arg0[%arg4 + %arg3 * symbol(%arg2) + 1] : memref<?xf64>
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }

// The loop writes a buffer it also reads through another view of it, which
// the dependence analysis would take for a buffer of its own: not parallel.
func.func @two_views(%y: memref<?xf64>, %n: index) {
  %v = memref.cast %y : memref<?xf64> to memref<?xf64, strided<[1], offset: ?>>
  affine.for %i = 0 to %n {
    %b = affine.load %v[%i] : memref<?xf64, strided<[1], offset: ?>>
    affine.store %b, %y[%i + 1] : memref<?xf64>
  }
  return
}

// CHECK:  func.func @two_views(%arg0: memref<?xf64>, %arg1: index) {
// CHECK-NEXT:   affine.for %arg2 = 0 to %arg1 {
// CHECK-NEXT:     %0 = affine.load %arg0[%arg2] : memref<?xf64>
// CHECK-NEXT:     affine.store %0, %arg0[%arg2 + 1] : memref<?xf64>
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }

// Three rows of a block written from the first row of the next: the offsets
// (e * 3) * nd, (e * 3 + 1) * nd, (e * 3 + 2) * nd and (e * 3 + 3) * nd are
// the one invariant product e * nd under a coefficient and a multiple of nd,
// not four terms that might meet, so the row loop is parallel; the blocks
// depend on each other.
func.func @block_rows(%y: memref<?xf64>, %ne: index, %nd: index) {
  affine.for %e = 0 to %ne {
    affine.for %i = 0 to %nd {
      %a = affine.load %y[%i + (%e * 3 + 3) * symbol(%nd)] : memref<?xf64>
      affine.store %a, %y[%i + (%e * 3) * symbol(%nd)] : memref<?xf64>
      affine.store %a, %y[%i + (%e * 3 + 1) * symbol(%nd)] : memref<?xf64>
      affine.store %a, %y[%i + (%e * 3 + 2) * symbol(%nd)] : memref<?xf64>
    }
  }
  return
}


// CHECK:  func.func @block_rows(%arg0: memref<?xf64>, %arg1: index, %arg2: index) {
// CHECK-NEXT:   affine.for %arg3 = 0 to %arg1 {
// CHECK-NEXT:     affine.parallel (%arg4) = (0) to (symbol(%arg2)) {
// CHECK-NEXT:       %0 = affine.load %arg0[%arg4 + (%arg3 * 3 + 3) * symbol(%arg2)] : memref<?xf64>
// CHECK-NEXT:       affine.store %0, %arg0[%arg4 + (%arg3 * symbol(%arg2)) * 3] : memref<?xf64>
// CHECK-NEXT:       affine.store %0, %arg0[%arg4 + (%arg3 * 3 + 1) * symbol(%arg2)] : memref<?xf64>
// CHECK-NEXT:       affine.store %0, %arg0[%arg4 + (%arg3 * 3 + 2) * symbol(%arg2)] : memref<?xf64>
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }
