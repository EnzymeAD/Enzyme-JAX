// RUN: enzymexlamlir-opt %s --affine-cfg | FileCheck %s

// The check that a rotated nest runs, around loops that run no iteration
// where it fails (one under a further check of another count): the loops
// run unconditionally.
func.func @nest(%n: index, %m: index, %x: memref<?xf64>, %y: memref<?xf64>) {
  %cst = arith.constant 0.0 : f64
  affine.if affine_set<()[s0] : (s0 - 1 >= 0)>()[%n] {
    affine.for %i = 0 to %n {
      affine.store %cst, %x[%i] : memref<?xf64>
    }
    affine.if affine_set<()[s0] : (s0 - 1 >= 0)>()[%m] {
      affine.for %i = 0 to %n {
        affine.for %j = 0 to %m {
          affine.store %cst, %y[%i + %j] : memref<?xf64>
        }
      }
    }
  }
  return
}

// CHECK:  func.func @nest(%arg0: index, %arg1: index, %arg2: memref<?xf64>, %arg3: memref<?xf64>) {
// CHECK-NEXT:   %cst = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:   affine.parallel (%arg4) = (0) to (symbol(%arg0)) {
// CHECK-NEXT:     affine.store %cst, %arg2[%arg4] : memref<?xf64>
// CHECK-NEXT:   }
// CHECK-NEXT:   affine.if #set()[%arg1] {
// CHECK-NEXT:     affine.for %arg4 = 0 to %arg0 {
// CHECK-NEXT:       affine.parallel (%arg5) = (0) to (symbol(%arg1)) {
// CHECK-NEXT:         affine.store %cst, %arg3[%arg4 + %arg5] : memref<?xf64>
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }

// A store outside the loops runs only where the check holds: the check
// stays.
func.func @store_outside(%n: index, %x: memref<?xf64>) {
  %cst = arith.constant 0.0 : f64
  affine.if affine_set<()[s0] : (s0 - 1 >= 0)>()[%n] {
    affine.store %cst, %x[0] : memref<?xf64>
    affine.for %i = 0 to %n {
      affine.store %cst, %x[%i] : memref<?xf64>
    }
  }
  return
}

// CHECK:  func.func @store_outside(%arg0: index, %arg1: memref<?xf64>) {
// CHECK-NEXT:   %cst = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:   affine.if #set()[%arg0] {
// CHECK-NEXT:     affine.store %cst, %arg1[0] : memref<?xf64>
// CHECK-NEXT:     affine.parallel (%arg2) = (0) to (symbol(%arg0)) {
// CHECK-NEXT:       affine.store %cst, %arg1[%arg2] : memref<?xf64>
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }

// A loop over another count can run where the check fails: the check stays.
func.func @other_count(%n: index, %k: index, %x: memref<?xf64>) {
  %cst = arith.constant 0.0 : f64
  affine.if affine_set<()[s0] : (s0 - 1 >= 0)>()[%n] {
    affine.for %i = 0 to %k {
      affine.store %cst, %x[%i] : memref<?xf64>
    }
  }
  return
}

// CHECK:  func.func @other_count(%arg0: index, %arg1: index, %arg2: memref<?xf64>) {
// CHECK-NEXT:   %cst = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:   affine.if #set()[%arg0] {
// CHECK-NEXT:     affine.parallel (%arg3) = (0) to (symbol(%arg1)) {
// CHECK-NEXT:       affine.store %cst, %arg2[%arg3] : memref<?xf64>
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }

// The branch's loop runs to the n checked, but its index math reads
// max(n, 1), which the check decides: that becomes n before the check goes.
func.func @max_read(%n32: i32, %x: memref<?xf64>) {
  %c1 = arith.constant 1 : i32
  %cst = arith.constant 0.0 : f64
  %n = arith.index_cast %n32 : i32 to index
  %m32 = arith.maxsi %n32, %c1 : i32
  %m = arith.index_cast %m32 : i32 to index
  affine.if affine_set<()[s0] : (s0 - 1 >= 0)>()[%n] {
    affine.for %i = 0 to %n {
      affine.store %cst, %x[%i + symbol(%m)] : memref<?xf64>
    }
  }
  return
}

// CHECK:  func.func @max_read(%arg0: i32, %arg1: memref<?xf64>) {
// CHECK-NEXT:   %cst = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:   %0 = arith.index_cast %arg0 : i32 to index
// CHECK-NEXT:   affine.parallel (%arg2) = (0) to (symbol(%0)) {
// CHECK-NEXT:     affine.store %cst, %arg1[%arg2 + symbol(%0)] : memref<?xf64>
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }
