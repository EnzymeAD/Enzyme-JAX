// RUN: enzymexlamlir-opt %s --affine-cfg | FileCheck %s

// A rotated loop's trip count, max(n, 1), under the check that n >= 1: the
// count is n there (the index math reads n too), and stays max(n, 1)
// outside the check.
func.func @max_checked(%n32: i32, %x: memref<?xf64>, %y: memref<?xindex>) {
  %c1 = arith.constant 1 : i32
  %cst = arith.constant 0.0 : f64
  %n = arith.index_cast %n32 : i32 to index
  %m32 = arith.maxsi %n32, %c1 : i32
  %m = arith.index_cast %m32 : i32 to index
  affine.if affine_set<()[s0] : (s0 - 1 >= 0)>()[%n] {
    affine.for %i = 0 to %m {
      affine.store %cst, %x[%i + symbol(%n)] : memref<?xf64>
    }
  }
  affine.store %m, %y[0] : memref<?xindex>
  return
}

// CHECK:  func.func @max_checked(%arg0: i32, %arg1: memref<?xf64>, %arg2: memref<?xindex>) {
// CHECK-NEXT:   %c1_i32 = arith.constant 1 : i32
// CHECK-NEXT:   %cst = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:   %0 = arith.index_cast %arg0 : i32 to index
// CHECK-NEXT:   %1 = arith.maxsi %arg0, %c1_i32 : i32
// CHECK-NEXT:   %2 = arith.index_cast %1 : i32 to index
// CHECK-NEXT:   affine.parallel (%arg3) = (0) to (symbol(%0)) {
// CHECK-NEXT:     affine.store %cst, %arg1[%arg3 + symbol(%0)] : memref<?xf64>
// CHECK-NEXT:   }
// CHECK-NEXT:   affine.store %2, %arg2[0] : memref<?xindex>
// CHECK-NEXT:   return
// CHECK-NEXT: }

// A min under a check that bounds the value by the constant.
func.func @min_checked(%n32: i32, %x: memref<?xf64>) {
  %c5 = arith.constant 5 : i32
  %cst = arith.constant 0.0 : f64
  %n = arith.index_cast %n32 : i32 to index
  %m32 = arith.minsi %n32, %c5 : i32
  %m = arith.index_cast %m32 : i32 to index
  affine.if affine_set<()[s0] : (-s0 + 5 >= 0)>()[%n] {
    affine.for %i = 0 to %m {
      affine.store %cst, %x[%i] : memref<?xf64>
    }
  }
  return
}

// CHECK:  func.func @min_checked(%arg0: i32, %arg1: memref<?xf64>) {
// CHECK-NEXT:   %cst = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:   %0 = arith.index_cast %arg0 : i32 to index
// CHECK-NEXT:   affine.if #set()[%0] {
// CHECK-NEXT:     affine.parallel (%arg2) = (0) to (symbol(%0)) {
// CHECK-NEXT:       affine.store %cst, %arg1[%arg2] : memref<?xf64>
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }

// The check puts n below the constant: the max is the constant there.
func.func @max_constant(%n32: i32, %x: memref<?xf64>) {
  %c1 = arith.constant 1 : i32
  %cst = arith.constant 0.0 : f64
  %n = arith.index_cast %n32 : i32 to index
  %m32 = arith.maxsi %n32, %c1 : i32
  %m = arith.index_cast %m32 : i32 to index
  affine.if affine_set<()[s0] : (-s0 >= 0)>()[%n] {
    affine.for %i = 0 to %m {
      affine.store %cst, %x[%i] : memref<?xf64>
    }
  }
  return
}

// CHECK:  func.func @max_constant(%arg0: i32, %arg1: memref<?xf64>) {
// CHECK-NEXT:   %cst = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:   %0 = arith.index_cast %arg0 : i32 to index
// CHECK-NEXT:   affine.if #set1()[%0] {
// CHECK-NEXT:     affine.store %cst, %arg1[0] : memref<?xf64>
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }

// A check of another value decides nothing.
func.func @unrelated(%n32: i32, %k: index, %x: memref<?xf64>) {
  %c1 = arith.constant 1 : i32
  %cst = arith.constant 0.0 : f64
  %n = arith.index_cast %n32 : i32 to index
  %m32 = arith.maxsi %n32, %c1 : i32
  %m = arith.index_cast %m32 : i32 to index
  affine.if affine_set<()[s0] : (s0 - 1 >= 0)>()[%k] {
    affine.for %i = 0 to %m {
      affine.store %cst, %x[%i + symbol(%n)] : memref<?xf64>
    }
  }
  return
}

// CHECK:  func.func @unrelated(%arg0: i32, %arg1: index, %arg2: memref<?xf64>) {
// CHECK-NEXT:   %c1_i32 = arith.constant 1 : i32
// CHECK-NEXT:   %cst = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:   %0 = arith.index_cast %arg0 : i32 to index
// CHECK-NEXT:   %1 = arith.maxsi %arg0, %c1_i32 : i32
// CHECK-NEXT:   %2 = arith.index_cast %1 : i32 to index
// CHECK-NEXT:   affine.if #set2()[%arg1] {
// CHECK-NEXT:     affine.parallel (%arg3) = (0) to (symbol(%2)) {
// CHECK-NEXT:       affine.store %cst, %arg2[%arg3 + symbol(%0)] : memref<?xf64>
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }
