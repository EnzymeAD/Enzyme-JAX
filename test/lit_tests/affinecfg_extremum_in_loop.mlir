// RUN: enzymexlamlir-opt %s --affine-cfg | FileCheck %s

// Inside a loop over 0 .. n, which runs only when n >= 1, max(n, 1) is n; it
// stays max(n, 1) outside.
func.func @in_loop(%n32: i32, %x: memref<?xf64>, %y: memref<?xindex>) {
  %c1 = arith.constant 1 : i32
  %cst = arith.constant 0.0 : f64
  %n = arith.index_cast %n32 : i32 to index
  %m32 = arith.maxsi %n32, %c1 : i32
  %m = arith.index_cast %m32 : i32 to index
  affine.for %i = 0 to %n {
    affine.store %cst, %x[%i + symbol(%m)] : memref<?xf64>
  }
  affine.store %m, %y[0] : memref<?xindex>
  return
}

// CHECK:  func.func @in_loop(%arg0: i32, %arg1: memref<?xf64>, %arg2: memref<?xindex>) {
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

// A loop over 2 .. n runs only when n >= 3, where min(n, 2) is 2.
func.func @lower_bound(%n32: i32, %x: memref<?xf64>) {
  %c2 = arith.constant 2 : i32
  %cst = arith.constant 0.0 : f64
  %n = arith.index_cast %n32 : i32 to index
  %m32 = arith.minsi %n32, %c2 : i32
  %m = arith.index_cast %m32 : i32 to index
  affine.for %i = 2 to %n {
    affine.store %cst, %x[%i + symbol(%m)] : memref<?xf64>
  }
  return
}

// CHECK:  func.func @lower_bound(%arg0: i32, %arg1: memref<?xf64>) {
// CHECK-NEXT:   %cst = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:   %0 = arith.index_cast %arg0 : i32 to index
// CHECK-NEXT:   affine.parallel (%arg2) = (2) to (symbol(%0)) {
// CHECK-NEXT:     affine.store %cst, %arg1[%arg2 + 2] : memref<?xf64>
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }

// A loop over another count decides nothing about n.
func.func @other_count(%n32: i32, %k: index, %x: memref<?xf64>) {
  %c1 = arith.constant 1 : i32
  %cst = arith.constant 0.0 : f64
  %n = arith.index_cast %n32 : i32 to index
  %m32 = arith.maxsi %n32, %c1 : i32
  %m = arith.index_cast %m32 : i32 to index
  affine.for %i = 0 to %k {
    affine.store %cst, %x[%i + symbol(%m) + symbol(%n)] : memref<?xf64>
  }
  return
}

// CHECK:  func.func @other_count(%arg0: i32, %arg1: index, %arg2: memref<?xf64>) {
// CHECK-NEXT:   %c1_i32 = arith.constant 1 : i32
// CHECK-NEXT:   %cst = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:   %0 = arith.index_cast %arg0 : i32 to index
// CHECK-NEXT:   %1 = arith.maxsi %arg0, %c1_i32 : i32
// CHECK-NEXT:   %2 = arith.index_cast %1 : i32 to index
// CHECK-NEXT:   affine.parallel (%arg3) = (0) to (symbol(%arg1)) {
// CHECK-NEXT:     affine.store %cst, %arg2[%arg3 + symbol(%2) + symbol(%0)] : memref<?xf64>
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }
