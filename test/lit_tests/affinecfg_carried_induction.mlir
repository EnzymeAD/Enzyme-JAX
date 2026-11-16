// RUN: enzymexlamlir-opt %s --affine-cfg | FileCheck %s

// A counter the loop advances by a constant: in each iteration it is its
// initial value plus that many steps, after the loop plus as many as the loop
// ran. With it gone the loop carries nothing and is parallel.
func.func @counter(%n: index, %x: memref<?xi32>, %out: memref<1xi32>) {
  %c0 = arith.constant 0 : i32
  %c3 = arith.constant 3 : i32
  %r = affine.for %i = 0 to %n iter_args(%o = %c0) -> (i32) {
    %idx = arith.index_cast %i : index to i32
    %v = arith.addi %idx, %o : i32
    affine.store %v, %x[%i] : memref<?xi32>
    %next = arith.addi %o, %c3 : i32
    affine.yield %next : i32
  }
  affine.store %r, %out[0] : memref<1xi32>
  return
}

// CHECK:  func.func @counter(%arg0: index, %arg1: memref<?xi32>, %arg2: memref<1xi32>) {
// CHECK-NEXT:   %c0 = arith.constant 0 : index
// CHECK-NEXT:   %c3_i32 = arith.constant 3 : i32
// CHECK-NEXT:   affine.parallel (%arg3) = (0) to (symbol(%arg0)) {
// CHECK-NEXT:     %3 = arith.index_cast %arg3 : index to i32
// CHECK-NEXT:     %4 = arith.muli %3, %c3_i32 : i32
// CHECK-NEXT:     %5 = arith.index_cast %arg3 : index to i32
// CHECK-NEXT:     %6 = arith.addi %5, %4 : i32
// CHECK-NEXT:     affine.store %6, %arg1[%arg3] : memref<?xi32>
// CHECK-NEXT:   }
// CHECK-NEXT:   %0 = arith.maxsi %arg0, %c0 : index
// CHECK-NEXT:   %1 = arith.index_cast %0 : index to i32
// CHECK-NEXT:   %2 = arith.muli %1, %c3_i32 : i32
// CHECK-NEXT:   affine.store %2, %arg2[0] : memref<1xi32>
// CHECK-NEXT:   return
// CHECK-NEXT: }

// A step defined outside the loop, from a lower bound other than zero by
// twos: the k-th iteration is (i - 4) floordiv 2.
func.func @invariant_step(%n: index, %s: i64, %x: memref<?xi64>) {
  %c7 = arith.constant 7 : i64
  %r = affine.for %i = 4 to %n step 2 iter_args(%o = %c7) -> (i64) {
    affine.store %o, %x[%i] : memref<?xi64>
    %next = arith.addi %o, %s : i64
    affine.yield %next : i64
  }
  return
}

// CHECK:  func.func @invariant_step(%arg0: index, %arg1: i64, %arg2: memref<?xi64>) {
// CHECK-NEXT:   %c7_i64 = arith.constant 7 : i64
// CHECK-NEXT:   affine.parallel (%arg3) = (4) to (symbol(%arg0)) step (2) {
// CHECK-NEXT:     %0 = affine.apply #map(%arg3)
// CHECK-NEXT:     %1 = arith.index_cast %0 : index to i64
// CHECK-NEXT:     %2 = arith.muli %1, %arg1 : i64
// CHECK-NEXT:     %3 = arith.addi %2, %c7_i64 : i64
// CHECK-NEXT:     affine.store %3, %arg2[%arg3] : memref<?xi64>
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }

// A sum of values that change from one iteration to the next is no
// induction: left carried.
func.func @varying(%n: index, %x: memref<?xi32>, %out: memref<1xi32>) {
  %c0 = arith.constant 0 : i32
  %r = affine.for %i = 0 to %n iter_args(%o = %c0) -> (i32) {
    %v = affine.load %x[%i] : memref<?xi32>
    affine.store %o, %x[%i] : memref<?xi32>
    %next = arith.addi %o, %v : i32
    affine.yield %next : i32
  }
  affine.store %r, %out[0] : memref<1xi32>
  return
}

// CHECK:  func.func @varying(%arg0: index, %arg1: memref<?xi32>, %arg2: memref<1xi32>) {
// CHECK-NEXT:   %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:   %0 = affine.for %arg3 = 0 to %arg0 iter_args(%arg4 = %c0_i32) -> (i32) {
// CHECK-NEXT:     %1 = affine.load %arg1[%arg3] : memref<?xi32>
// CHECK-NEXT:     affine.store %arg4, %arg1[%arg3] : memref<?xi32>
// CHECK-NEXT:     %2 = arith.addi %arg4, %1 : i32
// CHECK-NEXT:     affine.yield %2 : i32
// CHECK-NEXT:   }
// CHECK-NEXT:   affine.store %0, %arg2[0] : memref<1xi32>
// CHECK-NEXT:   return
// CHECK-NEXT: }

// A counter a kernel inside the loop reads as an index symbol: it is the
// loop's induction variable, and the kernel's index reads that.
func.func @counter_as_symbol(%n: index, %ny: index, %buf: memref<?xi64>, %val: i64) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c32 = arith.constant 32 : index
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %r = affine.for %i = 0 to 12 iter_args(%acc = %c0_i32) -> (i32) {
    %sym = arith.index_cast %acc : i32 to index
    %w = "enzymexla.gpu_wrapper"(%n, %ny, %c1, %c32, %c1, %c1) ({
      %flat = arith.muli %n, %c32 overflow<nsw> : index
      scf.parallel (%a5, %a6) = (%c0, %c0) to (%ny, %flat) step (%c1, %c1) {
        %idx = arith.addi %sym, %a6 : index
        memref.store %val, %buf[%idx] : memref<?xi64>
        scf.reduce
      }
      "enzymexla.polygeist_yield"() : () -> ()
    }) : (index, index, index, index, index, index) -> index
    %next = arith.addi %acc, %c1_i32 : i32
    affine.yield %next : i32
  }
  return
}

// CHECK:  func.func @counter_as_symbol(%arg0: index, %arg1: index, %arg2: memref<?xi64>, %arg3: i64) {
// CHECK-NEXT:   %c1 = arith.constant 1 : index
// CHECK-NEXT:   %c32 = arith.constant 32 : index
// CHECK-NEXT:   affine.for %arg4 = 0 to 12 {
// CHECK-NEXT:     %0 = "enzymexla.gpu_wrapper"(%arg0, %arg1, %c1, %c32, %c1, %c1) ({
// CHECK-NEXT:       affine.parallel (%arg5, %arg6) = (0, 0) to (symbol(%arg1), symbol(%arg0) * 32) {
// CHECK-NEXT:         affine.store %arg3, %arg2[%arg6 + %arg4] : memref<?xi64>
// CHECK-NEXT:       }
// CHECK-NEXT:       "enzymexla.polygeist_yield"() : () -> ()
// CHECK-NEXT:     }) : (index, index, index, index, index, index) -> index
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }
