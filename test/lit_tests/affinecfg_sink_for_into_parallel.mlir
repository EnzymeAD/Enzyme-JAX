// RUN: enzymexlamlir-opt %s --affine-cfg | FileCheck %s

// MFEM's transposed interpolators: a serial loop over i whose body reads a
// value of i and then lets each lane j accumulate into its own element.
// The lanes never touch each other's elements in any iteration, so the
// serial loop moves inside: each lane carries its own accumulation.
func.func @transpose(%x: memref<?xf64>, %G: memref<?xf64>, %w: memref<25xf64>, %n: index) {
  affine.for %i = 0 to %n {
    %a = affine.load %x[%i] : memref<?xf64>
    affine.parallel (%j) = (0) to (5) {
      %g = affine.load %G[%i + %j * symbol(%n)] : memref<?xf64>
      %p = arith.mulf %a, %g : f64
      %acc = affine.load %w[%j * 5] : memref<25xf64>
      %s = arith.addf %acc, %p : f64
      affine.store %s, %w[%j * 5] : memref<25xf64>
    }
  }
  return
}

// CHECK:  func.func @transpose(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: memref<25xf64>, %arg3: index) {
// CHECK-NEXT:   affine.parallel (%arg4) = (0) to (5) {
// CHECK-NEXT:     affine.for %arg5 = 0 to %arg3 {
// CHECK-NEXT:       %0 = affine.load %arg0[%arg5] : memref<?xf64>
// CHECK-NEXT:       %1 = affine.load %arg1[%arg5 + %arg4 * symbol(%arg3)] : memref<?xf64>
// CHECK-NEXT:       %2 = arith.mulf %0, %1 : f64
// CHECK-NEXT:       %3 = affine.load %arg2[%arg4 * 5] : memref<25xf64>
// CHECK-NEXT:       %4 = arith.addf %3, %2 : f64
// CHECK-NEXT:       affine.store %4, %arg2[%arg4 * 5] : memref<25xf64>
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }

// Under outer loops a and b, in one iteration of which the lanes stay
// apart (lane j writes a + 25 j + 5 b): the loops a and b keep their
// place, as another iteration of b can reach a lane's element.
func.func @nested(%x: memref<?xf64>, %G: memref<?xf64>, %n: index, %d: index, %ne: index) {
  affine.parallel (%e) = (0) to (symbol(%ne)) {
    %w = memref.alloca() : memref<375xf64>
    affine.for %a = 0 to %d {
      affine.for %b = 0 to %d {
        affine.for %i = 0 to %n {
          %v = affine.load %x[%i + %e * symbol(%n)] : memref<?xf64>
          affine.parallel (%j) = (0) to (symbol(%d)) {
            %g = affine.load %G[%i + %j * symbol(%n)] : memref<?xf64>
            %p = arith.mulf %v, %g : f64
            %acc = affine.load %w[%a + %j * 25 + %b * 5] : memref<375xf64>
            %s = arith.addf %acc, %p : f64
            affine.store %s, %w[%a + %j * 25 + %b * 5] : memref<375xf64>
          }
        }
      }
    }
  }
  return
}

// CHECK:  func.func @nested(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: index, %arg3: index, %arg4: index) {
// CHECK-NEXT:   affine.parallel (%arg5) = (0) to (symbol(%arg4)) {
// CHECK-NEXT:     %alloca = memref.alloca() : memref<375xf64>
// CHECK-NEXT:     affine.for %arg6 = 0 to %arg3 {
// CHECK-NEXT:       affine.for %arg7 = 0 to %arg3 {
// CHECK-NEXT:         affine.parallel (%arg8) = (0) to (symbol(%arg3)) {
// CHECK-NEXT:           affine.for %arg9 = 0 to %arg2 {
// CHECK-NEXT:             %0 = affine.load %arg0[%arg9 + %arg5 * symbol(%arg2)] : memref<?xf64>
// CHECK-NEXT:             %1 = affine.load %arg1[%arg9 + %arg8 * symbol(%arg2)] : memref<?xf64>
// CHECK-NEXT:             %2 = arith.mulf %0, %1 : f64
// CHECK-NEXT:             %3 = affine.load %alloca[%arg6 + %arg8 * 25 + %arg7 * 5] : memref<375xf64>
// CHECK-NEXT:             %4 = arith.addf %3, %2 : f64
// CHECK-NEXT:             affine.store %4, %alloca[%arg6 + %arg8 * 25 + %arg7 * 5] : memref<375xf64>
// CHECK-NEXT:           }
// CHECK-NEXT:         }
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }

// Lane j writes w[i + j]: lane j + 1 writes the same element one iteration
// later. The loop stays.
func.func @overlap(%x: memref<?xf64>, %G: memref<?xf64>, %w: memref<25xf64>, %n: index) {
  affine.for %i = 0 to %n {
    %a = affine.load %x[%i] : memref<?xf64>
    affine.parallel (%j) = (0) to (5) {
      %g = affine.load %G[%i + %j * symbol(%n)] : memref<?xf64>
      %p = arith.mulf %a, %g : f64
      %acc = affine.load %w[%i + %j] : memref<25xf64>
      %s = arith.addf %acc, %p : f64
      affine.store %s, %w[%i + %j] : memref<25xf64>
    }
  }
  return
}

// CHECK:  func.func @overlap(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: memref<25xf64>, %arg3: index) {
// CHECK-NEXT:   affine.for %arg4 = 0 to %arg3 {
// CHECK-NEXT:     %0 = affine.load %arg0[%arg4] : memref<?xf64>
// CHECK-NEXT:     affine.parallel (%arg5) = (0) to (5) {
// CHECK-NEXT:       %1 = affine.load %arg1[%arg4 + %arg5 * symbol(%arg3)] : memref<?xf64>
// CHECK-NEXT:       %2 = arith.mulf %0, %1 : f64
// CHECK-NEXT:       %3 = affine.load %arg2[%arg4 + %arg5] : memref<25xf64>
// CHECK-NEXT:       %4 = arith.addf %3, %2 : f64
// CHECK-NEXT:       affine.store %4, %arg2[%arg4 + %arg5] : memref<25xf64>
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }

// The value read before the lanes is one the lanes write: a lane would
// read it after another lane's later iteration wrote it. The loop stays.
func.func @prefix_reads(%x: memref<?xf64>, %G: memref<?xf64>, %w: memref<25xf64>, %n: index) {
  affine.for %i = 0 to %n {
    %a = affine.load %w[%i] : memref<25xf64>
    affine.parallel (%j) = (0) to (5) {
      %g = affine.load %G[%i + %j * symbol(%n)] : memref<?xf64>
      %p = arith.mulf %a, %g : f64
      %acc = affine.load %w[%j * 5] : memref<25xf64>
      %s = arith.addf %acc, %p : f64
      affine.store %s, %w[%j * 5] : memref<25xf64>
    }
  }
  return
}

// CHECK:  func.func @prefix_reads(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: memref<25xf64>, %arg3: index) {
// CHECK-NEXT:   affine.for %arg4 = 0 to %arg3 {
// CHECK-NEXT:     %0 = affine.load %arg2[%arg4] : memref<25xf64>
// CHECK-NEXT:     affine.parallel (%arg5) = (0) to (5) {
// CHECK-NEXT:       %1 = affine.load %arg1[%arg4 + %arg5 * symbol(%arg3)] : memref<?xf64>
// CHECK-NEXT:       %2 = arith.mulf %0, %1 : f64
// CHECK-NEXT:       %3 = affine.load %arg2[%arg5 * 5] : memref<25xf64>
// CHECK-NEXT:       %4 = arith.addf %3, %2 : f64
// CHECK-NEXT:       affine.store %4, %arg2[%arg5 * 5] : memref<25xf64>
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }

// The lanes' extent changes with i: the loop stays.
func.func @varying(%x: memref<?xf64>, %G: memref<?xf64>, %w: memref<25xf64>, %n: index) {
  affine.for %i = 0 to %n {
    %a = affine.load %x[%i] : memref<?xf64>
    affine.parallel (%j) = (0) to (%i + 1) {
      %g = affine.load %G[%i + %j * symbol(%n)] : memref<?xf64>
      %p = arith.mulf %a, %g : f64
      %acc = affine.load %w[%j * 5] : memref<25xf64>
      %s = arith.addf %acc, %p : f64
      affine.store %s, %w[%j * 5] : memref<25xf64>
    }
  }
  return
}

// CHECK:  func.func @varying(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: memref<25xf64>, %arg3: index) {
// CHECK-NEXT:   affine.for %arg4 = 0 to %arg3 {
// CHECK-NEXT:     %0 = affine.load %arg0[%arg4] : memref<?xf64>
// CHECK-NEXT:     affine.parallel (%arg5) = (0) to (%arg4 + 1) {
// CHECK-NEXT:       %1 = affine.load %arg1[%arg4 + %arg5 * symbol(%arg3)] : memref<?xf64>
// CHECK-NEXT:       %2 = arith.mulf %0, %1 : f64
// CHECK-NEXT:       %3 = affine.load %arg2[%arg5 * 5] : memref<25xf64>
// CHECK-NEXT:       %4 = arith.addf %3, %2 : f64
// CHECK-NEXT:       affine.store %4, %arg2[%arg5 * 5] : memref<25xf64>
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }

// A loop that is parallel on its own is parallelized, not sunk.
func.func @parallel_itself(%x: memref<?xf64>, %w: memref<?xf64>, %n: index) {
  affine.for %i = 0 to %n {
    %a = affine.load %x[%i] : memref<?xf64>
    affine.parallel (%j) = (0) to (5) {
      affine.store %a, %w[%i * 5 + %j] : memref<?xf64>
    }
  }
  return
}

// CHECK:  func.func @parallel_itself(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: index) {
// CHECK-NEXT:   affine.parallel (%arg3) = (0) to (symbol(%arg2)) {
// CHECK-NEXT:     %0 = affine.load %arg0[%arg3] : memref<?xf64>
// CHECK-NEXT:     affine.parallel (%arg4) = (0) to (5) {
// CHECK-NEXT:       affine.store %0, %arg1[%arg4 + %arg3 * 5] : memref<?xf64>
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }
