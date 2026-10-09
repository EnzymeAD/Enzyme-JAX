// RUN: enzymexlamlir-opt %s --affine-cfg | FileCheck %s

// A scratch of static shape every iteration writes whole before it reads any
// of it (MFEM's per-point matrix, filled and inverted at each quadrature
// point) carries nothing from one iteration to the next: the loop is
// parallel, and the scratch moves into its body, each lane's own.
func.func @private(%J: memref<?xf64>, %out: memref<?xf64>, %n: index) {
  %m = memref.alloca() : memref<4xf64>
  affine.for %q = 0 to %n {
    %a = affine.load %J[%q * 4] : memref<?xf64>
    %b = affine.load %J[%q * 4 + 1] : memref<?xf64>
    %c = affine.load %J[%q * 4 + 2] : memref<?xf64>
    %d = affine.load %J[%q * 4 + 3] : memref<?xf64>
    affine.store %d, %m[0] : memref<4xf64>
    affine.store %b, %m[1] : memref<4xf64>
    affine.store %c, %m[2] : memref<4xf64>
    affine.store %a, %m[3] : memref<4xf64>
    affine.for %i = 0 to 2 {
      %x = affine.load %m[%i * 2] : memref<4xf64>
      %y = affine.load %m[%i * 2 + 1] : memref<4xf64>
      %s = arith.addf %x, %y : f64
      affine.store %s, %out[%q * 2 + %i] : memref<?xf64>
    }
  }
  return
}

// A load before the stores reads the previous iteration's value: the loop
// stays.
func.func @reads_previous(%J: memref<?xf64>, %out: memref<?xf64>, %n: index) {
  %m = memref.alloca() : memref<4xf64>
  affine.for %q = 0 to %n {
    %a = affine.load %J[%q * 4] : memref<?xf64>
    %b = affine.load %J[%q * 4 + 1] : memref<?xf64>
    %c = affine.load %J[%q * 4 + 2] : memref<?xf64>
    %d = affine.load %J[%q * 4 + 3] : memref<?xf64>
    %prev = affine.load %m[3] : memref<4xf64>
    %d2 = arith.addf %d, %prev : f64
    affine.store %d2, %m[0] : memref<4xf64>
    affine.store %b, %m[1] : memref<4xf64>
    affine.store %c, %m[2] : memref<4xf64>
    affine.store %a, %m[3] : memref<4xf64>
    affine.for %i = 0 to 2 {
      %x = affine.load %m[%i * 2] : memref<4xf64>
      %y = affine.load %m[%i * 2 + 1] : memref<4xf64>
      %s = arith.addf %x, %y : f64
      affine.store %s, %out[%q * 2 + %i] : memref<?xf64>
    }
  }
  return
}

// The stores miss an element the loop reads: the loop stays.
func.func @partly_written(%J: memref<?xf64>, %out: memref<?xf64>, %n: index) {
  %m = memref.alloca() : memref<4xf64>
  affine.for %q = 0 to %n {
    %a = affine.load %J[%q * 4] : memref<?xf64>
    %b = affine.load %J[%q * 4 + 1] : memref<?xf64>
    %c = affine.load %J[%q * 4 + 2] : memref<?xf64>
    %d = affine.load %J[%q * 4 + 3] : memref<?xf64>
    affine.store %d, %m[0] : memref<4xf64>
    affine.store %b, %m[1] : memref<4xf64>
    affine.store %c, %m[2] : memref<4xf64>
    affine.for %i = 0 to 2 {
      %x = affine.load %m[%i * 2] : memref<4xf64>
      %y = affine.load %m[%i * 2 + 1] : memref<4xf64>
      %s = arith.addf %x, %y : f64
      affine.store %s, %out[%q * 2 + %i] : memref<?xf64>
    }
  }
  return
}

// The scratch is read after the loop: the loop stays.
func.func @read_after(%J: memref<?xf64>, %out: memref<?xf64>, %n: index) {
  %m = memref.alloca() : memref<4xf64>
  affine.for %q = 0 to %n {
    %a = affine.load %J[%q * 4] : memref<?xf64>
    %b = affine.load %J[%q * 4 + 1] : memref<?xf64>
    %c = affine.load %J[%q * 4 + 2] : memref<?xf64>
    %d = affine.load %J[%q * 4 + 3] : memref<?xf64>
    affine.store %d, %m[0] : memref<4xf64>
    affine.store %b, %m[1] : memref<4xf64>
    affine.store %c, %m[2] : memref<4xf64>
    affine.store %a, %m[3] : memref<4xf64>
    affine.for %i = 0 to 2 {
      %x = affine.load %m[%i * 2] : memref<4xf64>
      %y = affine.load %m[%i * 2 + 1] : memref<4xf64>
      %s = arith.addf %x, %y : f64
      affine.store %s, %out[%q * 2 + %i] : memref<?xf64>
    }
  }
  %last = affine.load %m[0] : memref<4xf64>
  affine.store %last, %out[0] : memref<?xf64>
  return
}

// The stores fill the scratch in one arm of a branch, and a load sits in the
// other: an iteration taking that arm reads what an earlier iteration
// stored. The loop stays serial.
func.func @other_arm(%J: memref<?xf64>, %out: memref<?xf64>, %n: index, %c: i1) {
  %m = memref.alloca() : memref<2xf64>
  affine.for %q = 0 to %n {
    %a = affine.load %J[%q * 2] : memref<?xf64>
    %b = affine.load %J[%q * 2 + 1] : memref<?xf64>
    scf.if %c {
      %x = affine.load %m[0] : memref<2xf64>
      affine.store %x, %out[%q] : memref<?xf64>
    } else {
      affine.store %a, %m[0] : memref<2xf64>
      affine.store %b, %m[1] : memref<2xf64>
    }
  }
  return
}

// Every user of the scratch is in one arm of a branch: the stores are not
// guaranteed to run in an iteration. The loop stays serial.
func.func @under_branch(%J: memref<?xf64>, %out: memref<?xf64>, %n: index, %c: i1) {
  %m = memref.alloca() : memref<2xf64>
  affine.for %q = 0 to %n {
    %a = affine.load %J[%q * 2] : memref<?xf64>
    %b = affine.load %J[%q * 2 + 1] : memref<?xf64>
    scf.if %c {
      affine.store %a, %m[0] : memref<2xf64>
      affine.store %b, %m[1] : memref<2xf64>
      %x = affine.load %m[0] : memref<2xf64>
      %y = affine.load %m[1] : memref<2xf64>
      %s = arith.addf %x, %y : f64
      affine.store %s, %out[%q] : memref<?xf64>
    }
  }
  return
}

// CHECK:  func.func @private(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: index) {
// CHECK-NEXT:   affine.parallel (%arg3) = (0) to (symbol(%arg2)) {
// CHECK-NEXT:     %alloca = memref.alloca() : memref<4xf64>
// CHECK-NEXT:     %0 = affine.load %arg0[%arg3 * 4] : memref<?xf64>
// CHECK-NEXT:     %1 = affine.load %arg0[%arg3 * 4 + 1] : memref<?xf64>
// CHECK-NEXT:     %2 = affine.load %arg0[%arg3 * 4 + 2] : memref<?xf64>
// CHECK-NEXT:     %3 = affine.load %arg0[%arg3 * 4 + 3] : memref<?xf64>
// CHECK-NEXT:     affine.store %3, %alloca[0] : memref<4xf64>
// CHECK-NEXT:     affine.store %1, %alloca[1] : memref<4xf64>
// CHECK-NEXT:     affine.store %2, %alloca[2] : memref<4xf64>
// CHECK-NEXT:     affine.store %0, %alloca[3] : memref<4xf64>
// CHECK-NEXT:     affine.parallel (%arg4) = (0) to (2) {
// CHECK-NEXT:       %4 = affine.load %alloca[%arg4 * 2] : memref<4xf64>
// CHECK-NEXT:       %5 = affine.load %alloca[%arg4 * 2 + 1] : memref<4xf64>
// CHECK-NEXT:       %6 = arith.addf %4, %5 : f64
// CHECK-NEXT:       affine.store %6, %arg1[%arg4 + %arg3 * 2] : memref<?xf64>
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }

// CHECK:  func.func @reads_previous(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: index) {
// CHECK-NEXT:   %alloca = memref.alloca() : memref<4xf64>
// CHECK-NEXT:   affine.for %arg3 = 0 to %arg2 {
// CHECK-NEXT:     %0 = affine.load %arg0[%arg3 * 4] : memref<?xf64>
// CHECK-NEXT:     %1 = affine.load %arg0[%arg3 * 4 + 1] : memref<?xf64>
// CHECK-NEXT:     %2 = affine.load %arg0[%arg3 * 4 + 2] : memref<?xf64>
// CHECK-NEXT:     %3 = affine.load %arg0[%arg3 * 4 + 3] : memref<?xf64>
// CHECK-NEXT:     %4 = affine.load %alloca[3] : memref<4xf64>
// CHECK-NEXT:     %5 = arith.addf %3, %4 : f64
// CHECK-NEXT:     affine.store %5, %alloca[0] : memref<4xf64>
// CHECK-NEXT:     affine.store %1, %alloca[1] : memref<4xf64>
// CHECK-NEXT:     affine.store %2, %alloca[2] : memref<4xf64>
// CHECK-NEXT:     affine.store %0, %alloca[3] : memref<4xf64>
// CHECK-NEXT:     affine.parallel (%arg4) = (0) to (2) {
// CHECK-NEXT:       %6 = affine.load %alloca[%arg4 * 2] : memref<4xf64>
// CHECK-NEXT:       %7 = affine.load %alloca[%arg4 * 2 + 1] : memref<4xf64>
// CHECK-NEXT:       %8 = arith.addf %6, %7 : f64
// CHECK-NEXT:       affine.store %8, %arg1[%arg4 + %arg3 * 2] : memref<?xf64>
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }

// CHECK:  func.func @partly_written(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: index) {
// CHECK-NEXT:   %alloca = memref.alloca() : memref<4xf64>
// CHECK-NEXT:   affine.for %arg3 = 0 to %arg2 {
// CHECK-NEXT:     %0 = affine.load %arg0[%arg3 * 4 + 1] : memref<?xf64>
// CHECK-NEXT:     %1 = affine.load %arg0[%arg3 * 4 + 2] : memref<?xf64>
// CHECK-NEXT:     %2 = affine.load %arg0[%arg3 * 4 + 3] : memref<?xf64>
// CHECK-NEXT:     affine.store %2, %alloca[0] : memref<4xf64>
// CHECK-NEXT:     affine.store %0, %alloca[1] : memref<4xf64>
// CHECK-NEXT:     affine.store %1, %alloca[2] : memref<4xf64>
// CHECK-NEXT:     affine.parallel (%arg4) = (0) to (2) {
// CHECK-NEXT:       %3 = affine.load %alloca[%arg4 * 2] : memref<4xf64>
// CHECK-NEXT:       %4 = affine.load %alloca[%arg4 * 2 + 1] : memref<4xf64>
// CHECK-NEXT:       %5 = arith.addf %3, %4 : f64
// CHECK-NEXT:       affine.store %5, %arg1[%arg4 + %arg3 * 2] : memref<?xf64>
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }

// CHECK:  func.func @read_after(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: index) {
// CHECK-NEXT:   %alloca = memref.alloca() : memref<4xf64>
// CHECK-NEXT:   affine.for %arg3 = 0 to %arg2 {
// CHECK-NEXT:     %1 = affine.load %arg0[%arg3 * 4] : memref<?xf64>
// CHECK-NEXT:     %2 = affine.load %arg0[%arg3 * 4 + 1] : memref<?xf64>
// CHECK-NEXT:     %3 = affine.load %arg0[%arg3 * 4 + 2] : memref<?xf64>
// CHECK-NEXT:     %4 = affine.load %arg0[%arg3 * 4 + 3] : memref<?xf64>
// CHECK-NEXT:     affine.store %4, %alloca[0] : memref<4xf64>
// CHECK-NEXT:     affine.store %2, %alloca[1] : memref<4xf64>
// CHECK-NEXT:     affine.store %3, %alloca[2] : memref<4xf64>
// CHECK-NEXT:     affine.store %1, %alloca[3] : memref<4xf64>
// CHECK-NEXT:     affine.parallel (%arg4) = (0) to (2) {
// CHECK-NEXT:       %5 = affine.load %alloca[%arg4 * 2] : memref<4xf64>
// CHECK-NEXT:       %6 = affine.load %alloca[%arg4 * 2 + 1] : memref<4xf64>
// CHECK-NEXT:       %7 = arith.addf %5, %6 : f64
// CHECK-NEXT:       affine.store %7, %arg1[%arg4 + %arg3 * 2] : memref<?xf64>
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   %0 = affine.load %alloca[0] : memref<4xf64>
// CHECK-NEXT:   affine.store %0, %arg1[0] : memref<?xf64>
// CHECK-NEXT:   return
// CHECK-NEXT: }

// CHECK:  func.func @other_arm(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: index, %arg3: i1) {
// CHECK-NEXT:   %alloca = memref.alloca() : memref<2xf64>
// CHECK-NEXT:   affine.for %arg4 = 0 to %arg2 {
// CHECK-NEXT:     %0 = affine.load %arg0[%arg4 * 2] : memref<?xf64>
// CHECK-NEXT:     %1 = affine.load %arg0[%arg4 * 2 + 1] : memref<?xf64>
// CHECK-NEXT:     scf.if %arg3 {
// CHECK-NEXT:       %2 = affine.load %alloca[0] : memref<2xf64>
// CHECK-NEXT:       affine.store %2, %arg1[%arg4] : memref<?xf64>
// CHECK-NEXT:     } else {
// CHECK-NEXT:       affine.store %0, %alloca[0] : memref<2xf64>
// CHECK-NEXT:       affine.store %1, %alloca[1] : memref<2xf64>
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }

// CHECK:  func.func @under_branch(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: index, %arg3: i1) {
// CHECK-NEXT:   %alloca = memref.alloca() : memref<2xf64>
// CHECK-NEXT:   affine.for %arg4 = 0 to %arg2 {
// CHECK-NEXT:     %0 = affine.load %arg0[%arg4 * 2] : memref<?xf64>
// CHECK-NEXT:     %1 = affine.load %arg0[%arg4 * 2 + 1] : memref<?xf64>
// CHECK-NEXT:     scf.if %arg3 {
// CHECK-NEXT:       affine.store %0, %alloca[0] : memref<2xf64>
// CHECK-NEXT:       affine.store %1, %alloca[1] : memref<2xf64>
// CHECK-NEXT:       %2 = affine.load %alloca[0] : memref<2xf64>
// CHECK-NEXT:       %3 = affine.load %alloca[1] : memref<2xf64>
// CHECK-NEXT:       %4 = arith.addf %2, %3 : f64
// CHECK-NEXT:       affine.store %4, %arg1[%arg4] : memref<?xf64>
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }
