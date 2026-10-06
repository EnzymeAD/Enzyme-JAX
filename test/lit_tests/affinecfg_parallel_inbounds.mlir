// RUN: enzymexlamlir-opt %s --affine-cfg | FileCheck %s

// A per-thread scratch laid out for at most 5 points a side, filled by loops
// over a runtime count: the stores stay in bounds at the nest's corners only
// if the count is at most 5 (the one at +250 gives 31 * (n - 1) + 250 <= 374),
// and then no two iterations write one element.
func.func @flattened(%n: index, %x: memref<?xf64>, %y: memref<?xf64>) {
  %a = memref.alloca() : memref<375xf64>
  affine.for %i = 0 to %n {
    affine.for %j = 0 to %n {
      affine.for %k = 0 to %n {
        %v = affine.load %x[%i] : memref<?xf64>
        affine.store %v, %a[%i + %k * 5 + %j * 25] : memref<375xf64>
        affine.store %v, %a[%i + %k * 5 + %j * 25 + 250] : memref<375xf64>
      }
    }
  }
  %r = affine.load %a[7] : memref<375xf64>
  affine.store %r, %y[0] : memref<?xf64>
  return
}

// CHECK:  func.func @flattened(%arg0: index, %arg1: memref<?xf64>, %arg2: memref<?xf64>) {
// CHECK-NEXT:   %alloca = memref.alloca() : memref<375xf64>
// CHECK-NEXT:   affine.parallel (%arg3, %arg4, %arg5) = (0, 0, 0) to (symbol(%arg0), symbol(%arg0), symbol(%arg0)) {
// CHECK-NEXT:     %1 = affine.load %arg1[%arg3] : memref<?xf64>
// CHECK-NEXT:     affine.store %1, %alloca[%arg3 + %arg5 * 5 + %arg4 * 25] : memref<375xf64>
// CHECK-NEXT:     affine.store %1, %alloca[%arg3 + %arg5 * 5 + %arg4 * 25 + 250] : memref<375xf64>
// CHECK-NEXT:   }
// CHECK-NEXT:   %0 = affine.load %alloca[7] : memref<375xf64>
// CHECK-NEXT:   affine.store %0, %arg2[0] : memref<?xf64>
// CHECK-NEXT:   return
// CHECK-NEXT: }

// The same under a check of the count: a condition on no loop variable holds
// at every point of the nest or at none.
func.func @under_check(%n: index, %x: memref<?xf64>, %y: memref<?xf64>) {
  %a = memref.alloca() : memref<375xf64>
  affine.if affine_set<()[s0] : (s0 - 1 >= 0)>()[%n] {
    affine.for %i = 0 to %n {
      affine.for %j = 0 to %n {
        affine.for %k = 0 to %n {
          %v = affine.load %x[%i] : memref<?xf64>
          affine.store %v, %a[%i + %k * 5 + %j * 25] : memref<375xf64>
          affine.store %v, %a[%i + %k * 5 + %j * 25 + 250] : memref<375xf64>
        }
      }
    }
  }
  %r = affine.load %a[7] : memref<375xf64>
  affine.store %r, %y[0] : memref<?xf64>
  return
}

// CHECK:  func.func @under_check(%arg0: index, %arg1: memref<?xf64>, %arg2: memref<?xf64>) {
// CHECK-NEXT:   %alloca = memref.alloca() : memref<375xf64>
// CHECK-NEXT:   affine.if #set()[%arg0] {
// CHECK-NEXT:     affine.parallel (%arg3, %arg4, %arg5) = (0, 0, 0) to (symbol(%arg0), symbol(%arg0), symbol(%arg0)) {
// CHECK-NEXT:       %1 = affine.load %arg1[%arg3] : memref<?xf64>
// CHECK-NEXT:       affine.store %1, %alloca[%arg3 + %arg5 * 5 + %arg4 * 25] : memref<375xf64>
// CHECK-NEXT:       affine.store %1, %alloca[%arg3 + %arg5 * 5 + %arg4 * 25 + 250] : memref<375xf64>
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   %0 = affine.load %alloca[7] : memref<375xf64>
// CHECK-NEXT:   affine.store %0, %arg2[0] : memref<?xf64>
// CHECK-NEXT:   return
// CHECK-NEXT: }

// A larger buffer bounds the count only by 25, where two iterations of the
// outer loops can write one element: those stay serial. The inner loop is
// parallel: its iterations meet only 50 apart (one store 250 past the
// other), beyond the count.
func.func @larger(%n: index, %x: memref<?xf64>, %y: memref<?xf64>) {
  %a = memref.alloca() : memref<1000xf64>
  affine.for %i = 0 to %n {
    affine.for %j = 0 to %n {
      affine.for %k = 0 to %n {
        %v = affine.load %x[%i] : memref<?xf64>
        affine.store %v, %a[%i + %k * 5 + %j * 25] : memref<1000xf64>
        affine.store %v, %a[%i + %k * 5 + %j * 25 + 250] : memref<1000xf64>
      }
    }
  }
  %r = affine.load %a[7] : memref<1000xf64>
  affine.store %r, %y[0] : memref<?xf64>
  return
}

// CHECK:  func.func @larger(%arg0: index, %arg1: memref<?xf64>, %arg2: memref<?xf64>) {
// CHECK-NEXT:   %alloca = memref.alloca() : memref<1000xf64>
// CHECK-NEXT:   affine.for %arg3 = 0 to %arg0 {
// CHECK-NEXT:     affine.for %arg4 = 0 to %arg0 {
// CHECK-NEXT:       affine.parallel (%arg5) = (0) to (symbol(%arg0)) {
// CHECK-NEXT:         %1 = affine.load %arg1[%arg3] : memref<?xf64>
// CHECK-NEXT:         affine.store %1, %alloca[%arg3 + %arg5 * 5 + %arg4 * 25] : memref<1000xf64>
// CHECK-NEXT:         affine.store %1, %alloca[%arg3 + %arg5 * 5 + %arg4 * 25 + 250] : memref<1000xf64>
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   %0 = affine.load %alloca[7] : memref<1000xf64>
// CHECK-NEXT:   affine.store %0, %arg2[0] : memref<?xf64>
// CHECK-NEXT:   return
// CHECK-NEXT: }

// The store that bounds the count to 5 runs only at some points: its corner
// bounds nothing. The other store alone allows a count of 13, where the
// outer loops' iterations can meet; the inner loop's, 50 apart, cannot.
func.func @conditional(%n: index, %x: memref<?xf64>, %y: memref<?xf64>) {
  %a = memref.alloca() : memref<375xf64>
  affine.for %i = 0 to %n {
    affine.for %j = 0 to %n {
      affine.for %k = 0 to %n {
        %v = affine.load %x[%i] : memref<?xf64>
        affine.store %v, %a[%i + %k * 5 + %j * 25] : memref<375xf64>
        affine.if affine_set<(d0) : (d0 - 1 >= 0)>(%i) {
          affine.store %v, %a[%i + %k * 5 + %j * 25 + 250] : memref<375xf64>
        }
      }
    }
  }
  %r = affine.load %a[7] : memref<375xf64>
  affine.store %r, %y[0] : memref<?xf64>
  return
}

// CHECK:  func.func @conditional(%arg0: index, %arg1: memref<?xf64>, %arg2: memref<?xf64>) {
// CHECK-NEXT:   %alloca = memref.alloca() : memref<375xf64>
// CHECK-NEXT:   affine.for %arg3 = 0 to %arg0 {
// CHECK-NEXT:     affine.for %arg4 = 0 to %arg0 {
// CHECK-NEXT:       affine.parallel (%arg5) = (0) to (symbol(%arg0)) {
// CHECK-NEXT:         %1 = affine.load %arg1[%arg3] : memref<?xf64>
// CHECK-NEXT:         affine.store %1, %alloca[%arg3 + %arg5 * 5 + %arg4 * 25] : memref<375xf64>
// CHECK-NEXT:         affine.if #set1(%arg3) {
// CHECK-NEXT:           affine.store %1, %alloca[%arg3 + %arg5 * 5 + %arg4 * 25 + 250] : memref<375xf64>
// CHECK-NEXT:         }
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   %0 = affine.load %alloca[7] : memref<375xf64>
// CHECK-NEXT:   affine.store %0, %arg2[0] : memref<?xf64>
// CHECK-NEXT:   return
// CHECK-NEXT: }
