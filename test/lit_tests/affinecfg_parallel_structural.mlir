// RUN: enzymexlamlir-opt %s --affine-cfg | FileCheck %s

// A loop over the rows of a kernel whose body holds a parallel loop over the
// row's elements: the nested affine.parallel only structures its accesses,
// which are checked like any other.
func.func @rows(%x: memref<?xf64>, %y: memref<?xf64>, %n: index) {
  affine.for %r = 0 to %n {
    affine.parallel (%c) = (0) to (16) {
      %a = affine.load %x[%r * 16 + %c] : memref<?xf64>
      %b = arith.mulf %a, %a : f64
      affine.store %b, %y[%r * 16 + %c] : memref<?xf64>
    }
  }
  return
}

// CHECK:  func.func @rows(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: index) {
// CHECK-NEXT:   affine.parallel (%arg3) = (0) to (symbol(%arg2) * 16) {
// CHECK-NEXT:     %0 = affine.load %arg0[%arg3] : memref<?xf64>
// CHECK-NEXT:     %1 = arith.mulf %0, %0 : f64
// CHECK-NEXT:     affine.store %1, %arg1[%arg3] : memref<?xf64>
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }

// A store under an scf.if: analyzed as if it always ran.
func.func @guarded(%x: memref<?xf64>, %y: memref<?xf64>, %n: index, %go: i1) {
  affine.for %i = 0 to %n {
    scf.if %go {
      %a = affine.load %x[%i] : memref<?xf64>
      affine.store %a, %y[%i] : memref<?xf64>
    }
  }
  return
}

// CHECK:  func.func @guarded(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: index, %arg3: i1) {
// CHECK-NEXT:   affine.parallel (%arg4) = (0) to (symbol(%arg2)) {
// CHECK-NEXT:     scf.if %arg3 {
// CHECK-NEXT:       %0 = affine.load %arg0[%arg4] : memref<?xf64>
// CHECK-NEXT:       affine.store %0, %arg1[%arg4] : memref<?xf64>
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }

// Under the scf.if, each iteration reads what the one before wrote: not
// parallel.
func.func @guarded_shift(%y: memref<?xf64>, %n: index, %go: i1) {
  affine.for %i = 0 to %n {
    scf.if %go {
      %a = affine.load %y[%i] : memref<?xf64>
      affine.store %a, %y[%i + 1] : memref<?xf64>
    }
  }
  return
}

// CHECK:  func.func @guarded_shift(%arg0: memref<?xf64>, %arg1: index, %arg2: i1) {
// CHECK-NEXT:   affine.for %arg3 = 0 to %arg1 {
// CHECK-NEXT:     scf.if %arg2 {
// CHECK-NEXT:       %0 = affine.load %arg0[%arg3] : memref<?xf64>
// CHECK-NEXT:       affine.store %0, %arg0[%arg3 + 1] : memref<?xf64>
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }
