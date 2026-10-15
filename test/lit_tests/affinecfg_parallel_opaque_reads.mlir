// RUN: enzymexlamlir-opt %s --affine-cfg | FileCheck %s

// A gather through an index table: the reads are at indices the analysis
// cannot follow, but nothing in the loop writes a buffer they may read (the
// buffers are noalias arguments), so they meet no write and the loop is
// parallel.
func.func @gather(%x: memref<?xf64> {llvm.noalias}, %idx: memref<?xindex>, %y: memref<?xf64> {llvm.noalias}, %n: index) {
  affine.for %i = 0 to %n {
    %k = affine.load %idx[%i] : memref<?xindex>
    %v = memref.load %x[%k] : memref<?xf64>
    affine.store %v, %y[%i] : memref<?xf64>
  }
  return
}

// CHECK:  func.func @gather(%arg0: memref<?xf64> {llvm.noalias}, %arg1: memref<?xindex>, %arg2: memref<?xf64> {llvm.noalias}, %arg3: index) {
// CHECK-NEXT:   affine.parallel (%arg4) = (0) to (symbol(%arg3)) {
// CHECK-NEXT:     %0 = affine.load %arg1[%arg4] : memref<?xindex>
// CHECK-NEXT:     %1 = memref.load %arg0[%0] : memref<?xf64>
// CHECK-NEXT:     affine.store %1, %arg2[%arg4] : memref<?xf64>
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }

// The loop writes the buffer it reads at an index it cannot follow: not
// parallel.
func.func @gather_written(%x: memref<?xf64>, %idx: memref<?xindex>, %n: index) {
  affine.for %i = 0 to %n {
    %k = affine.load %idx[%i] : memref<?xindex>
    %v = memref.load %x[%k] : memref<?xf64>
    affine.store %v, %x[%i] : memref<?xf64>
  }
  return
}

// CHECK:  func.func @gather_written(%arg0: memref<?xf64>, %arg1: memref<?xindex>, %arg2: index) {
// CHECK-NEXT:   affine.for %arg3 = 0 to %arg2 {
// CHECK-NEXT:     %0 = affine.load %arg1[%arg3] : memref<?xindex>
// CHECK-NEXT:     %1 = memref.load %arg0[%0] : memref<?xf64>
// CHECK-NEXT:     affine.store %1, %arg0[%arg3] : memref<?xf64>
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }

// The loop writes a buffer that may be the one it reads at an index it
// cannot follow (two arguments, neither noalias): not parallel.
func.func @gather_may_alias(%x: memref<?xf64>, %idx: memref<?xindex>, %y: memref<?xf64>, %n: index) {
  affine.for %i = 0 to %n {
    %k = affine.load %idx[%i] : memref<?xindex>
    %v = memref.load %x[%k] : memref<?xf64>
    affine.store %v, %y[%i] : memref<?xf64>
  }
  return
}

// CHECK:  func.func @gather_may_alias(%arg0: memref<?xf64>, %arg1: memref<?xindex>, %arg2: memref<?xf64>, %arg3: index) {
// CHECK-NEXT:   affine.for %arg4 = 0 to %arg3 {
// CHECK-NEXT:     %0 = affine.load %arg1[%arg4] : memref<?xindex>
// CHECK-NEXT:     %1 = memref.load %arg0[%0] : memref<?xf64>
// CHECK-NEXT:     affine.store %1, %arg2[%arg4] : memref<?xf64>
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }
