// RUN: enzymexlamlir-opt %s --affine-cfg | FileCheck %s

// The same view of the output is taken in both arms of a branch: two memref
// values of one object at no offset from it. The dependence analysis compares
// the accesses through both as accesses of one buffer, and the loop is
// parallel.
module {
  func.func @arms(%p: !llvm.ptr, %x: memref<?xf64>, %n: index, %flag: i1) {
    affine.for %i = 0 to %n {
      %v = affine.load %x[%i] : memref<?xf64>
      scf.if %flag {
        %m = "enzymexla.pointer2memref"(%p) : (!llvm.ptr) -> memref<?xf64>
        affine.store %v, %m[%i] : memref<?xf64>
      } else {
        %m = "enzymexla.pointer2memref"(%p) : (!llvm.ptr) -> memref<?xf64>
        %w = arith.negf %v : f64
        affine.store %w, %m[%i] : memref<?xf64>
      }
    }
    return
  }
}


// CHECK:  func.func @arms(%arg0: !llvm.ptr, %arg1: memref<?xf64>, %arg2: index, %arg3: i1) {
// CHECK-NEXT:   affine.parallel (%arg4) = (0) to (symbol(%arg2)) {
// CHECK-NEXT:     %0 = affine.load %arg1[%arg4] : memref<?xf64>
// CHECK-NEXT:     scf.if %arg3 {
// CHECK-NEXT:       %1 = "enzymexla.pointer2memref"(%arg0) : (!llvm.ptr) -> memref<?xf64>
// CHECK-NEXT:       affine.store %0, %1[%arg4] : memref<?xf64>
// CHECK-NEXT:     } else {
// CHECK-NEXT:       %1 = "enzymexla.pointer2memref"(%arg0) : (!llvm.ptr) -> memref<?xf64>
// CHECK-NEXT:       %2 = arith.negf %0 : f64
// CHECK-NEXT:       affine.store %2, %1[%arg4] : memref<?xf64>
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }
