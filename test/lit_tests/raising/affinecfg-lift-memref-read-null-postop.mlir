// RUN: enzymexlamlir-opt %s --affine-cfg | FileCheck %s

func.func @lift_no_postop(%c: i1, %m: memref<?xindex>, %n: memref<?x?xf64>,
                          %out: memref<?xf64>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  scf.for %iv = %c0 to %c4 step %c1 {
    %r = scf.if %c -> (index) {
      scf.yield %c0 : index
    } else {
      scf.yield %c1 : index
    }
    %i = memref.load %m[%c0] : memref<?xindex>
    %v = memref.load %n[%r, %i] : memref<?x?xf64>
    memref.store %v, %out[%iv] : memref<?xf64>
  }
  return
}

// CHECK: func.func @lift_no_postop(%arg0: i1, %arg1: memref<?xindex>, %arg2: memref<?x?xf64>, %arg3: memref<?xf64>) {
// CHECK-NEXT:   %c0 = arith.constant 0 : index
// CHECK-NEXT:   %c1 = arith.constant 1 : index
// CHECK-NEXT:   affine.for %arg4 = 0 to 4 {
// CHECK-NEXT:   %0 = affine.load %arg1[0] : memref<?xindex>
// CHECK-NEXT:     %1:2 = scf.if %arg0 -> (index, f64) {
// CHECK-NEXT:       %2 = memref.load %arg2[%c0, %0] : memref<?x?xf64>
// CHECK-NEXT:       scf.yield %c0, %2 : index, f64
// CHECK-NEXT:     } else {
// CHECK-NEXT:       %2 = memref.load %arg2[%c1, %0] : memref<?x?xf64>
// CHECK-NEXT:       scf.yield %c1, %2 : index, f64
// CHECK-NEXT:     }
// CHECK-NEXT:     affine.store %1#1, %arg3[%arg4] : memref<?xf64>
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }
