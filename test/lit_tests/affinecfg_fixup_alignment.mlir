// RUN: enzymexlamlir-opt --affine-cfg %s | FileCheck %s

func.func @aligned(%x: memref<?xf64>, %w: memref<25xf64>, %k: index) {
  affine.for %dy = 0 to 4 {
    %xv = affine.load %x[%k + %dy * 5] : memref<?xf64>
    %acc = affine.load %w[%k] {alignment = 8 : i64} : memref<25xf64>
    %s = arith.addf %acc, %xv : f64
    affine.store %s, %w[%k] {alignment = 8 : i64} : memref<25xf64>
  }
  return
}

// CHECK:  func.func @aligned(%arg0: memref<?xf64>, %arg1: memref<25xf64>, %arg2: index) {
// CHECK-NEXT:    %0 = affine.load %arg1[symbol(%arg2)] {alignment = 8 : i64} : memref<25xf64>
// CHECK-NEXT:    %1 = affine.parallel (%arg3) = (0) to (4) reduce ("addf") -> (f64) {
// CHECK-NEXT:      %3 = affine.load %arg0[%arg3 * 5 + symbol(%arg2)] : memref<?xf64>
// CHECK-NEXT:      affine.yield %3 : f64
// CHECK-NEXT:    }
// CHECK-NEXT:    %2 = arith.addf %0, %1 : f64
// CHECK-NEXT:    affine.store %2, %arg1[symbol(%arg2)] {alignment = 8 : i64} : memref<25xf64>
// CHECK-NEXT:    return
// CHECK-NEXT:  }
