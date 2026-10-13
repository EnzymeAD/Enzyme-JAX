// RUN: enzymexlamlir-opt --affine-cfg --split-input-file %s | FileCheck %s

#map = affine_map<(d0) -> (-d0 + 999)>
#map1 = affine_map<(d0, d1) -> (d0 + d1 + 1)>
func.func @accumulate_inbounds(%arg0: f64, %arg1: memref<1000x1000xf64>, %arg2: memref<1000x1200xf64>, %arg3: memref<1000x1200xf64>) {
  affine.for %i = 0 to 1000 {
    affine.for %j = 0 to 1200 {
      affine.for %k = 0 to #map(%i) {
        %idx = affine.apply #map1(%i, %k)
        %a = affine.load %arg1[%idx, %i] : memref<1000x1000xf64>
        %b = affine.load %arg2[%idx, %j] : memref<1000x1200xf64>
        %prod = arith.mulf %a, %b : f64
        %acc = affine.load %arg3[%i, %j] : memref<1000x1200xf64>
        %sum = arith.addf %acc, %prod : f64
        affine.store %sum, %arg3[%i, %j] : memref<1000x1200xf64>
      }
      %v = affine.load %arg3[%i, %j] : memref<1000x1200xf64>
      %scaled = arith.mulf %arg0, %v : f64
      affine.store %scaled, %arg3[%i, %j] : memref<1000x1200xf64>
    }
  }
  return
}

// CHECK-LABEL: func.func @accumulate_inbounds(
// CHECK-SAME: %arg0: f64, %arg1: memref<1000x1000xf64>, %arg2: memref<1000x1200xf64>, %arg3: memref<1000x1200xf64>) {
// CHECK-NEXT:   affine.parallel (%arg4, %arg5) = (0, 0) to (1000, 1200) {
// CHECK-NEXT:     %0 = affine.load %arg3[%arg4, %arg5] : memref<1000x1200xf64>
// CHECK-NEXT:     %1 = affine.parallel (%arg6) = (0) to (-%arg4 + 999) reduce ("addf") -> (f64) {
// CHECK-NEXT:       %5 = affine.load %arg1[%arg4 + %arg6 + 1, %arg4] : memref<1000x1000xf64>
// CHECK-NEXT:       %6 = affine.load %arg2[%arg4 + %arg6 + 1, %arg5] : memref<1000x1200xf64>
// CHECK-NEXT:       %7 = arith.mulf %5, %6 : f64
// CHECK-NEXT:       affine.yield %7 : f64
// CHECK-NEXT:     }
// CHECK-NEXT:     %2 = arith.addf %0, %1 : f64
// CHECK-NEXT:     affine.store %2, %arg3[%arg4, %arg5] : memref<1000x1200xf64>
// CHECK-NEXT:     %3 = affine.load %arg3[%arg4, %arg5] : memref<1000x1200xf64>
// CHECK-NEXT:     %4 = arith.mulf %arg0, %3 : f64
// CHECK-NEXT:     affine.store %4, %arg3[%arg4, %arg5] : memref<1000x1200xf64>
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }

// -----

// When the accumulated index is not provably in bounds on zero-trip iterations
// (e.g. %i = 999 on memref<999x1200xf64>), the affine.if guards must be kept.
#map = affine_map<(d0) -> (-d0 + 999)>
#map1 = affine_map<(d0, d1) -> (d0 + d1 + 1)>
func.func @accumulate_maybe_oob(%arg1: memref<1000x1000xf64>, %arg2: memref<1000x1200xf64>, %arg3: memref<999x1200xf64>) {
  affine.for %i = 0 to 1000 {
    affine.for %j = 0 to 1200 {
      affine.for %k = 0 to #map(%i) {
        %idx = affine.apply #map1(%i, %k)
        %a = affine.load %arg1[%idx, %i] : memref<1000x1000xf64>
        %b = affine.load %arg2[%idx, %j] : memref<1000x1200xf64>
        %prod = arith.mulf %a, %b : f64
        %acc = affine.load %arg3[%i, %j] : memref<999x1200xf64>
        %sum = arith.addf %acc, %prod : f64
        affine.store %sum, %arg3[%i, %j] : memref<999x1200xf64>
      }
    }
  }
  return
}

// CHECK-LABEL: func.func @accumulate_maybe_oob(
// CHECK-SAME: %arg0: memref<1000x1000xf64>, %arg1: memref<1000x1200xf64>, %arg2: memref<999x1200xf64>) {
// CHECK-NEXT:   %cst = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:   affine.parallel (%arg3, %arg4) = (0, 0) to (1000, 1200) {
// CHECK-NEXT:     %0 = affine.if #set(%arg3) -> f64 {
// CHECK-NEXT:       %3 = affine.load %arg2[%arg3, %arg4] : memref<999x1200xf64>
// CHECK-NEXT:       affine.yield %3 : f64
// CHECK-NEXT:     } else {
// CHECK-NEXT:       affine.yield %cst : f64
// CHECK-NEXT:     }
// CHECK-NEXT:     %1 = affine.parallel (%arg5) = (0) to (-%arg3 + 999) reduce ("addf") -> (f64) {
// CHECK-NEXT:       %3 = affine.load %arg0[%arg3 + %arg5 + 1, %arg3] : memref<1000x1000xf64>
// CHECK-NEXT:       %4 = affine.load %arg1[%arg3 + %arg5 + 1, %arg4] : memref<1000x1200xf64>
// CHECK-NEXT:       %5 = arith.mulf %3, %4 : f64
// CHECK-NEXT:       affine.yield %5 : f64
// CHECK-NEXT:     }
// CHECK-NEXT:     %2 = arith.addf %0, %1 : f64
// CHECK-NEXT:     affine.if #set(%arg3) {
// CHECK-NEXT:       affine.store %2, %arg2[%arg3, %arg4] : memref<999x1200xf64>
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }
