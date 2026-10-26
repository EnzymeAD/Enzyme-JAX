// RUN: enzymexlamlir-opt --affine-cfg --split-input-file %s | FileCheck %s

// trmm: B[i][j] accumulated over k > i from B[k][j], of the same buffer. The
// loads read elements the store never writes within the loop, so the store
// moves past the loop and the loop is a parallel reduction.
#map = affine_map<(d0) -> (d0 + 1)>
#set = affine_set<(d0) : (-d0 + 998 >= 0)>
func.func @trmm(%alpha: memref<f64>, %A: memref<1000x1000xf64>, %B: memref<1000x1200xf64>) {
  %cst = arith.constant 0.000000e+00 : f64
  %0 = affine.load %alpha[] : memref<f64>
  affine.for %i = 0 to 1000 {
    affine.parallel (%j) = (0) to (1200) {
      %1 = affine.if #set(%i) -> f64 {
        %5 = affine.load %B[%i, %j] : memref<1000x1200xf64>
        affine.yield %5 : f64
      } else {
        affine.yield %cst : f64
      }
      %2 = affine.for %k = #map(%i) to 1000 iter_args(%acc = %1) -> (f64) {
        %5 = affine.load %A[%k, %i] : memref<1000x1000xf64>
        %6 = affine.load %B[%k, %j] : memref<1000x1200xf64>
        %7 = arith.mulf %5, %6 : f64
        %8 = arith.addf %acc, %7 : f64
        affine.store %8, %B[%i, %j] : memref<1000x1200xf64>
        affine.yield %8 : f64
      }
      %3 = affine.load %B[%i, %j] : memref<1000x1200xf64>
      %4 = arith.mulf %0, %3 : f64
      affine.store %4, %B[%i, %j] : memref<1000x1200xf64>
    }
  }
  return
}

// CHECK:  func.func @trmm(%arg0: memref<f64>, %arg1: memref<1000x1000xf64>, %arg2: memref<1000x1200xf64>) {
// CHECK-NEXT:    %cst = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:    %0 = affine.load %arg0[] : memref<f64>
// CHECK-NEXT:    affine.for %arg3 = 0 to 1000 {
// CHECK-NEXT:      affine.parallel (%arg4) = (0) to (1200) {
// CHECK-NEXT:        %1 = affine.if #set(%arg3) -> f64 {
// CHECK-NEXT:          %6 = affine.load %arg2[%arg3, %arg4] : memref<1000x1200xf64>
// CHECK-NEXT:          affine.yield %6 : f64
// CHECK-NEXT:        } else {
// CHECK-NEXT:          affine.yield %cst : f64
// CHECK-NEXT:        }
// CHECK-NEXT:        %2 = affine.parallel (%arg5) = (%arg3 + 1) to (1000) reduce ("addf") -> (f64) {
// CHECK-NEXT:          %6 = affine.load %arg1[%arg5, %arg3] : memref<1000x1000xf64>
// CHECK-NEXT:          %7 = affine.load %arg2[%arg5, %arg4] : memref<1000x1200xf64>
// CHECK-NEXT:          %8 = arith.mulf %6, %7 : f64
// CHECK-NEXT:          affine.yield %8 : f64
// CHECK-NEXT:        }
// CHECK-NEXT:        %3 = arith.addf %1, %2 : f64
// CHECK-NEXT:        affine.if #set(%arg3) {
// CHECK-NEXT:          affine.store %3, %arg2[%arg3, %arg4] : memref<1000x1200xf64>
// CHECK-NEXT:        }
// CHECK-NEXT:        %4 = affine.load %arg2[%arg3, %arg4] : memref<1000x1200xf64>
// CHECK-NEXT:        %5 = arith.mulf %0, %4 : f64
// CHECK-NEXT:        affine.store %5, %arg2[%arg3, %arg4] : memref<1000x1200xf64>
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// CHECK-NEXT:    return
// CHECK-NEXT:  }

// -----

// Over k <= i the loads read B[i][j], which the store writes: the store
// stays in the loop.
#map = affine_map<(d0) -> (d0 + 1)>
func.func @trmm_lower(%A: memref<1000x1000xf64>, %B: memref<1000x1200xf64>) {
  affine.for %i = 0 to 1000 {
    affine.parallel (%j) = (0) to (1200) {
      %1 = affine.load %B[%i, %j] : memref<1000x1200xf64>
      %2 = affine.for %k = 0 to #map(%i) iter_args(%acc = %1) -> (f64) {
        %5 = affine.load %A[%k, %i] : memref<1000x1000xf64>
        %6 = affine.load %B[%k, %j] : memref<1000x1200xf64>
        %7 = arith.mulf %5, %6 : f64
        %8 = arith.addf %acc, %7 : f64
        affine.store %8, %B[%i, %j] : memref<1000x1200xf64>
        affine.yield %8 : f64
      }
    }
  }
  return
}

// CHECK:  func.func @trmm_lower(%arg0: memref<1000x1000xf64>, %arg1: memref<1000x1200xf64>) {
// CHECK-NEXT:    affine.for %arg2 = 0 to 1000 {
// CHECK-NEXT:      affine.parallel (%arg3) = (0) to (1200) {
// CHECK-NEXT:        %0 = affine.load %arg1[%arg2, %arg3] : memref<1000x1200xf64>
// CHECK-NEXT:        %1 = affine.for %arg4 = 0 to #map(%arg2) iter_args(%arg5 = %0) -> (f64) {
// CHECK-NEXT:          %2 = affine.load %arg0[%arg4, %arg2] : memref<1000x1000xf64>
// CHECK-NEXT:          %3 = affine.load %arg1[%arg4, %arg3] : memref<1000x1200xf64>
// CHECK-NEXT:          %4 = arith.mulf %2, %3 : f64
// CHECK-NEXT:          %5 = arith.addf %arg5, %4 : f64
// CHECK-NEXT:          affine.store %5, %arg1[%arg2, %arg3] : memref<1000x1200xf64>
// CHECK-NEXT:          affine.yield %5 : f64
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// CHECK-NEXT:    return
// CHECK-NEXT:  }
