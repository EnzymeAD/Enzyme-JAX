// RUN: enzymexlamlir-opt %s --affine-cfg | FileCheck %s

// A sum taken only where a condition holds: the if yields what it adds, or
// -0.0 (which leaves any sum as it is), and the loop reduces the sum.
func.func @scf_if(%n: index, %x: memref<?xf64>, %m: memref<?xi1>, %out: memref<1xf64>) {
  %zero = arith.constant 0.0 : f64
  %r = affine.for %i = 0 to %n iter_args(%acc = %zero) -> (f64) {
    %c = affine.load %m[%i] : memref<?xi1>
    %s = scf.if %c -> (f64) {
      %v = affine.load %x[%i] : memref<?xf64>
      %a = arith.addf %acc, %v : f64
      scf.yield %a : f64
    } else {
      scf.yield %acc : f64
    }
    affine.yield %s : f64
  }
  affine.store %r, %out[0] : memref<1xf64>
  return
}

// CHECK:  func.func @scf_if(%arg0: index, %arg1: memref<?xf64>, %arg2: memref<?xi1>, %arg3: memref<1xf64>) {
// CHECK-NEXT:   %cst = arith.constant -0.000000e+00 : f64
// CHECK-NEXT:   %0 = affine.parallel (%arg4) = (0) to (symbol(%arg0)) reduce ("addf") -> (f64) {
// CHECK-NEXT:     %1 = affine.load %arg2[%arg4] : memref<?xi1>
// CHECK-NEXT:     %2 = scf.if %1 -> (f64) {
// CHECK-NEXT:       %3 = affine.load %arg1[%arg4] : memref<?xf64>
// CHECK-NEXT:       scf.yield %3 : f64
// CHECK-NEXT:     } else {
// CHECK-NEXT:       scf.yield %cst : f64
// CHECK-NEXT:     }
// CHECK-NEXT:     affine.yield %2 : f64
// CHECK-NEXT:   }
// CHECK-NEXT:   affine.store %0, %arg3[0] : memref<1xf64>
// CHECK-NEXT:   return
// CHECK-NEXT: }

// The same under an affine.if on the iteration.
func.func @affine_if(%n: index, %x: memref<?xf64>, %out: memref<1xf64>) {
  %zero = arith.constant 0.0 : f64
  %r = affine.for %i = 0 to %n iter_args(%acc = %zero) -> (f64) {
    %s = affine.if affine_set<(d0) : (d0 - 2 >= 0)>(%i) -> f64 {
      %v = affine.load %x[%i] : memref<?xf64>
      %a = arith.addf %acc, %v : f64
      affine.yield %a : f64
    } else {
      affine.yield %acc : f64
    }
    affine.yield %s : f64
  }
  affine.store %r, %out[0] : memref<1xf64>
  return
}

// CHECK:  func.func @affine_if(%arg0: index, %arg1: memref<?xf64>, %arg2: memref<1xf64>) {
// CHECK-NEXT:   %cst = arith.constant -0.000000e+00 : f64
// CHECK-NEXT:   %0 = affine.parallel (%arg3) = (0) to (symbol(%arg0)) reduce ("addf") -> (f64) {
// CHECK-NEXT:     %1 = affine.if #set(%arg3) -> f64 {
// CHECK-NEXT:       %2 = affine.load %arg1[%arg3] : memref<?xf64>
// CHECK-NEXT:       affine.yield %2 : f64
// CHECK-NEXT:     } else {
// CHECK-NEXT:       affine.yield %cst : f64
// CHECK-NEXT:     }
// CHECK-NEXT:     affine.yield %1 : f64
// CHECK-NEXT:   }
// CHECK-NEXT:   affine.store %0, %arg2[0] : memref<1xf64>
// CHECK-NEXT:   return
// CHECK-NEXT: }

// Through a select, an integer count.
func.func @select(%n: index, %m: memref<?xi1>, %out: memref<1xi32>) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %r = affine.for %i = 0 to %n iter_args(%acc = %c0) -> (i32) {
    %c = affine.load %m[%i] : memref<?xi1>
    %a = arith.addi %acc, %c1 : i32
    %s = arith.select %c, %a, %acc : i32
    affine.yield %s : i32
  }
  affine.store %r, %out[0] : memref<1xi32>
  return
}

// CHECK:  func.func @select(%arg0: index, %arg1: memref<?xi1>, %arg2: memref<1xi32>) {
// CHECK-NEXT:   %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:   %c1_i32 = arith.constant 1 : i32
// CHECK-NEXT:   %0 = affine.parallel (%arg3) = (0) to (symbol(%arg0)) reduce ("addi") -> (i32) {
// CHECK-NEXT:     %1 = affine.load %arg1[%arg3] : memref<?xi1>
// CHECK-NEXT:     %2 = arith.select %1, %c1_i32, %c0_i32 : i32
// CHECK-NEXT:     affine.yield %2 : i32
// CHECK-NEXT:   }
// CHECK-NEXT:   affine.store %0, %arg2[0] : memref<1xi32>
// CHECK-NEXT:   return
// CHECK-NEXT: }

// A multiply-add accumulation under a condition: the if yields the product.
func.func @fmuladd(%n: index, %x: memref<?xf64>, %m: memref<?xi1>, %out: memref<1xf64>) {
  %zero = arith.constant 0.0 : f64
  %r = affine.for %i = 0 to %n iter_args(%acc = %zero) -> (f64) {
    %c = affine.load %m[%i] : memref<?xi1>
    %s = scf.if %c -> (f64) {
      %v = affine.load %x[%i] : memref<?xf64>
      %a = enzymexla.math.fmuladd %v, %v, %acc : f64
      scf.yield %a : f64
    } else {
      scf.yield %acc : f64
    }
    affine.yield %s : f64
  }
  affine.store %r, %out[0] : memref<1xf64>
  return
}

// CHECK:  func.func @fmuladd(%arg0: index, %arg1: memref<?xf64>, %arg2: memref<?xi1>, %arg3: memref<1xf64>) {
// CHECK-NEXT:   %cst = arith.constant -0.000000e+00 : f64
// CHECK-NEXT:   %0 = affine.parallel (%arg4) = (0) to (symbol(%arg0)) reduce ("addf") -> (f64) {
// CHECK-NEXT:     %1 = affine.load %arg2[%arg4] : memref<?xi1>
// CHECK-NEXT:     %2 = scf.if %1 -> (f64) {
// CHECK-NEXT:       %3 = affine.load %arg1[%arg4] : memref<?xf64>
// CHECK-NEXT:       %4 = arith.mulf %3, %3 fastmath<contract> : f64
// CHECK-NEXT:       scf.yield %4 : f64
// CHECK-NEXT:     } else {
// CHECK-NEXT:       scf.yield %cst : f64
// CHECK-NEXT:     }
// CHECK-NEXT:     affine.yield %2 : f64
// CHECK-NEXT:   }
// CHECK-NEXT:   affine.store %0, %arg3[0] : memref<1xf64>
// CHECK-NEXT:   return
// CHECK-NEXT: }

// A value replaced where a condition holds (the last one taken) is no
// accumulation: left carried.
func.func @last(%n: index, %x: memref<?xf64>, %m: memref<?xi1>, %out: memref<1xf64>) {
  %zero = arith.constant 0.0 : f64
  %r = affine.for %i = 0 to %n iter_args(%acc = %zero) -> (f64) {
    %c = affine.load %m[%i] : memref<?xi1>
    %v = affine.load %x[%i] : memref<?xf64>
    %s = arith.select %c, %v, %acc : f64
    affine.yield %s : f64
  }
  affine.store %r, %out[0] : memref<1xf64>
  return
}

// CHECK:  func.func @last(%arg0: index, %arg1: memref<?xf64>, %arg2: memref<?xi1>, %arg3: memref<1xf64>) {
// CHECK-NEXT:   %cst = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:   %0 = affine.for %arg4 = 0 to %arg0 iter_args(%arg5 = %cst) -> (f64) {
// CHECK-NEXT:     %1 = affine.load %arg2[%arg4] : memref<?xi1>
// CHECK-NEXT:     %2 = affine.load %arg1[%arg4] : memref<?xf64>
// CHECK-NEXT:     %3 = arith.select %1, %2, %arg5 : f64
// CHECK-NEXT:     affine.yield %3 : f64
// CHECK-NEXT:   }
// CHECK-NEXT:   affine.store %0, %arg3[0] : memref<1xf64>
// CHECK-NEXT:   return
// CHECK-NEXT: }

// An accumulation under a condition in a loop that is not parallel
// otherwise (each iteration reads what the previous one wrote): nothing to
// reduce in parallel, left as it is.
func.func @serial(%n: index, %x: memref<?xf64>, %m: memref<?xi1>, %out: memref<1xf64>) {
  %zero = arith.constant 0.0 : f64
  %r = affine.for %i = 1 to %n iter_args(%acc = %zero) -> (f64) {
    %c = affine.load %m[%i] : memref<?xi1>
    %p = affine.load %x[%i - 1] : memref<?xf64>
    affine.store %p, %x[%i] : memref<?xf64>
    %s = scf.if %c -> (f64) {
      %a = arith.addf %acc, %p : f64
      scf.yield %a : f64
    } else {
      scf.yield %acc : f64
    }
    affine.yield %s : f64
  }
  affine.store %r, %out[0] : memref<1xf64>
  return
}

// CHECK:  func.func @serial(%arg0: index, %arg1: memref<?xf64>, %arg2: memref<?xi1>, %arg3: memref<1xf64>) {
// CHECK-NEXT:   %cst = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:   %0 = affine.for %arg4 = 1 to %arg0 iter_args(%arg5 = %cst) -> (f64) {
// CHECK-NEXT:     %1 = affine.load %arg2[%arg4] : memref<?xi1>
// CHECK-NEXT:     %2 = affine.load %arg1[%arg4 - 1] : memref<?xf64>
// CHECK-NEXT:     affine.store %2, %arg1[%arg4] : memref<?xf64>
// CHECK-NEXT:     %3 = scf.if %1 -> (f64) {
// CHECK-NEXT:       %4 = arith.addf %arg5, %2 : f64
// CHECK-NEXT:       scf.yield %4 : f64
// CHECK-NEXT:     } else {
// CHECK-NEXT:       scf.yield %arg5 : f64
// CHECK-NEXT:     }
// CHECK-NEXT:     affine.yield %3 : f64
// CHECK-NEXT:   }
// CHECK-NEXT:   affine.store %0, %arg3[0] : memref<1xf64>
// CHECK-NEXT:   return
// CHECK-NEXT: }
