// RUN: enzymexlamlir-opt %s --affine-cfg | FileCheck %s

// A rotated loop, as mem2reg leaves `if (n > 0) do { s += ...; } while (...)`,
// carries besides the sum a copy of it, which the body never reads and only
// the last iteration sets: the loop's live-out. Where the loop runs the copy
// is the sum, and where it does not, its own initial value. The loop, rebuilt
// without the copy, keeps its attributes.
func.func @copy_unguarded(%x: memref<?xf64>, %n: index, %junk: f64) -> f64 {
  %z = arith.constant 0.0 : f64
  %r:2 = affine.for %i = 0 to %n iter_args(%s = %z, %l = %junk) -> (f64, f64) {
    %a = affine.load %x[%i] : memref<?xf64>
    %t = arith.addf %s, %a : f64
    affine.store %t, %x[%i + 1] : memref<?xf64>
    affine.yield %t, %t : f64, f64
  } {test.kept}
  return %r#1 : f64
}

// CHECK:  func.func @copy_unguarded(%arg0: memref<?xf64>, %arg1: index, %arg2: f64) -> f64 {
// CHECK-NEXT:   %cst = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:   %0 = affine.for %arg3 = 0 to %arg1 iter_args(%arg4 = %cst) -> (f64) {
// CHECK-NEXT:     %2 = affine.load %arg0[%arg3] : memref<?xf64>
// CHECK-NEXT:     %3 = arith.addf %arg4, %2 : f64
// CHECK-NEXT:     affine.store %3, %arg0[%arg3 + 1] : memref<?xf64>
// CHECK-NEXT:     affine.yield %3 : f64
// CHECK-NEXT:   } {test.kept}
// CHECK-NEXT:   %1 = affine.if #set()[%arg1] -> f64 {
// CHECK-NEXT:     affine.yield %0 : f64
// CHECK-NEXT:   } else {
// CHECK-NEXT:     affine.yield %arg2 : f64
// CHECK-NEXT:   }
// CHECK-NEXT:   return %1 : f64
// CHECK-NEXT: }

// In the check that the loop runs, the copy is the sum.
#runs = affine_set<()[s0] : (s0 - 1 >= 0)>
func.func @copy_in_check(%x: memref<?xf64>, %n: index, %junk: f64) -> f64 {
  %z = arith.constant 0.0 : f64
  %v = affine.if #runs()[%n] -> f64 {
    %r:2 = affine.for %i = 0 to %n iter_args(%s = %z, %l = %junk) -> (f64, f64) {
      %a = affine.load %x[%i] : memref<?xf64>
      %t = arith.addf %s, %a : f64
      affine.store %t, %x[%i + 1] : memref<?xf64>
      affine.yield %t, %t : f64, f64
    }
    affine.yield %r#1 : f64
  } else {
    affine.yield %z : f64
  }
  return %v : f64
}

// CHECK:  func.func @copy_in_check(%arg0: memref<?xf64>, %arg1: index, %arg2: f64) -> f64 {
// CHECK-NEXT:   %cst = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:   %0 = affine.for %arg3 = 0 to %arg1 iter_args(%arg4 = %cst) -> (f64) {
// CHECK-NEXT:     %1 = affine.load %arg0[%arg3] : memref<?xf64>
// CHECK-NEXT:     %2 = arith.addf %arg4, %1 : f64
// CHECK-NEXT:     affine.store %2, %arg0[%arg3 + 1] : memref<?xf64>
// CHECK-NEXT:     affine.yield %2 : f64
// CHECK-NEXT:   }
// CHECK-NEXT:   return %0 : f64
// CHECK-NEXT: }

// An scf.for that stays one (its bound is read in a loop): the copy becomes
// a select on the bounds.
func.func @scf_copy(%x: memref<?xf64>, %nb: memref<?xindex>, %m: index, %junk: f64, %out: memref<?xf64>) {
  %z = arith.constant 0.0 : f64
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  affine.for %j = 0 to %m {
    %n = memref.load %nb[%j] : memref<?xindex>
    %r:2 = scf.for %i = %c0 to %n step %c1 iter_args(%s = %z, %l = %junk) -> (f64, f64) {
      %a = memref.load %x[%i] : memref<?xf64>
      %t = arith.addf %s, %a : f64
      scf.yield %t, %t : f64, f64
    }
    memref.store %r#1, %out[%j] : memref<?xf64>
  }
  return
}

// CHECK:  func.func @scf_copy(%arg0: memref<?xf64>, %arg1: memref<?xindex>, %arg2: index, %arg3: f64, %arg4: memref<?xf64>) {
// CHECK-NEXT:   %cst = arith.constant -0.000000e+00 : f64
// CHECK-NEXT:   %c0 = arith.constant 0 : index
// CHECK-NEXT:   %0 = affine.parallel (%arg5) = (0) to (symbol(%arg2)) reduce ("maxs") -> (i64) {
// CHECK-NEXT:     %2 = affine.load %arg1[%arg5] : memref<?xindex>
// CHECK-NEXT:     %3 = arith.index_cast %2 : index to i64
// CHECK-NEXT:     affine.yield %3 : i64
// CHECK-NEXT:   }
// CHECK-NEXT:   %1 = arith.index_cast %0 : i64 to index
// CHECK-NEXT:   affine.parallel (%arg5) = (0) to (symbol(%arg2)) {
// CHECK-NEXT:     %2 = affine.load %arg1[%arg5] : memref<?xindex>
// CHECK-NEXT:     %3 = affine.parallel (%arg6) = (0) to (symbol(%1)) reduce ("addf") -> (f64) {
// CHECK-NEXT:       %6 = arith.cmpi slt, %arg6, %2 : index
// CHECK-NEXT:       %7 = scf.if %6 -> (f64) {
// CHECK-NEXT:         %8 = affine.load %arg0[%arg6] : memref<?xf64>
// CHECK-NEXT:         scf.yield %8 : f64
// CHECK-NEXT:       } else {
// CHECK-NEXT:         scf.yield %cst : f64
// CHECK-NEXT:       }
// CHECK-NEXT:       affine.yield %7 : f64
// CHECK-NEXT:     }
// CHECK-NEXT:     %4 = arith.cmpi sgt, %2, %c0 : index
// CHECK-NEXT:     %5 = arith.select %4, %3, %arg3 : f64
// CHECK-NEXT:     affine.store %5, %arg4[%arg5] : memref<?xf64>
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }
