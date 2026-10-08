// RUN: enzymexlamlir-opt %s --affine-cfg | FileCheck %s

// The extent of a component, D1D + ((c == 2) xor transposed), reads a flag
// of the kernel together with the component loop's variable: the loop is
// split on the flag, and under each value the extent is a conditional of
// the variable alone.
func.func @xor_flag(%D1D: i32, %flag: i1, %X: memref<?xf64>, %Y: memref<?xf64>, %ne: index) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c2_i32 = arith.constant 2 : i32
  %cst = arith.constant 1.0 : f64
  affine.parallel (%e) = (0) to (symbol(%ne)) {
    affine.for %c = 0 to 3 {
      %ci = arith.index_cast %c : index to i32
      %is2 = arith.cmpi eq, %ci, %c2_i32 : i32
      %x = arith.xori %is2, %flag : i1
      %ext = arith.extsi %x : i1 to i32
      %n = arith.addi %D1D, %ext : i32
      %nmax = arith.maxsi %n, %c1_i32 : i32
      %ub = arith.addi %nmax, %c1_i32 : i32
      scf.for %dz = %c1_i32 to %ub step %c1_i32 : i32 {
        %dzm = arith.addi %dz, %c1_i32 : i32
        %i = arith.index_cast %dzm : i32 to index
        %v = memref.load %X[%i] : memref<?xf64>
        %w = arith.addf %v, %cst : f64
        memref.store %w, %Y[%i] : memref<?xf64>
      }
    }
  }
  return
}

// A bound that reads the flag without a loop variable is a symbol as it
// is: no split.
func.func @select_flag(%n: index, %m: index, %flag: i1, %X: memref<?xf64>) {
  %cst = arith.constant 1.0 : f64
  %ub = arith.select %flag, %n, %m : index
  affine.for %i = 0 to %ub {
    %v = affine.load %X[%i] : memref<?xf64>
    %w = arith.addf %v, %cst : f64
    affine.store %w, %X[%i] : memref<?xf64>
  }
  return
}

// CHECK:  func.func @xor_flag(%arg0: i32, %arg1: i1, %arg2: memref<?xf64>, %arg3: memref<?xf64>, %arg4: index) {
// CHECK-NEXT:   %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:   %c-1_i32 = arith.constant -1 : i32
// CHECK-NEXT:   %c1_i32 = arith.constant 1 : i32
// CHECK-NEXT:   %cst = arith.constant 1.000000e+00 : f64
// CHECK-NEXT:   affine.parallel (%arg5) = (0) to (symbol(%arg4)) {
// CHECK-NEXT:     affine.for %arg6 = 0 to 3 {
// CHECK-NEXT:       scf.if %arg1 {
// CHECK-NEXT:         %0 = affine.if #set(%arg6) -> i32 {
// CHECK-NEXT:           affine.yield %c0_i32 : i32
// CHECK-NEXT:         } else {
// CHECK-NEXT:           affine.yield %c-1_i32 : i32
// CHECK-NEXT:         }
// CHECK-NEXT:         %1 = arith.addi %arg0, %0 : i32
// CHECK-NEXT:         %2 = arith.maxsi %1, %c1_i32 : i32
// CHECK-NEXT:         %3 = arith.addi %2, %c1_i32 : i32
// CHECK-NEXT:         scf.for %arg7 = %c1_i32 to %3 step %c1_i32  : i32 {
// CHECK-NEXT:           %4 = arith.addi %arg7, %c1_i32 : i32
// CHECK-NEXT:           %5 = arith.index_cast %4 : i32 to index
// CHECK-NEXT:           %6 = memref.load %arg2[%5] : memref<?xf64>
// CHECK-NEXT:           %7 = arith.addf %6, %cst : f64
// CHECK-NEXT:           memref.store %7, %arg3[%5] : memref<?xf64>
// CHECK-NEXT:         }
// CHECK-NEXT:       } else {
// CHECK-NEXT:         %0 = affine.if #set(%arg6) -> i32 {
// CHECK-NEXT:           affine.yield %c-1_i32 : i32
// CHECK-NEXT:         } else {
// CHECK-NEXT:           affine.yield %c0_i32 : i32
// CHECK-NEXT:         }
// CHECK-NEXT:         %1 = arith.addi %arg0, %0 : i32
// CHECK-NEXT:         %2 = arith.maxsi %1, %c1_i32 : i32
// CHECK-NEXT:         %3 = arith.addi %2, %c1_i32 : i32
// CHECK-NEXT:         scf.for %arg7 = %c1_i32 to %3 step %c1_i32  : i32 {
// CHECK-NEXT:           %4 = arith.addi %arg7, %c1_i32 : i32
// CHECK-NEXT:           %5 = arith.index_cast %4 : i32 to index
// CHECK-NEXT:           %6 = memref.load %arg2[%5] : memref<?xf64>
// CHECK-NEXT:           %7 = arith.addf %6, %cst : f64
// CHECK-NEXT:           memref.store %7, %arg3[%5] : memref<?xf64>
// CHECK-NEXT:         }
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }

// CHECK:  func.func @select_flag(%arg0: index, %arg1: index, %arg2: i1, %arg3: memref<?xf64>) {
// CHECK-NEXT:   %cst = arith.constant 1.000000e+00 : f64
// CHECK-NEXT:   %0 = arith.select %arg2, %arg0, %arg1 : index
// CHECK-NEXT:   affine.parallel (%arg4) = (0) to (symbol(%0)) {
// CHECK-NEXT:     %1 = affine.load %arg3[%arg4] : memref<?xf64>
// CHECK-NEXT:     %2 = arith.addf %1, %cst : f64
// CHECK-NEXT:     affine.store %2, %arg3[%arg4] : memref<?xf64>
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }
