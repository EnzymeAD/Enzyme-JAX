// RUN: enzymexlamlir-opt --affine-cfg --split-input-file %s | FileCheck %s

// A CSR transpose: the inner loop runs over a row's entries, its bounds
// read from the offsets, so it is not affine. It runs to the longest row
// instead, under `j < hi`, and the rows become its lanes.
func.func @csr_transpose(%off: memref<?xi32>, %idx: memref<?xi32>, %x: memref<?xf64>, %y: memref<?xf64>, %n: index) {
  %cst = arith.constant 0.0 : f64
  %c1_i32 = arith.constant 1 : i32
  affine.parallel (%i) = (0) to (symbol(%n)) {
    %lo = affine.load %off[%i] : memref<?xi32>
    %hi = affine.load %off[%i + 1] : memref<?xi32>
    %acc = scf.for %j = %lo to %hi step %c1_i32 iter_args(%a = %cst) -> (f64) : i32 {
      %ji = arith.index_cast %j : i32 to index
      %c = memref.load %idx[%ji] : memref<?xi32>
      %ci = arith.index_cast %c : i32 to index
      %v = memref.load %x[%ci] : memref<?xf64>
      %s = arith.addf %a, %v : f64
      scf.yield %s : f64
    }
    affine.store %acc, %y[%i] : memref<?xf64>
  }
  return
}

// CHECK:  func.func @csr_transpose(%arg0: memref<?xi32>, %arg1: memref<?xi32>, %arg2: memref<?xf64>, %arg3: memref<?xf64>, %arg4: index) {
// CHECK-NEXT:    %cst = arith.constant -0.000000e+00 : f64
// CHECK-NEXT:    %0 = affine.parallel (%arg5) = (0) to (symbol(%arg4)) reduce ("maxs") -> (i32) {
// CHECK-NEXT:      %2 = affine.load %arg0[%arg5] : memref<?xi32>
// CHECK-NEXT:      %3 = affine.load %arg0[%arg5 + 1] : memref<?xi32>
// CHECK-NEXT:      %4 = arith.subi %3, %2 : i32
// CHECK-NEXT:      affine.yield %4 : i32
// CHECK-NEXT:    }
// CHECK-NEXT:    %1 = arith.index_cast %0 : i32 to index
// CHECK-NEXT:    affine.parallel (%arg5) = (0) to (symbol(%arg4)) {
// CHECK-NEXT:      %2 = affine.load %arg0[%arg5] : memref<?xi32>
// CHECK-NEXT:      %3 = affine.load %arg0[%arg5 + 1] : memref<?xi32>
// CHECK-NEXT:      %4 = affine.parallel (%arg6) = (0) to (symbol(%1)) reduce ("addf") -> (f64) {
// CHECK-NEXT:        %5 = arith.index_cast %arg6 : index to i32
// CHECK-NEXT:        %6 = arith.addi %2, %5 : i32
// CHECK-NEXT:        %7 = arith.cmpi slt, %6, %3 : i32
// CHECK-NEXT:        %8 = scf.if %7 -> (f64) {
// CHECK-NEXT:          %9 = arith.index_cast %6 : i32 to index
// CHECK-NEXT:          %10 = memref.load %arg1[%9] : memref<?xi32>
// CHECK-NEXT:          %11 = arith.index_cast %10 : i32 to index
// CHECK-NEXT:          %12 = memref.load %arg2[%11] : memref<?xf64>
// CHECK-NEXT:          scf.yield %12 : f64
// CHECK-NEXT:        } else {
// CHECK-NEXT:          scf.yield %cst : f64
// CHECK-NEXT:        }
// CHECK-NEXT:        affine.yield %8 : f64
// CHECK-NEXT:      }
// CHECK-NEXT:      affine.store %4, %arg3[%arg5] : memref<?xf64>
// CHECK-NEXT:    }
// CHECK-NEXT:    return
// CHECK-NEXT:  }

// -----



// The rotated form clang makes of it, with a vector dimension between the
// row loop and the entries, and the bounds offset by one under the guard:
// the loop runs to the longest row of all the rows, and the guard stays.
func.func @rotated(%off: memref<?xi32>, %idx: memref<?xi32>, %x: memref<?xf64>, %y: memref<?xf64>, %n: index, %vdim: index) {
  %cst = arith.constant 0.0 : f64
  %c1_i32 = arith.constant 1 : i32
  affine.parallel (%i) = (0) to (symbol(%n)) {
    %lo = affine.load %off[%i] : memref<?xi32>
    %hi = affine.load %off[%i + 1] : memref<?xi32>
    %runs = arith.cmpi slt, %lo, %hi : i32
    affine.for %c = 0 to %vdim {
      %acc = scf.if %runs -> (f64) {
        %lo1 = arith.addi %lo, %c1_i32 : i32
        %hi1 = arith.addi %hi, %c1_i32 : i32
        %r = scf.for %j = %lo1 to %hi1 step %c1_i32 iter_args(%a = %cst) -> (f64) : i32 {
          %jm = arith.subi %j, %c1_i32 : i32
          %ji = arith.index_cast %jm : i32 to index
          %col = memref.load %idx[%ji] : memref<?xi32>
          %ci = arith.index_cast %col : i32 to index
          %v = memref.load %x[%ci] : memref<?xf64>
          %s = arith.addf %a, %v : f64
          scf.yield %s : f64
        }
        scf.yield %r : f64
      } else {
        scf.yield %cst : f64
      }
      affine.store %acc, %y[%i + %c * symbol(%n)] : memref<?xf64>
    }
  }
  return
}

// CHECK:  func.func @rotated(%arg0: memref<?xi32>, %arg1: memref<?xi32>, %arg2: memref<?xf64>, %arg3: memref<?xf64>, %arg4: index, %arg5: index) {
// CHECK-NEXT:    %cst = arith.constant -0.000000e+00 : f64
// CHECK-NEXT:    %cst_0 = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:    %c1_i32 = arith.constant 1 : i32
// CHECK-NEXT:    %0 = affine.parallel (%arg6) = (0) to (symbol(%arg4)) reduce ("maxs") -> (i32) {
// CHECK-NEXT:      %2 = affine.load %arg0[%arg6] : memref<?xi32>
// CHECK-NEXT:      %3 = affine.load %arg0[%arg6 + 1] : memref<?xi32>
// CHECK-NEXT:      %4 = arith.addi %2, %c1_i32 : i32
// CHECK-NEXT:      %5 = arith.addi %3, %c1_i32 : i32
// CHECK-NEXT:      %6 = arith.subi %5, %4 : i32
// CHECK-NEXT:      affine.yield %6 : i32
// CHECK-NEXT:    }
// CHECK-NEXT:    %1 = arith.index_cast %0 : i32 to index
// CHECK-NEXT:    affine.parallel (%arg6) = (0) to (symbol(%arg4)) {
// CHECK-NEXT:      %2 = affine.load %arg0[%arg6] : memref<?xi32>
// CHECK-NEXT:      %3 = affine.load %arg0[%arg6 + 1] : memref<?xi32>
// CHECK-NEXT:      %4 = arith.cmpi slt, %2, %3 : i32
// CHECK-NEXT:      affine.for %arg7 = 0 to %arg5 {
// CHECK-NEXT:        %5 = scf.if %4 -> (f64) {
// CHECK-NEXT:          %6 = arith.addi %3, %c1_i32 : i32
// CHECK-NEXT:          %7 = affine.parallel (%arg8) = (0) to (symbol(%1)) reduce ("addf") -> (f64) {
// CHECK-NEXT:            %8 = arith.index_cast %arg8 : index to i32
// CHECK-NEXT:            %9 = arith.addi %8, %2 : i32
// CHECK-NEXT:            %10 = arith.addi %9, %c1_i32 : i32
// CHECK-NEXT:            %11 = arith.cmpi slt, %10, %6 : i32
// CHECK-NEXT:            %12 = scf.if %11 -> (f64) {
// CHECK-NEXT:              %13 = arith.index_cast %9 : i32 to index
// CHECK-NEXT:              %14 = memref.load %arg1[%13] : memref<?xi32>
// CHECK-NEXT:              %15 = arith.index_cast %14 : i32 to index
// CHECK-NEXT:              %16 = memref.load %arg2[%15] : memref<?xf64>
// CHECK-NEXT:              scf.yield %16 : f64
// CHECK-NEXT:            } else {
// CHECK-NEXT:              scf.yield %cst : f64
// CHECK-NEXT:            }
// CHECK-NEXT:            affine.yield %12 : f64
// CHECK-NEXT:          }
// CHECK-NEXT:          scf.yield %7 : f64
// CHECK-NEXT:        } else {
// CHECK-NEXT:          scf.yield %cst_0 : f64
// CHECK-NEXT:        }
// CHECK-NEXT:        affine.store %5, %arg3[%arg6 + %arg7 * symbol(%arg4)] : memref<?xf64>
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// CHECK-NEXT:    return
// CHECK-NEXT:  }

// -----



// Bounds varying with two loops: the reduction is over both.
func.func @two_rows(%off: memref<?xi32>, %x: memref<?xf64>, %y: memref<?xf64>, %n: index) {
  %cst = arith.constant 0.0 : f64
  %c1 = arith.constant 1 : index
  affine.parallel (%i) = (0) to (symbol(%n)) {
    affine.for %c = 0 to 3 {
      %lo = affine.load %off[%i * 3 + %c] : memref<?xi32>
      %hi = affine.load %off[%i * 3 + %c + 1] : memref<?xi32>
      %loi = arith.index_cast %lo : i32 to index
      %hii = arith.index_cast %hi : i32 to index
      %acc = scf.for %j = %loi to %hii step %c1 iter_args(%a = %cst) -> (f64) {
        %v = memref.load %x[%j] : memref<?xf64>
        %s = arith.addf %a, %v : f64
        scf.yield %s : f64
      }
      affine.store %acc, %y[%i * 3 + %c] : memref<?xf64>
    }
  }
  return
}

// CHECK:  func.func @two_rows(%arg0: memref<?xi32>, %arg1: memref<?xf64>, %arg2: memref<?xf64>, %arg3: index) {
// CHECK-NEXT:    %cst = arith.constant -0.000000e+00 : f64
// CHECK-NEXT:    %0 = affine.parallel (%arg4) = (0) to (symbol(%arg3)) reduce ("maxs") -> (i64) {
// CHECK-NEXT:      %2 = affine.parallel (%arg5) = (0) to (3) reduce ("maxs") -> (i64) {
// CHECK-NEXT:        %3 = affine.load %arg0[%arg5 + %arg4 * 3] : memref<?xi32>
// CHECK-NEXT:        %4 = affine.load %arg0[%arg5 + %arg4 * 3 + 1] : memref<?xi32>
// CHECK-NEXT:        %5 = arith.index_cast %3 : i32 to index
// CHECK-NEXT:        %6 = arith.index_cast %4 : i32 to index
// CHECK-NEXT:        %7 = arith.subi %6, %5 : index
// CHECK-NEXT:        %8 = arith.index_cast %7 : index to i64
// CHECK-NEXT:        affine.yield %8 : i64
// CHECK-NEXT:      }
// CHECK-NEXT:      affine.yield %2 : i64
// CHECK-NEXT:    }
// CHECK-NEXT:    %1 = arith.index_cast %0 : i64 to index
// CHECK-NEXT:    affine.parallel (%arg4) = (0) to (symbol(%arg3)) {
// CHECK-NEXT:      affine.for %arg5 = 0 to 3 {
// CHECK-NEXT:        %2 = affine.load %arg0[%arg5 + %arg4 * 3] : memref<?xi32>
// CHECK-NEXT:        %3 = affine.load %arg0[%arg5 + %arg4 * 3 + 1] : memref<?xi32>
// CHECK-NEXT:        %4 = arith.index_cast %2 : i32 to index
// CHECK-NEXT:        %5 = arith.index_cast %3 : i32 to index
// CHECK-NEXT:        %6 = affine.parallel (%arg6) = (0) to (symbol(%1)) reduce ("addf") -> (f64) {
// CHECK-NEXT:          %7 = arith.addi %4, %arg6 : index
// CHECK-NEXT:          %8 = arith.cmpi slt, %7, %5 : index
// CHECK-NEXT:          %9 = scf.if %8 -> (f64) {
// CHECK-NEXT:            %10 = memref.load %arg1[%7] : memref<?xf64>
// CHECK-NEXT:            scf.yield %10 : f64
// CHECK-NEXT:          } else {
// CHECK-NEXT:            scf.yield %cst : f64
// CHECK-NEXT:          }
// CHECK-NEXT:          affine.yield %9 : f64
// CHECK-NEXT:        }
// CHECK-NEXT:        affine.store %6, %arg2[%arg5 + %arg4 * 3] : memref<?xf64>
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// CHECK-NEXT:    return
// CHECK-NEXT:  }

// -----



// A bound read under a condition does not run on every row: the loop
// stays.
func.func @guarded_read(%off: memref<?xi32>, %flags: memref<?xi1>, %x: memref<?xf64>, %y: memref<?xf64>, %n: index) {
  %cst = arith.constant 0.0 : f64
  %c1_i32 = arith.constant 1 : i32
  %c0_i32 = arith.constant 0 : i32
  affine.parallel (%i) = (0) to (symbol(%n)) {
    %f = affine.load %flags[%i] : memref<?xi1>
    %lo = affine.load %off[%i] : memref<?xi32>
    %hi = scf.if %f -> (i32) {
      %h = affine.load %off[%i + 1] : memref<?xi32>
      scf.yield %h : i32
    } else {
      scf.yield %c0_i32 : i32
    }
    %acc = scf.for %j = %lo to %hi step %c1_i32 iter_args(%a = %cst) -> (f64) : i32 {
      %ji = arith.index_cast %j : i32 to index
      %v = memref.load %x[%ji] : memref<?xf64>
      %s = arith.addf %a, %v : f64
      scf.yield %s : f64
    }
    affine.store %acc, %y[%i] : memref<?xf64>
  }
  return
}

// CHECK:  func.func @guarded_read(%arg0: memref<?xi32>, %arg1: memref<?xi1>, %arg2: memref<?xf64>, %arg3: memref<?xf64>, %arg4: index) {
// CHECK-NEXT:    %cst = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:    %c1_i32 = arith.constant 1 : i32
// CHECK-NEXT:    %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:    affine.parallel (%arg5) = (0) to (symbol(%arg4)) {
// CHECK-NEXT:      %0 = affine.load %arg1[%arg5] : memref<?xi1>
// CHECK-NEXT:      %1 = affine.load %arg0[%arg5] : memref<?xi32>
// CHECK-NEXT:      %2 = scf.if %0 -> (i32) {
// CHECK-NEXT:        %4 = affine.load %arg0[%arg5 + 1] : memref<?xi32>
// CHECK-NEXT:        scf.yield %4 : i32
// CHECK-NEXT:      } else {
// CHECK-NEXT:        scf.yield %c0_i32 : i32
// CHECK-NEXT:      }
// CHECK-NEXT:      %3 = scf.for %arg6 = %1 to %2 step %c1_i32 iter_args(%arg7 = %cst) -> (f64)  : i32 {
// CHECK-NEXT:        %4 = arith.index_cast %arg6 : i32 to index
// CHECK-NEXT:        %5 = memref.load %arg2[%4] : memref<?xf64>
// CHECK-NEXT:        %6 = arith.addf %arg7, %5 : f64
// CHECK-NEXT:        scf.yield %6 : f64
// CHECK-NEXT:      }
// CHECK-NEXT:      affine.store %3, %arg3[%arg5] : memref<?xf64>
// CHECK-NEXT:    }
// CHECK-NEXT:    return
// CHECK-NEXT:  }

// -----



// The offsets are written in the row loop: the loop stays.
func.func @written(%off: memref<?xi32>, %x: memref<?xf64>, %y: memref<?xf64>, %n: index) {
  %cst = arith.constant 0.0 : f64
  %c1_i32 = arith.constant 1 : i32
  affine.parallel (%i) = (0) to (symbol(%n)) {
    %lo = affine.load %off[%i] : memref<?xi32>
    %hi = affine.load %off[%i + 1] : memref<?xi32>
    %acc = scf.for %j = %lo to %hi step %c1_i32 iter_args(%a = %cst) -> (f64) : i32 {
      %ji = arith.index_cast %j : i32 to index
      %v = memref.load %x[%ji] : memref<?xf64>
      %s = arith.addf %a, %v : f64
      scf.yield %s : f64
    }
    affine.store %acc, %y[%i] : memref<?xf64>
    affine.store %hi, %off[%i] : memref<?xi32>
  }
  return
}

// CHECK:  func.func @written(%arg0: memref<?xi32>, %arg1: memref<?xf64>, %arg2: memref<?xf64>, %arg3: index) {
// CHECK-NEXT:    %cst = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:    %c1_i32 = arith.constant 1 : i32
// CHECK-NEXT:    affine.parallel (%arg4) = (0) to (symbol(%arg3)) {
// CHECK-NEXT:      %0 = affine.load %arg0[%arg4] : memref<?xi32>
// CHECK-NEXT:      %1 = affine.load %arg0[%arg4 + 1] : memref<?xi32>
// CHECK-NEXT:      %2 = scf.for %arg5 = %0 to %1 step %c1_i32 iter_args(%arg6 = %cst) -> (f64)  : i32 {
// CHECK-NEXT:        %3 = arith.index_cast %arg5 : i32 to index
// CHECK-NEXT:        %4 = memref.load %arg1[%3] : memref<?xf64>
// CHECK-NEXT:        %5 = arith.addf %arg6, %4 : f64
// CHECK-NEXT:        scf.yield %5 : f64
// CHECK-NEXT:      }
// CHECK-NEXT:      affine.store %2, %arg2[%arg4] : memref<?xf64>
// CHECK-NEXT:      affine.store %1, %arg0[%arg4] : memref<?xi32>
// CHECK-NEXT:    }
// CHECK-NEXT:    return
// CHECK-NEXT:  }
