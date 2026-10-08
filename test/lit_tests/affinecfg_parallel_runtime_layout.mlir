// RUN: enzymexlamlir-opt %s --affine-cfg | FileCheck %s

// Loops over a buffer laid out by runtime sizes, as a kernel's
// Reshape(op, coeffDim, NQ, NE) reads it at c + coeffDim * (q + NQ * e).

// The rotation of a loop that might run no times leaves its bound as
// max(n, 1), while its accesses stride by n: the iterations writing rows
// 2e and 2e + 1 of width n then seem to meet. Past a check that n is
// positive the bound is n, and they stay apart.
llvm.func @abort() attributes {noreturn}
func.func @rows(%x: memref<?xf64>, %y: memref<?xf64>, %n32: i32, %ne: index) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i64 = arith.constant 1 : i64
  %n = arith.index_cast %n32 : i32 to index
  %n64 = arith.extui %n32 : i32 to i64
  %m = arith.maxsi %n64, %c1_i64 : i64
  %bound = arith.index_cast %m : i64 to index
  %bad = arith.cmpi sle, %n32, %c0_i32 : i32
  cf.cond_br %bad, ^fail, ^ok
^fail:
  llvm.call @abort() : () -> ()
  llvm.unreachable
^ok:
  affine.for %e = 0 to %ne {
    affine.for %i = 0 to %bound {
      %v = affine.load %x[%i + %e * symbol(%n)] : memref<?xf64>
      affine.store %v, %y[%i + (%e * 2) * symbol(%n)] : memref<?xf64>
      affine.store %v, %y[%i + (%e * 2 + 1) * symbol(%n)] : memref<?xf64>
    }
  }
  return
}

// CHECK:  func.func @rows(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: i32, %arg3: index) {
// CHECK-NEXT:   %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:   %c1_i64 = arith.constant 1 : i64
// CHECK-NEXT:   %0 = arith.index_cast %arg2 : i32 to index
// CHECK-NEXT:   %1 = arith.extui %arg2 : i32 to i64
// CHECK-NEXT:   %2 = arith.maxsi %1, %c1_i64 : i64
// CHECK-NEXT:   %3 = arith.index_cast %2 : i64 to index
// CHECK-NEXT:   %4 = arith.cmpi sle, %arg2, %c0_i32 : i32
// CHECK-NEXT:   cf.cond_br %4, ^bb1, ^bb2
// CHECK-NEXT: ^bb1:  // pred: ^bb0
// CHECK-NEXT:   llvm.call @abort() : () -> ()
// CHECK-NEXT:   llvm.unreachable
// CHECK-NEXT: ^bb2:  // pred: ^bb0
// CHECK-NEXT:   affine.parallel (%arg4, %arg5) = (0, 0) to (symbol(%arg3), symbol(%3)) {
// CHECK-NEXT:     %5 = affine.load %arg0[%arg5 + %arg4 * symbol(%0)] : memref<?xf64>
// CHECK-NEXT:     affine.store %5, %arg1[%arg5 + (%arg4 * symbol(%0)) * 2] : memref<?xf64>
// CHECK-NEXT:     affine.store %5, %arg1[%arg5 + (%arg4 * 2 + 1) * symbol(%0)] : memref<?xf64>
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }

// Without the check, n may be 0 with the loop running once at i = 0: the
// bound says nothing about n, and the nest stays serial.
func.func @unchecked(%x: memref<?xf64>, %y: memref<?xf64>, %n32: i32, %ne: index) {
  %c1_i64 = arith.constant 1 : i64
  %n = arith.index_cast %n32 : i32 to index
  %n64 = arith.extui %n32 : i32 to i64
  %m = arith.maxsi %n64, %c1_i64 : i64
  %bound = arith.index_cast %m : i64 to index
  affine.for %e = 0 to %ne {
    affine.for %i = 0 to %bound {
      %v = affine.load %x[%i + %e * symbol(%n)] : memref<?xf64>
      affine.store %v, %y[%i + (%e * 2) * symbol(%n)] : memref<?xf64>
      affine.store %v, %y[%i + (%e * 2 + 1) * symbol(%n)] : memref<?xf64>
    }
  }
  return
}

// CHECK:  func.func @unchecked(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: i32, %arg3: index) {
// CHECK-NEXT:   %c1_i64 = arith.constant 1 : i64
// CHECK-NEXT:   %0 = arith.index_cast %arg2 : i32 to index
// CHECK-NEXT:   %1 = arith.extui %arg2 : i32 to i64
// CHECK-NEXT:   %2 = arith.maxsi %1, %c1_i64 : i64
// CHECK-NEXT:   %3 = arith.index_cast %2 : i64 to index
// CHECK-NEXT:   affine.for %arg4 = 0 to %arg3 {
// CHECK-NEXT:     affine.for %arg5 = 0 to %3 {
// CHECK-NEXT:       %4 = affine.load %arg0[%arg5 + %arg4 * symbol(%0)] : memref<?xf64>
// CHECK-NEXT:       affine.store %4, %arg1[%arg5 + (%arg4 * symbol(%0)) * 2] : memref<?xf64>
// CHECK-NEXT:       affine.store %4, %arg1[%arg5 + (%arg4 * 2 + 1) * symbol(%0)] : memref<?xf64>
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }

// The count is read as an index two ways, cast directly and through a zero
// extension; a check that it is positive makes them one value, and the
// width the nest strides by is the bound of the loop it is read in.
func.func @index_forms(%n32: i32, %x: memref<?xf64>, %y: memref<?xf64>) {
  %c0_i32 = arith.constant 0 : i32
  %pos = arith.cmpi sgt, %n32, %c0_i32 : i32
  %n = arith.index_cast %n32 : i32 to index
  %n64 = arith.extui %n32 : i32 to i64
  %w = arith.index_cast %n64 : i64 to index
  scf.if %pos {
    affine.for %e = 0 to %n {
      affine.for %i = 0 to %n {
        %v = affine.load %x[%i + %e * symbol(%w)] : memref<?xf64>
        affine.store %v, %y[%i + (%e * 2) * symbol(%w)] : memref<?xf64>
        affine.store %v, %y[%i + (%e * 2 + 1) * symbol(%w)] : memref<?xf64>
      }
    }
  }
  return
}

// CHECK:  func.func @index_forms(%arg0: i32, %arg1: memref<?xf64>, %arg2: memref<?xf64>) {
// CHECK-NEXT:   %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:   %0 = arith.cmpi sgt, %arg0, %c0_i32 : i32
// CHECK-NEXT:   %1 = arith.index_cast %arg0 : i32 to index
// CHECK-NEXT:   %2 = arith.extui %arg0 : i32 to i64
// CHECK-NEXT:   %3 = arith.index_cast %2 : i64 to index
// CHECK-NEXT:   scf.if %0 {
// CHECK-NEXT:     affine.parallel (%arg3, %arg4) = (0, 0) to (symbol(%1), symbol(%1)) {
// CHECK-NEXT:       %4 = affine.load %arg1[%arg4 + %arg3 * symbol(%3)] : memref<?xf64>
// CHECK-NEXT:       affine.store %4, %arg2[%arg4 + (%arg3 * symbol(%3)) * 2] : memref<?xf64>
// CHECK-NEXT:       affine.store %4, %arg2[%arg4 + (%arg3 * 2 + 1) * symbol(%3)] : memref<?xf64>
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }

// NQ = Q1D * Q1D is at least 1 past a check that Q1D is not zero: the loop
// over q < NQ stays in its row of width NQ.
func.func @square(%q1d: i32, %x: memref<?xf64>, %y: memref<?xf64>, %ne: index) {
  %c0_i32 = arith.constant 0 : i32
  %nz = arith.cmpi ne, %q1d, %c0_i32 : i32
  %nq32 = arith.muli %q1d, %q1d overflow<nsw> : i32
  %nq = arith.index_cast %nq32 : i32 to index
  scf.if %nz {
    affine.for %e = 0 to %ne {
      affine.for %q = 0 to %nq {
        %v = affine.load %x[%q + %e * symbol(%nq)] : memref<?xf64>
        affine.store %v, %y[%q + (%e * 2) * symbol(%nq)] : memref<?xf64>
        affine.store %v, %y[%q + (%e * 2 + 1) * symbol(%nq)] : memref<?xf64>
      }
    }
  }
  return
}

// CHECK:  func.func @square(%arg0: i32, %arg1: memref<?xf64>, %arg2: memref<?xf64>, %arg3: index) {
// CHECK-NEXT:   %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:   %0 = arith.cmpi ne, %arg0, %c0_i32 : i32
// CHECK-NEXT:   %1 = arith.muli %arg0, %arg0 overflow<nsw> : i32
// CHECK-NEXT:   %2 = arith.index_cast %1 : i32 to index
// CHECK-NEXT:   scf.if %0 {
// CHECK-NEXT:     affine.parallel (%arg4, %arg5) = (0, 0) to (symbol(%arg3), symbol(%2)) {
// CHECK-NEXT:       %3 = affine.load %arg1[%arg5 + %arg4 * symbol(%2)] : memref<?xf64>
// CHECK-NEXT:       affine.store %3, %arg2[%arg5 + (%arg4 * symbol(%2)) * 2] : memref<?xf64>
// CHECK-NEXT:       affine.store %3, %arg2[%arg5 + (%arg4 * 2 + 1) * symbol(%2)] : memref<?xf64>
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }

// y(c, q, e) = w(q) * x(c, q, e) over the three runtime sizes, each loop
// rotated to run at least once: the flat index c + cd * (q + nq * e) splits
// into (e, q, c), each below its size, and the nest is parallel throughout.
func.func @three_levels(%w: memref<?xf64>, %x: memref<?xf64>, %y: memref<?xf64>, %nq32: i32, %cd32: i32, %ne32: i32) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i64 = arith.constant 1 : i64
  %nq = arith.index_cast %nq32 : i32 to index
  %cd = arith.index_cast %cd32 : i32 to index
  %nq64 = arith.extui %nq32 : i32 to i64
  %bq64 = arith.maxsi %nq64, %c1_i64 : i64
  %bq = arith.index_cast %bq64 : i64 to index
  %cd64 = arith.extui %cd32 : i32 to i64
  %bc64 = arith.maxsi %cd64, %c1_i64 : i64
  %bc = arith.index_cast %bc64 : i64 to index
  %ne64 = arith.extui %ne32 : i32 to i64
  %be64 = arith.maxsi %ne64, %c1_i64 : i64
  %be = arith.index_cast %be64 : i64 to index
  %pe = arith.cmpi sgt, %ne32, %c0_i32 : i32
  scf.if %pe {
    %pq = arith.cmpi sgt, %nq32, %c0_i32 : i32
    %pc = arith.cmpi sgt, %cd32, %c0_i32 : i32
    %p = arith.andi %pq, %pc : i1
    scf.if %p {
      affine.for %e = 0 to %be {
        affine.for %q = 0 to %bq {
          affine.parallel (%c) = (0) to (symbol(%bc)) {
            %wv = affine.load %w[%q] : memref<?xf64>
            %xv = affine.load %x[%c + (%q + %e * symbol(%nq)) * symbol(%cd)] : memref<?xf64>
            %m = arith.mulf %wv, %xv : f64
            affine.store %m, %y[%c + (%q + %e * symbol(%nq)) * symbol(%cd)] : memref<?xf64>
          }
        }
      }
    }
  }
  return
}

// CHECK:  func.func @three_levels(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: memref<?xf64>, %arg3: i32, %arg4: i32, %arg5: i32) {
// CHECK-NEXT:   %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:   %c1_i64 = arith.constant 1 : i64
// CHECK-NEXT:   %0 = arith.index_cast %arg3 : i32 to index
// CHECK-NEXT:   %1 = arith.index_cast %arg4 : i32 to index
// CHECK-NEXT:   %2 = arith.extui %arg3 : i32 to i64
// CHECK-NEXT:   %3 = arith.maxsi %2, %c1_i64 : i64
// CHECK-NEXT:   %4 = arith.index_cast %3 : i64 to index
// CHECK-NEXT:   %5 = arith.extui %arg4 : i32 to i64
// CHECK-NEXT:   %6 = arith.maxsi %5, %c1_i64 : i64
// CHECK-NEXT:   %7 = arith.index_cast %6 : i64 to index
// CHECK-NEXT:   %8 = arith.extui %arg5 : i32 to i64
// CHECK-NEXT:   %9 = arith.maxsi %8, %c1_i64 : i64
// CHECK-NEXT:   %10 = arith.index_cast %9 : i64 to index
// CHECK-NEXT:   %11 = arith.cmpi sgt, %arg5, %c0_i32 : i32
// CHECK-NEXT:   scf.if %11 {
// CHECK-NEXT:     %12 = arith.cmpi sgt, %arg3, %c0_i32 : i32
// CHECK-NEXT:     %13 = arith.cmpi sgt, %arg4, %c0_i32 : i32
// CHECK-NEXT:     %14 = arith.andi %12, %13 : i1
// CHECK-NEXT:     scf.if %14 {
// CHECK-NEXT:       affine.parallel (%arg6, %arg7, %arg8) = (0, 0, 0) to (symbol(%10), symbol(%4), symbol(%7)) {
// CHECK-NEXT:         %15 = affine.load %arg0[%arg7] : memref<?xf64>
// CHECK-NEXT:         %16 = affine.load %arg1[%arg8 + (%arg7 + %arg6 * symbol(%0)) * symbol(%1)] : memref<?xf64>
// CHECK-NEXT:         %17 = arith.mulf %15, %16 : f64
// CHECK-NEXT:         affine.store %17, %arg2[%arg8 + (%arg7 + %arg6 * symbol(%0)) * symbol(%1)] : memref<?xf64>
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }

// A layout whose row count a flag picks, y(q, k, e) at q + nq * (k + sel * e)
// with sel 3 or 4: rows k = 0, 1, 2 of block e are below sel, and the blocks
// nq * sel apart.
func.func @chosen_rows(%x: memref<?xf64>, %y: memref<?xf64>, %q1d: i32, %cd: i32, %ne: index) {
  %c0_i32 = arith.constant 0 : i32
  %c4_i32 = arith.constant 4 : i32
  %c1_i64 = arith.constant 1 : i64
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %sym = arith.cmpi ne, %cd, %c4_i32 : i32
  %sel = arith.select %sym, %c3, %c4 : index
  %nq32 = arith.muli %q1d, %q1d overflow<nsw> : i32
  %nq = arith.index_cast %nq32 : i32 to index
  %nq64 = arith.extui %nq32 : i32 to i64
  %bq64 = arith.maxsi %nq64, %c1_i64 : i64
  %bq = arith.index_cast %bq64 : i64 to index
  %nz = arith.cmpi ne, %q1d, %c0_i32 : i32
  scf.if %nz {
    affine.for %e = 0 to %ne {
      affine.for %q = 0 to %bq {
        %v = affine.load %x[%q + %e * symbol(%nq)] : memref<?xf64>
        affine.store %v, %y[%q + (%e * symbol(%sel)) * symbol(%nq)] : memref<?xf64>
        affine.store %v, %y[%q + (%e * symbol(%sel) + 1) * symbol(%nq)] : memref<?xf64>
        affine.store %v, %y[%q + (%e * symbol(%sel) + 2) * symbol(%nq)] : memref<?xf64>
      }
    }
  }
  return
}

// CHECK:  func.func @chosen_rows(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: i32, %arg3: i32, %arg4: index) {
// CHECK-NEXT:   %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:   %c4_i32 = arith.constant 4 : i32
// CHECK-NEXT:   %c1_i64 = arith.constant 1 : i64
// CHECK-NEXT:   %c3 = arith.constant 3 : index
// CHECK-NEXT:   %c4 = arith.constant 4 : index
// CHECK-NEXT:   %0 = arith.cmpi ne, %arg3, %c4_i32 : i32
// CHECK-NEXT:   %1 = arith.select %0, %c3, %c4 : index
// CHECK-NEXT:   %2 = arith.muli %arg2, %arg2 overflow<nsw> : i32
// CHECK-NEXT:   %3 = arith.index_cast %2 : i32 to index
// CHECK-NEXT:   %4 = arith.extui %2 : i32 to i64
// CHECK-NEXT:   %5 = arith.maxsi %4, %c1_i64 : i64
// CHECK-NEXT:   %6 = arith.index_cast %5 : i64 to index
// CHECK-NEXT:   %7 = arith.cmpi ne, %arg2, %c0_i32 : i32
// CHECK-NEXT:   scf.if %7 {
// CHECK-NEXT:     affine.parallel (%arg5, %arg6) = (0, 0) to (symbol(%arg4), symbol(%6)) {
// CHECK-NEXT:       %8 = affine.load %arg0[%arg6 + %arg5 * symbol(%3)] : memref<?xf64>
// CHECK-NEXT:       affine.store %8, %arg1[%arg6 + (%arg5 * symbol(%1)) * symbol(%3)] : memref<?xf64>
// CHECK-NEXT:       affine.store %8, %arg1[%arg6 + (%arg5 * symbol(%1) + 1) * symbol(%3)] : memref<?xf64>
// CHECK-NEXT:       affine.store %8, %arg1[%arg6 + (%arg5 * symbol(%1) + 2) * symbol(%3)] : memref<?xf64>
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }

// A transposed interpolator's accumulation into y at
// qx + e * s + (dy + a * D) * Q, inside the element loop e, with s the
// element's stride, a size the index of no loop variable of the nest: the
// term e * s is one offset for every access of y in an iteration of e, so
// it is set aside and the rest splits into (a, dy, qx), which makes the
// loops a and dy parallel.
func.func @element_offset(%y: memref<?xf64>, %G: memref<?xf64>, %Q: index, %D: index, %ne: index, %s: index) {
  affine.parallel (%e) = (0) to (symbol(%ne)) {
    affine.for %a = 0 to %D {
      affine.for %dy = 0 to %D {
        affine.parallel (%qx) = (0) to (symbol(%Q)) {
          %v = affine.load %y[%qx + %e * symbol(%s) + (%dy + %a * symbol(%D)) * symbol(%Q)] : memref<?xf64>
          %g = affine.load %G[%qx + %dy * symbol(%Q)] : memref<?xf64>
          %r = arith.addf %v, %g : f64
          affine.store %r, %y[%qx + %e * symbol(%s) + (%dy + %a * symbol(%D)) * symbol(%Q)] : memref<?xf64>
        }
      }
    }
  }
  return
}

// The same, where one access of y lacks the offset: the offsets do not
// cancel between the two, and nothing splits.
func.func @element_offset_differs(%y: memref<?xf64>, %G: memref<?xf64>, %Q: index, %D: index, %ne: index, %s: index) {
  affine.parallel (%e) = (0) to (symbol(%ne)) {
    affine.for %a = 0 to %D {
      affine.for %dy = 0 to %D {
        affine.parallel (%qx) = (0) to (symbol(%Q)) {
          %v = affine.load %y[%qx + (%dy + %a * symbol(%D)) * symbol(%Q)] : memref<?xf64>
          %g = affine.load %G[%qx + %dy * symbol(%Q)] : memref<?xf64>
          %r = arith.addf %v, %g : f64
          affine.store %r, %y[%qx + %e * symbol(%s) + (%dy + %a * symbol(%D)) * symbol(%Q)] : memref<?xf64>
        }
      }
    }
  }
  return
}

// CHECK:  func.func @element_offset(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: index, %arg3: index, %arg4: index, %arg5: index) {
// CHECK-NEXT:   affine.parallel (%arg6, %arg7, %arg8, %arg9) = (0, 0, 0, 0) to (symbol(%arg4), symbol(%arg3), symbol(%arg3), symbol(%arg2)) {
// CHECK-NEXT:     %0 = affine.load %arg0[%arg9 + %arg6 * symbol(%arg5) + (%arg8 + %arg7 * symbol(%arg3)) * symbol(%arg2)] : memref<?xf64>
// CHECK-NEXT:     %1 = affine.load %arg1[%arg9 + %arg8 * symbol(%arg2)] : memref<?xf64>
// CHECK-NEXT:     %2 = arith.addf %0, %1 : f64
// CHECK-NEXT:     affine.store %2, %arg0[%arg9 + %arg6 * symbol(%arg5) + (%arg8 + %arg7 * symbol(%arg3)) * symbol(%arg2)] : memref<?xf64>
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }

// CHECK:  func.func @element_offset_differs(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: index, %arg3: index, %arg4: index, %arg5: index) {
// CHECK-NEXT:   affine.parallel (%arg6) = (0) to (symbol(%arg4)) {
// CHECK-NEXT:     affine.for %arg7 = 0 to %arg3 {
// CHECK-NEXT:       affine.for %arg8 = 0 to %arg3 {
// CHECK-NEXT:         affine.parallel (%arg9) = (0) to (symbol(%arg2)) {
// CHECK-NEXT:           %0 = affine.load %arg0[%arg9 + (%arg8 + %arg7 * symbol(%arg3)) * symbol(%arg2)] : memref<?xf64>
// CHECK-NEXT:           %1 = affine.load %arg1[%arg9 + %arg8 * symbol(%arg2)] : memref<?xf64>
// CHECK-NEXT:           %2 = arith.addf %0, %1 : f64
// CHECK-NEXT:           affine.store %2, %arg0[%arg9 + %arg6 * symbol(%arg5) + (%arg8 + %arg7 * symbol(%arg3)) * symbol(%arg2)] : memref<?xf64>
// CHECK-NEXT:         }
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }

// PAHdivMassSetup3D's rows: the diagonal block at row i + 3 + s28 (s28 the
// rows above it, 2 or 3 by the symmetry flag), zero-filled by one nest and
// written by another, both inside the q and e loops. The term s28 * nq is
// no stride of its own (nothing is indexed at s28 * nq): it is the level of
// stride nq at row i + 3 + s28. And the element term e * coeff * nq is one
// placeholder in both accesses although their maps number the symbols
// differently, so the q loop sees the two nests apart.
func.func @symbolic_row(%D: memref<?xf64>, %J: memref<?xf64>, %cd: i32, %nq: index, %ne: index) {
  %c9_i32 = arith.constant 9 : i32
  %c0_i32 = arith.constant 0 : i32
  %c3_i32 = arith.constant 3 : i32
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c6 = arith.constant 6 : index
  %c9 = arith.constant 9 : index
  %cst = arith.constant 0.0 : f64
  %sym = arith.cmpi ne, %cd, %c9_i32 : i32
  %coeff = arith.select %sym, %c6, %c9 : index
  %s25 = arith.extui %sym : i1 to i32
  %s26 = arith.subi %c3_i32, %s25 : i32
  %s27 = arith.maxsi %s26, %c0_i32 : i32
  %s28 = arith.index_cast %s27 : i32 to index
  %s29 = arith.index_cast %s25 : i32 to index
  %s79 = arith.subi %c3, %s29 : index
  %s80 = arith.select %sym, %c3, %c1 : index
  %s81 = arith.subi %c4, %s80 : index
  affine.parallel (%e) = (0) to (symbol(%ne)) {
    affine.for %q = 0 to %nq {
      affine.parallel (%i) = (0) to (symbol(%s79)) {
        affine.store %cst, %D[%q + (%i + 3 + %e * symbol(%coeff)) * symbol(%nq)] : memref<?xf64>
      }
      affine.for %i = 0 to %s81 {
        affine.store %cst, %D[%q + (%i + symbol(%s28) + 3 + %e * symbol(%coeff)) * symbol(%nq)] : memref<?xf64>
        %j = affine.load %J[%q + (%i + %e * 3) * symbol(%nq)] : memref<?xf64>
        %old = affine.load %D[%q + (%i + symbol(%s28) + 3 + %e * symbol(%coeff)) * symbol(%nq)] : memref<?xf64>
        %new = arith.addf %old, %j : f64
        affine.store %new, %D[%q + (%i + symbol(%s28) + 3 + %e * symbol(%coeff)) * symbol(%nq)] : memref<?xf64>
      }
    }
  }
  return
}

// CHECK:  func.func @symbolic_row(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: i32, %arg3: index, %arg4: index) {
// CHECK-NEXT:   %c9_i32 = arith.constant 9 : i32
// CHECK-NEXT:   %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:   %c3_i32 = arith.constant 3 : i32
// CHECK-NEXT:   %c1 = arith.constant 1 : index
// CHECK-NEXT:   %c3 = arith.constant 3 : index
// CHECK-NEXT:   %c4 = arith.constant 4 : index
// CHECK-NEXT:   %c6 = arith.constant 6 : index
// CHECK-NEXT:   %c9 = arith.constant 9 : index
// CHECK-NEXT:   %cst = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:   %0 = arith.cmpi ne, %arg2, %c9_i32 : i32
// CHECK-NEXT:   %1 = arith.select %0, %c6, %c9 : index
// CHECK-NEXT:   %2 = arith.extui %0 : i1 to i32
// CHECK-NEXT:   %3 = arith.subi %c3_i32, %2 : i32
// CHECK-NEXT:   %4 = arith.maxsi %3, %c0_i32 : i32
// CHECK-NEXT:   %5 = arith.index_cast %4 : i32 to index
// CHECK-NEXT:   %6 = arith.index_cast %2 : i32 to index
// CHECK-NEXT:   %7 = arith.subi %c3, %6 : index
// CHECK-NEXT:   %8 = arith.select %0, %c3, %c1 : index
// CHECK-NEXT:   %9 = arith.subi %c4, %8 : index
// CHECK-NEXT:   affine.parallel (%arg5, %arg6) = (0, 0) to (symbol(%arg4), symbol(%arg3)) {
// CHECK-NEXT:     affine.parallel (%arg7) = (0) to (symbol(%7)) {
// CHECK-NEXT:       affine.store %cst, %arg0[%arg6 + (%arg7 + %arg5 * symbol(%1) + 3) * symbol(%arg3)] : memref<?xf64>
// CHECK-NEXT:     }
// CHECK-NEXT:     affine.parallel (%arg7) = (0) to (symbol(%9)) {
// CHECK-NEXT:       affine.store %cst, %arg0[%arg6 + (%arg7 + symbol(%5) + 3 + %arg5 * symbol(%1)) * symbol(%arg3)] : memref<?xf64>
// CHECK-NEXT:       %10 = affine.load %arg1[%arg6 + (%arg7 + %arg5 * 3) * symbol(%arg3)] : memref<?xf64>
// CHECK-NEXT:       %11 = affine.load %arg0[%arg6 + (%arg7 + symbol(%5) + 3 + %arg5 * symbol(%1)) * symbol(%arg3)] : memref<?xf64>
// CHECK-NEXT:       %12 = arith.addf %11, %10 : f64
// CHECK-NEXT:       affine.store %12, %arg0[%arg6 + (%arg7 + symbol(%5) + 3 + %arg5 * symbol(%1)) * symbol(%arg3)] : memref<?xf64>
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }

// The sizes a kernel reads are clamps of one value: the loop bound
// max(min(d1d, 14), 1) of a rotated loop over the kernel's clamped size,
// the stride min(d1d, 24) of the host's clamp. Past the check d1d > 0 the
// bound is at most the stride, as the piecewise-linear functions of d1d
// they are, so the rows stay apart.
func.func @clamped(%D: memref<?xf64>, %d1d: i32, %ne: index) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c14_i32 = arith.constant 14 : i32
  %c24_i32 = arith.constant 24 : i32
  %cst = arith.constant 1.0 : f64
  %m14 = arith.minsi %d1d, %c14_i32 : i32
  %b = arith.maxsi %m14, %c1_i32 : i32
  %bound = arith.index_cast %b : i32 to index
  %m24 = arith.minsi %d1d, %c24_i32 : i32
  %stride = arith.index_cast %m24 : i32 to index
  %x = arith.index_cast %d1d : i32 to index
  %pos = arith.cmpi sgt, %d1d, %c0_i32 : i32
  scf.if %pos {
    affine.parallel (%e) = (0) to (symbol(%ne)) {
      affine.for %i = 0 to %bound {
        affine.for %j = 0 to %bound {
          %v = affine.load %D[%j + (%i + %e * 3 * symbol(%stride)) * symbol(%stride)] : memref<?xf64>
          %w = arith.addf %v, %cst : f64
          affine.store %w, %D[%j + (%i + %e * 3 * symbol(%stride)) * symbol(%stride)] : memref<?xf64>
        }
      }
    }
  }
  return
}

// Without the check, d1d <= 0 leaves the bound 1 above the stride, and
// the loops stay.
func.func @clamped_unchecked(%D: memref<?xf64>, %d1d: i32, %ne: index) {
  %c1_i32 = arith.constant 1 : i32
  %c14_i32 = arith.constant 14 : i32
  %c24_i32 = arith.constant 24 : i32
  %cst = arith.constant 1.0 : f64
  %m14 = arith.minsi %d1d, %c14_i32 : i32
  %b = arith.maxsi %m14, %c1_i32 : i32
  %bound = arith.index_cast %b : i32 to index
  %m24 = arith.minsi %d1d, %c24_i32 : i32
  %stride = arith.index_cast %m24 : i32 to index
  %x = arith.index_cast %d1d : i32 to index
  affine.parallel (%e) = (0) to (symbol(%ne)) {
    affine.for %i = 0 to %bound {
      affine.for %j = 0 to %bound {
        %v = affine.load %D[%j + (%i + %e * 3 * symbol(%stride)) * symbol(%stride)] : memref<?xf64>
        %w = arith.addf %v, %cst : f64
        affine.store %w, %D[%j + (%i + %e * 3 * symbol(%stride)) * symbol(%stride)] : memref<?xf64>
      }
    }
  }
  return
}

// CHECK:  func.func @clamped(%arg0: memref<?xf64>, %arg1: i32, %arg2: index) {
// CHECK-NEXT:   %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:   %c1_i32 = arith.constant 1 : i32
// CHECK-NEXT:   %c14_i32 = arith.constant 14 : i32
// CHECK-NEXT:   %c24_i32 = arith.constant 24 : i32
// CHECK-NEXT:   %cst = arith.constant 1.000000e+00 : f64
// CHECK-NEXT:   %0 = arith.minsi %arg1, %c14_i32 : i32
// CHECK-NEXT:   %1 = arith.maxsi %0, %c1_i32 : i32
// CHECK-NEXT:   %2 = arith.index_cast %1 : i32 to index
// CHECK-NEXT:   %3 = arith.minsi %arg1, %c24_i32 : i32
// CHECK-NEXT:   %4 = arith.index_cast %3 : i32 to index
// CHECK-NEXT:   %5 = arith.cmpi sgt, %arg1, %c0_i32 : i32
// CHECK-NEXT:   scf.if %5 {
// CHECK-NEXT:     affine.parallel (%arg3, %arg4, %arg5) = (0, 0, 0) to (symbol(%arg2), symbol(%2), symbol(%2)) {
// CHECK-NEXT:       %6 = affine.load %arg0[%arg5 + (%arg4 + (%arg3 * symbol(%4)) * 3) * symbol(%4)] : memref<?xf64>
// CHECK-NEXT:       %7 = arith.addf %6, %cst : f64
// CHECK-NEXT:       affine.store %7, %arg0[%arg5 + (%arg4 + (%arg3 * symbol(%4)) * 3) * symbol(%4)] : memref<?xf64>
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }

// CHECK:  func.func @clamped_unchecked(%arg0: memref<?xf64>, %arg1: i32, %arg2: index) {
// CHECK-NEXT:   %c1_i32 = arith.constant 1 : i32
// CHECK-NEXT:   %c14_i32 = arith.constant 14 : i32
// CHECK-NEXT:   %c24_i32 = arith.constant 24 : i32
// CHECK-NEXT:   %cst = arith.constant 1.000000e+00 : f64
// CHECK-NEXT:   %0 = arith.minsi %arg1, %c14_i32 : i32
// CHECK-NEXT:   %1 = arith.maxsi %0, %c1_i32 : i32
// CHECK-NEXT:   %2 = arith.index_cast %1 : i32 to index
// CHECK-NEXT:   %3 = arith.minsi %arg1, %c24_i32 : i32
// CHECK-NEXT:   %4 = arith.index_cast %3 : i32 to index
// CHECK-NEXT:   affine.parallel (%arg3) = (0) to (symbol(%arg2)) {
// CHECK-NEXT:     affine.for %arg4 = 0 to %2 {
// CHECK-NEXT:       affine.parallel (%arg5) = (0) to (symbol(%2)) {
// CHECK-NEXT:         %5 = affine.load %arg0[%arg5 + (%arg4 + (%arg3 * symbol(%4)) * 3) * symbol(%4)] : memref<?xf64>
// CHECK-NEXT:         %6 = arith.addf %5, %cst : f64
// CHECK-NEXT:         affine.store %6, %arg0[%arg5 + (%arg4 + (%arg3 * symbol(%4)) * 3) * symbol(%4)] : memref<?xf64>
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }

// The loop bound and the stride read one integer through different casts,
// index_cast(extui(n)) and index_cast(n): they are one value where n is
// non-negative, which the guard n >= 1 on the way gives, and the component
// loop is parallel.
func.func @cast_forms(%D: memref<?xf64>, %n32: i32, %ne: index) {
  %cst = arith.constant 1.0 : f64
  %s = arith.index_cast %n32 : i32 to index
  %z = arith.extui %n32 : i32 to i64
  %b = arith.index_cast %z : i64 to index
  affine.parallel (%e) = (0) to (symbol(%ne)) {
    affine.if affine_set<()[s0] : (s0 - 1 >= 0)>()[%s] {
    affine.for %c = 0 to 3 {
      affine.for %i = 0 to %b {
        %v = affine.load %D[%i + (%c + %e * 3) * symbol(%s)] : memref<?xf64>
        %w = arith.addf %v, %cst : f64
        affine.store %w, %D[%i + (%c + %e * 3) * symbol(%s)] : memref<?xf64>
      }
    }
    }
  }
  return
}

// CHECK:  func.func @cast_forms(%arg0: memref<?xf64>, %arg1: i32, %arg2: index) {
// CHECK-NEXT:   %cst = arith.constant 1.000000e+00 : f64
// CHECK-NEXT:   %0 = arith.index_cast %arg1 : i32 to index
// CHECK-NEXT:   %1 = arith.extui %arg1 : i32 to i64
// CHECK-NEXT:   %2 = arith.index_cast %1 : i64 to index
// CHECK-NEXT:   affine.parallel (%arg3) = (0) to (symbol(%arg2)) {
// CHECK-NEXT:     affine.if #set()[%0] {
// CHECK-NEXT:       affine.parallel (%arg4, %arg5) = (0, 0) to (3, symbol(%2)) {
// CHECK-NEXT:         %3 = affine.load %arg0[%arg5 + (%arg4 + %arg3 * 3) * symbol(%0)] : memref<?xf64>
// CHECK-NEXT:         %4 = arith.addf %3, %cst : f64
// CHECK-NEXT:         affine.store %4, %arg0[%arg5 + (%arg4 + %arg3 * 3) * symbol(%0)] : memref<?xf64>
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }
