// RUN: enzymexlamlir-opt %s -tessera-lower-guards -split-input-file | FileCheck %s

// `invertible` and `positive_definite` are the two predicates whose exact test
// is a factorization -- an LU with pivoting and a Cholesky respectively. Either
// would cost as much as the operation being optimized, so neither is worth
// emitting, and that is true no matter who writes it.
//
// Instead each emits a cheap SUFFICIENT condition in O(n^2): strict diagonal
// dominance, which implies nonsingularity (Levy-Desplanques), and additionally
// a symmetric positive diagonal, which implies positive definiteness. Both are
// sound -- a true answer is never wrong. Both are incomplete: a matrix that
// does qualify may test false and take the general path, costing only the
// check. That is the right way round.

module {
  tessera.define @lib.foo(%a: !llvm.ptr {tessera.layout = {elem = f64, rows = 2 : i64, cols = 2 : i64, row_major = true}}) -> f64 attributes {byRefTypes = [unit], pure = true} {
    %c = llvm.mlir.constant(0.0 : f64) : f64
    tessera.return %c : f64
  }
  tessera.define @lib.fast_foo(%a: !llvm.ptr) -> f64 attributes {byRefTypes = [unit], pure = true} {
    %c = llvm.mlir.constant(0.0 : f64) : f64
    tessera.return %c : f64
  }

  // Row 0: |A[0][0]| > |A[0][1]|. Row 1: |A[1][1]| > |A[1][0]|.
  // CHECK-LABEL: llvm.func @invertible
  llvm.func @invertible(%x: !llvm.ptr) -> f64 {
    // CHECK: %[[D0:.*]] = llvm.intr.fabs(%{{.*}}) : (f64) -> f64
    // CHECK: %[[Z0:.*]] = llvm.mlir.constant(0.000000e+00 : f64) : f64
    // CHECK: %[[O0:.*]] = llvm.intr.fabs(%{{.*}}) : (f64) -> f64
    // CHECK: %[[S0:.*]] = llvm.fadd %[[Z0]], %[[O0]] : f64
    // CHECK: %[[R0:.*]] = llvm.fcmp "ogt" %[[D0]], %[[S0]] : f64
    // CHECK: %[[D1:.*]] = llvm.intr.fabs(%{{.*}}) : (f64) -> f64
    // CHECK: %[[Z1:.*]] = llvm.mlir.constant(0.000000e+00 : f64) : f64
    // CHECK: %[[O1:.*]] = llvm.intr.fabs(%{{.*}}) : (f64) -> f64
    // CHECK: %[[S1:.*]] = llvm.fadd %[[Z1]], %[[O1]] : f64
    // CHECK: %[[R1:.*]] = llvm.fcmp "ogt" %[[D1]], %[[S1]] : f64
    // CHECK: %[[ALL:.*]] = llvm.and %[[R0]], %[[R1]] : i1
    // CHECK: llvm.cond_br %[[ALL]]

    // Nothing is called out to: the whole test is inline arithmetic.
    // CHECK-NOT: llvm.call @{{.*}}factor
    %0 = tessera.guard "invertible(x)" args(%x) {argNames = ["x"]} : (!llvm.ptr) -> f64 {
      %1 = tessera.call @lib.fast_foo(%x) : (!llvm.ptr) -> f64
      tessera.yield %1 : f64
    } else {
      %2 = tessera.call @lib.foo(%x) : (!llvm.ptr) -> f64
      tessera.yield %2 : f64
    }
    llvm.return %0 : f64
  }
}

// -----

module {
  tessera.define @lib.foo(%a: !llvm.ptr {tessera.layout = {elem = f64, rows = 2 : i64, cols = 2 : i64, row_major = true}}) -> f64 attributes {byRefTypes = [unit], pure = true} {
    %c = llvm.mlir.constant(0.0 : f64) : f64
    tessera.return %c : f64
  }
  tessera.define @lib.chol_foo(%a: !llvm.ptr) -> f64 attributes {byRefTypes = [unit], pure = true} {
    %c = llvm.mlir.constant(0.0 : f64) : f64
    tessera.return %c : f64
  }

  // Three conjuncts: symmetry, a positive diagonal, and diagonal dominance.
  // CHECK-LABEL: llvm.func @positive_definite
  llvm.func @positive_definite(%x: !llvm.ptr) -> f64 {
    // Symmetry, for a 2x2, is the single comparison A[0][1] == A[1][0].
    // CHECK: %[[SYM:.*]] = llvm.fcmp "oeq"

    // A positive diagonal.
    // CHECK: %[[PZ:.*]] = llvm.mlir.constant(0.000000e+00 : f64) : f64
    // CHECK: %[[P0:.*]] = llvm.fcmp "ogt" %{{.*}}, %[[PZ]] : f64
    // CHECK: %[[P1:.*]] = llvm.fcmp "ogt" %{{.*}}, %[[PZ]] : f64
    // CHECK: %[[POS:.*]] = llvm.and %[[P0]], %[[P1]] : i1

    // Diagonal dominance.
    // CHECK: llvm.intr.fabs
    // CHECK: %[[DOM:.*]] = llvm.and %{{.*}}, %{{.*}} : i1

    // CHECK: %[[T:.*]] = llvm.and %[[SYM]], %[[POS]] : i1
    // CHECK: %[[ALL:.*]] = llvm.and %[[T]], %[[DOM]] : i1
    // CHECK: llvm.cond_br %[[ALL]]
    %0 = tessera.guard "positive_definite(x)" args(%x) {argNames = ["x"]} : (!llvm.ptr) -> f64 {
      %1 = tessera.call @lib.chol_foo(%x) : (!llvm.ptr) -> f64
      tessera.yield %1 : f64
    } else {
      %2 = tessera.call @lib.foo(%x) : (!llvm.ptr) -> f64
      tessera.yield %2 : f64
    }
    llvm.return %0 : f64
  }
}

// -----

module {
  tessera.define @lib.foo(%a: !llvm.ptr {tessera.layout = {elem = i32, rows = 2 : i64, cols = 2 : i64, row_major = true}}) -> i32 attributes {byRefTypes = [unit], pure = true} {
    %c = llvm.mlir.constant(0 : i32) : i32
    tessera.return %c : i32
  }
  tessera.define @lib.fast_foo(%a: !llvm.ptr) -> i32 attributes {byRefTypes = [unit], pure = true} {
    %c = llvm.mlir.constant(0 : i32) : i32
    tessera.return %c : i32
  }

  // An integer element type has no fabs, so the absolute value is a select.
  // CHECK-LABEL: llvm.func @invertible_integer
  llvm.func @invertible_integer(%x: !llvm.ptr) -> i32 {
    // CHECK: %[[NEG:.*]] = llvm.sub %{{.*}}, %{{.*}} : i32
    // CHECK: %[[LT:.*]] = llvm.icmp "slt" %{{.*}}, %{{.*}} : i32
    // CHECK: %[[ABS:.*]] = llvm.select %[[LT]], %[[NEG]], %{{.*}} : i1, i32
    // CHECK: llvm.icmp "sgt"
    // CHECK: llvm.cond_br
    %0 = tessera.guard "invertible(x)" args(%x) {argNames = ["x"]} : (!llvm.ptr) -> i32 {
      %1 = tessera.call @lib.fast_foo(%x) : (!llvm.ptr) -> i32
      tessera.yield %1 : i32
    } else {
      %2 = tessera.call @lib.foo(%x) : (!llvm.ptr) -> i32
      tessera.yield %2 : i32
    }
    llvm.return %0 : i32
  }
}
