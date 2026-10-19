// RUN: enzymexlamlir-opt %s --canonicalize-scf-for | FileCheck %s

// A descending do-while: the before region reads row j = c-1, the after
// region writes it, and the final evaluation's row is written after the
// loop from the forwarded results. prepareFor negates the negative step, so
// the converted for ascends with the counter riding along as an iter arg,
// and the extra appended iteration -- the one whose after region must not
// run -- is the one at the top: the guard is iv < ub whichever way the
// original loop counted. Guarded sgt-against-ub instead, the after region
// ran for no iteration at all: this loop is CholeskyFactors::LMult, where
// only row zero was ever written, every white-noise sample kept its raw
// entries, and MFEM's white_noise statistics test failed its mean bound.

llvm.func @lmult_rows(%x: !llvm.ptr, %m: i64) {
  %c1 = arith.constant 1 : i64
  %cm1 = arith.constant -1 : i64
  %cst2 = arith.constant 2.0 : f64
  %r:2 = scf.while (%c = %m) : (i64) -> (i64, f64) {
    %j = arith.addi %c, %cm1 : i64
    %p = llvm.getelementptr %x[%j] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %v = llvm.load %p : !llvm.ptr -> f64
    %cont = arith.cmpi sgt, %c, %c1 : i64
    scf.condition(%cont) %j, %v : i64, f64
  } do {
  ^bb0(%j: i64, %v: f64):
    %two = arith.mulf %v, %cst2 : f64
    %q = llvm.getelementptr %x[%j] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %two, %q : f64, !llvm.ptr
    scf.yield %j : i64
  }
  %pf = llvm.getelementptr %x[%r#0] : (!llvm.ptr, i64) -> !llvm.ptr, f64
  llvm.store %r#1, %pf : f64, !llvm.ptr
  llvm.return
}

// The after region runs on every iteration but the extra top one.
// CHECK-LABEL: llvm.func @lmult_rows
// CHECK:         %[[UB0:.+]] = arith.addi %arg1, %c1{{.*}} : i64
// CHECK:         %[[MAX:.+]] = arith.maxsi %[[UB0]], %c2{{.*}} : i64
// CHECK:         %[[UB:.+]] = arith.addi %[[MAX]], %c1{{.*}} : i64
// CHECK:         scf.for %[[IV:.+]] = %c2{{.*}} to %[[UB]] step %c1
// CHECK:           %[[GUARD:.+]] = arith.cmpi slt, %[[IV]], %[[UB0]] : i64
// CHECK:           scf.if %[[GUARD]]
// CHECK:             llvm.store
