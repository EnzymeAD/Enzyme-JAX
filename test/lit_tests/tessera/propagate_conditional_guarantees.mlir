// RUN: enzymexlamlir-opt %s -tessera-propagate-properties -split-input-file | FileCheck %s

// A conditional guarantee establishes its property only at a call where the
// facts there prove the condition. @gram computes C = op(A)^T op(B); when A
// and B are one full-rank matrix and trans is 1, C is SPD.

module {
  tessera.define private @lib.solve(!llvm.ptr, !llvm.ptr) -> i32 attributes {argModes = [unit, unit], pure = false}
  llvm.func @mat_create(!llvm.ptr) -> i32
  llvm.func @mat_scale(!llvm.ptr, f64) -> i32
  llvm.func @gram(!llvm.ptr, !llvm.ptr, !llvm.ptr, i32) attributes {tessera.conditional_establishes = [{arg = 2 : i64, condition = "FullRank(A) && A == B && trans == Trans", constants = {Trans = 1 : i64}, names = {A = 0 : i64, B = 1 : i64, trans = 3 : i64}, property = "SPD"}]}
  llvm.func @__tessera_fact_0(!llvm.ptr {tessera.establishes = ["FullRank"]}) attributes {tessera.fact_marker}

  // CHECK-LABEL: llvm.func @proven
  // CHECK: tessera.call @lib.solve({{.*}}) <arg_attrs = [{tessera.property = ["SPD"]}, {}]>
  llvm.func @proven(%b: !llvm.ptr) -> i32 {
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %x = llvm.alloca %c1 x !llvm.ptr : (i32) -> !llvm.ptr
    %g = llvm.alloca %c1 x !llvm.ptr : (i32) -> !llvm.ptr
    %e0 = llvm.call @mat_create(%x) : (!llvm.ptr) -> i32
    %e1 = llvm.call @mat_create(%g) : (!llvm.ptr) -> i32
    llvm.call @__tessera_fact_0(%x) : (!llvm.ptr) -> ()
    %x1 = llvm.load %x : !llvm.ptr -> !llvm.ptr
    %x2 = llvm.load %x : !llvm.ptr -> !llvm.ptr
    %g1 = llvm.load %g : !llvm.ptr -> !llvm.ptr
    llvm.call @gram(%x1, %x2, %g1, %c1) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32) -> ()
    %g2 = llvm.load %g : !llvm.ptr -> !llvm.ptr
    %r = tessera.call @lib.solve(%g2, %b) : (!llvm.ptr, !llvm.ptr) -> i32
    llvm.return %r : i32
  }

  // No fact that X has full rank.
  // CHECK-LABEL: llvm.func @no_full_rank
  // CHECK: tessera.call @lib.solve
  // CHECK-NOT: tessera.property
  // CHECK: llvm.return
  llvm.func @no_full_rank(%b: !llvm.ptr) -> i32 {
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %x = llvm.alloca %c1 x !llvm.ptr : (i32) -> !llvm.ptr
    %g = llvm.alloca %c1 x !llvm.ptr : (i32) -> !llvm.ptr
    %e0 = llvm.call @mat_create(%x) : (!llvm.ptr) -> i32
    %e1 = llvm.call @mat_create(%g) : (!llvm.ptr) -> i32
    %x1 = llvm.load %x : !llvm.ptr -> !llvm.ptr
    %x2 = llvm.load %x : !llvm.ptr -> !llvm.ptr
    %g1 = llvm.load %g : !llvm.ptr -> !llvm.ptr
    llvm.call @gram(%x1, %x2, %g1, %c1) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32) -> ()
    %g2 = llvm.load %g : !llvm.ptr -> !llvm.ptr
    %r = tessera.call @lib.solve(%g2, %b) : (!llvm.ptr, !llvm.ptr) -> i32
    llvm.return %r : i32
  }

  // X^T Y of two matrices is not SPD.
  // CHECK-LABEL: llvm.func @two_matrices
  // CHECK: tessera.call @lib.solve
  // CHECK-NOT: tessera.property
  // CHECK: llvm.return
  llvm.func @two_matrices(%b: !llvm.ptr) -> i32 {
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %x = llvm.alloca %c1 x !llvm.ptr : (i32) -> !llvm.ptr
    %y = llvm.alloca %c1 x !llvm.ptr : (i32) -> !llvm.ptr
    %g = llvm.alloca %c1 x !llvm.ptr : (i32) -> !llvm.ptr
    %e0 = llvm.call @mat_create(%x) : (!llvm.ptr) -> i32
    %e1 = llvm.call @mat_create(%y) : (!llvm.ptr) -> i32
    %e2 = llvm.call @mat_create(%g) : (!llvm.ptr) -> i32
    llvm.call @__tessera_fact_0(%x) : (!llvm.ptr) -> ()
    %x1 = llvm.load %x : !llvm.ptr -> !llvm.ptr
    %y1 = llvm.load %y : !llvm.ptr -> !llvm.ptr
    %g1 = llvm.load %g : !llvm.ptr -> !llvm.ptr
    llvm.call @gram(%x1, %y1, %g1, %c1) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32) -> ()
    %g2 = llvm.load %g : !llvm.ptr -> !llvm.ptr
    %r = tessera.call @lib.solve(%g2, %b) : (!llvm.ptr, !llvm.ptr) -> i32
    llvm.return %r : i32
  }

  // Another matrix is stored into X between the two loads: they may differ.
  // CHECK-LABEL: llvm.func @reassigned_between_loads
  // CHECK: tessera.call @lib.solve
  // CHECK-NOT: tessera.property
  // CHECK: llvm.return
  llvm.func @reassigned_between_loads(%b: !llvm.ptr, %other: !llvm.ptr) -> i32 {
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %x = llvm.alloca %c1 x !llvm.ptr : (i32) -> !llvm.ptr
    %g = llvm.alloca %c1 x !llvm.ptr : (i32) -> !llvm.ptr
    %e0 = llvm.call @mat_create(%x) : (!llvm.ptr) -> i32
    %e1 = llvm.call @mat_create(%g) : (!llvm.ptr) -> i32
    llvm.call @__tessera_fact_0(%x) : (!llvm.ptr) -> ()
    %x1 = llvm.load %x : !llvm.ptr -> !llvm.ptr
    llvm.store %other, %x : !llvm.ptr, !llvm.ptr
    %x2 = llvm.load %x : !llvm.ptr -> !llvm.ptr
    %g1 = llvm.load %g : !llvm.ptr -> !llvm.ptr
    llvm.call @gram(%x1, %x2, %g1, %c1) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32) -> ()
    %g2 = llvm.load %g : !llvm.ptr -> !llvm.ptr
    %r = tessera.call @lib.solve(%g2, %b) : (!llvm.ptr, !llvm.ptr) -> i32
    llvm.return %r : i32
  }

  // The flag makes it A B^T rather than A^T B.
  // CHECK-LABEL: llvm.func @wrong_flag
  // CHECK: tessera.call @lib.solve
  // CHECK-NOT: tessera.property
  // CHECK: llvm.return
  llvm.func @wrong_flag(%b: !llvm.ptr) -> i32 {
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %x = llvm.alloca %c1 x !llvm.ptr : (i32) -> !llvm.ptr
    %g = llvm.alloca %c1 x !llvm.ptr : (i32) -> !llvm.ptr
    %e0 = llvm.call @mat_create(%x) : (!llvm.ptr) -> i32
    %e1 = llvm.call @mat_create(%g) : (!llvm.ptr) -> i32
    llvm.call @__tessera_fact_0(%x) : (!llvm.ptr) -> ()
    %x1 = llvm.load %x : !llvm.ptr -> !llvm.ptr
    %x2 = llvm.load %x : !llvm.ptr -> !llvm.ptr
    %g1 = llvm.load %g : !llvm.ptr -> !llvm.ptr
    llvm.call @gram(%x1, %x2, %g1, %c0) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32) -> ()
    %g2 = llvm.load %g : !llvm.ptr -> !llvm.ptr
    %r = tessera.call @lib.solve(%g2, %b) : (!llvm.ptr, !llvm.ptr) -> i32
    llvm.return %r : i32
  }

  // The product is SPD, then scaled by an unknown factor.
  // CHECK-LABEL: llvm.func @changed_after
  // CHECK: tessera.call @lib.solve
  // CHECK-NOT: tessera.property
  // CHECK: llvm.return
  llvm.func @changed_after(%b: !llvm.ptr, %s: f64) -> i32 {
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %x = llvm.alloca %c1 x !llvm.ptr : (i32) -> !llvm.ptr
    %g = llvm.alloca %c1 x !llvm.ptr : (i32) -> !llvm.ptr
    %e0 = llvm.call @mat_create(%x) : (!llvm.ptr) -> i32
    %e1 = llvm.call @mat_create(%g) : (!llvm.ptr) -> i32
    llvm.call @__tessera_fact_0(%x) : (!llvm.ptr) -> ()
    %x1 = llvm.load %x : !llvm.ptr -> !llvm.ptr
    %x2 = llvm.load %x : !llvm.ptr -> !llvm.ptr
    %g1 = llvm.load %g : !llvm.ptr -> !llvm.ptr
    llvm.call @gram(%x1, %x2, %g1, %c1) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32) -> ()
    %g2 = llvm.load %g : !llvm.ptr -> !llvm.ptr
    %e2 = llvm.call @mat_scale(%g2, %s) : (!llvm.ptr, f64) -> i32
    %g3 = llvm.load %g : !llvm.ptr -> !llvm.ptr
    %r = tessera.call @lib.solve(%g3, %b) : (!llvm.ptr, !llvm.ptr) -> i32
    llvm.return %r : i32
  }
}

// -----

// The same on a tessera op, which keeps the function's attributes on its
// definition, with a condition on a floating-point argument.
module {
  tessera.define private @lib.solve(!llvm.ptr, !llvm.ptr) -> i32 attributes {argModes = [unit, unit], pure = false}
  tessera.define private @blas.gram(!llvm.ptr, !llvm.ptr, !llvm.ptr, f64) attributes {argModes = [unit, unit, unit, unit], pure = false, tessera.conditional_establishes = [{arg = 2 : i64, condition = "FullRank(A) && A == B && beta == 0.0", constants = {}, names = {A = 0 : i64, B = 1 : i64, beta = 3 : i64}, property = "SPD"}]}
  llvm.func @mat_create(!llvm.ptr) -> i32
  llvm.func @__tessera_fact_0(!llvm.ptr {tessera.establishes = ["FullRank"]}) attributes {tessera.fact_marker}

  // CHECK-LABEL: llvm.func @op_proven
  // CHECK: tessera.call @lib.solve({{.*}}) <arg_attrs = [{tessera.property = ["SPD"]}, {}]>
  llvm.func @op_proven(%b: !llvm.ptr) -> i32 {
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %zero = llvm.mlir.constant(0.0 : f64) : f64
    %x = llvm.alloca %c1 x !llvm.ptr : (i32) -> !llvm.ptr
    %g = llvm.alloca %c1 x !llvm.ptr : (i32) -> !llvm.ptr
    %e0 = llvm.call @mat_create(%x) : (!llvm.ptr) -> i32
    %e1 = llvm.call @mat_create(%g) : (!llvm.ptr) -> i32
    llvm.call @__tessera_fact_0(%x) : (!llvm.ptr) -> ()
    %x1 = llvm.load %x : !llvm.ptr -> !llvm.ptr
    %g1 = llvm.load %g : !llvm.ptr -> !llvm.ptr
    tessera.call @blas.gram(%x1, %x1, %g1, %zero) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, f64) -> ()
    %g2 = llvm.load %g : !llvm.ptr -> !llvm.ptr
    %r = tessera.call @lib.solve(%g2, %b) : (!llvm.ptr, !llvm.ptr) -> i32
    llvm.return %r : i32
  }

  // beta is 1: C = X^T X + C need not be SPD.
  // CHECK-LABEL: llvm.func @op_nonzero_beta
  // CHECK: tessera.call @lib.solve
  // CHECK-NOT: tessera.property
  // CHECK: llvm.return
  llvm.func @op_nonzero_beta(%b: !llvm.ptr) -> i32 {
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %one = llvm.mlir.constant(1.0 : f64) : f64
    %x = llvm.alloca %c1 x !llvm.ptr : (i32) -> !llvm.ptr
    %g = llvm.alloca %c1 x !llvm.ptr : (i32) -> !llvm.ptr
    %e0 = llvm.call @mat_create(%x) : (!llvm.ptr) -> i32
    %e1 = llvm.call @mat_create(%g) : (!llvm.ptr) -> i32
    llvm.call @__tessera_fact_0(%x) : (!llvm.ptr) -> ()
    %x1 = llvm.load %x : !llvm.ptr -> !llvm.ptr
    %g1 = llvm.load %g : !llvm.ptr -> !llvm.ptr
    tessera.call @blas.gram(%x1, %x1, %g1, %one) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, f64) -> ()
    %g2 = llvm.load %g : !llvm.ptr -> !llvm.ptr
    %r = tessera.call @lib.solve(%g2, %b) : (!llvm.ptr, !llvm.ptr) -> i32
    llvm.return %r : i32
  }
}
