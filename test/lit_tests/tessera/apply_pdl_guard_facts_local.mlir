// RUN: enzymexlamlir-opt %s -parse-optimization-rules -tessera-propagate-properties -tessera-apply-pdl | FileCheck %s

// A property a guard checked holds only in its first region. A rule applied
// to the original call, in the guard's second region, must not read it: here
// diagonal implies triangular_upper, which would rewrite the original call to
// the triangular solve without a check.

module {
  tessera.define private @lib.solve(!llvm.ptr, !llvm.ptr) -> i32 attributes {argModes = [unit, unit], pure = false}
  tessera.define private @lib.dsolve(!llvm.ptr, !llvm.ptr) -> i32 attributes {argModes = [unit, unit], pure = false}
  tessera.define private @lib.tsolve(!llvm.ptr, !llvm.ptr) -> i32 attributes {argModes = [unit, unit], pure = false}

  tessera.optimizations {
    tessera.optimization "if triangular_upper(a), lib.solve(a, b) -> lib.tsolve(a, b)"
    tessera.optimization "if diagonal(a), lib.solve(a, b) -> lib.dsolve(a, b)"
  }

  // CHECK-LABEL: llvm.func @unknown
  // CHECK: tessera.guard
  // CHECK: tessera.call @lib.dsolve
  // CHECK: } else {
  // CHECK-NOT: tessera.call @lib.tsolve
  // CHECK: tessera.guard "triangular_upper(a)"
  // CHECK: tessera.call @lib.tsolve
  // CHECK: } else {
  // CHECK: tessera.call @lib.solve
  llvm.func @unknown(%a: !llvm.ptr {llvm.noalias}, %b: !llvm.ptr) -> i32 {
    %r = tessera.call @lib.solve(%a, %b) : (!llvm.ptr, !llvm.ptr) -> i32
    llvm.return %r : i32
  }
}
