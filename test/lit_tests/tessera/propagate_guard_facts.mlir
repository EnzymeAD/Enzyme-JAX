// RUN: enzymexlamlir-opt %s -parse-optimization-rules -tessera-propagate-properties -tessera-apply-pdl | FileCheck %s

// A guard's first region runs only when its condition holds, so a rule applied
// to a call there may use each property the condition requires, through `&&`,
// without checking it again. Here the first rule's guarded rewrite produces
// lib.g, which the second rule then rewrites in place, with no second guard.
// A property only one side of `||` requires is not known.

module {
  tessera.define private @lib.f(!llvm.ptr) -> f64 attributes {argModes = [unit], pure = false}
  tessera.define private @lib.g(!llvm.ptr) -> f64 attributes {argModes = [unit], pure = false}
  tessera.define private @lib.h(!llvm.ptr) -> f64 attributes {argModes = [unit], pure = false}
  tessera.define private @lib.k(!llvm.ptr) -> f64 attributes {argModes = [unit], pure = false}

  tessera.optimizations {
    tessera.optimization "if symmetric(x), lib.f(x) -> lib.g(x)"
    tessera.optimization "if symmetric(x), lib.g(x) -> lib.h(x)"
    tessera.optimization "if symmetric(x) || diagonal(x), lib.k(x) -> lib.g(x)"
  }

  // CHECK-LABEL: llvm.func @required
  // CHECK: tessera.guard "symmetric(x)"
  // CHECK-NOT: tessera.guard
  // CHECK: tessera.call @lib.h
  // CHECK: } else {
  // CHECK: tessera.call @lib.f
  llvm.func @required(%a: !llvm.ptr {llvm.noalias}) -> f64 {
    %r = tessera.call @lib.f(%a) : (!llvm.ptr) -> f64
    llvm.return %r : f64
  }

  // Either property may be the one that held, so lib.g keeps its own guard.
  // CHECK-LABEL: llvm.func @either
  // CHECK: tessera.guard "symmetric(x) || diagonal(x)"
  // CHECK: tessera.guard "symmetric(x)"
  // CHECK: tessera.call @lib.h
  llvm.func @either(%a: !llvm.ptr {llvm.noalias}) -> f64 {
    %r = tessera.call @lib.k(%a) : (!llvm.ptr) -> f64
    llvm.return %r : f64
  }
}
