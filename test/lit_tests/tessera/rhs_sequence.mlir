// RUN: enzymexlamlir-opt %s -parse-optimization-rules -tessera-apply-pdl -split-input-file -verify-diagnostics | FileCheck %s

// A right-hand side of several expressions separated by ';' runs them in
// order, like C statements, and the last one replaces the matched call. The
// calls before it are made for their effect. This is what a library that is
// told facts, rather than offering a specialized function, needs: `f(x) ->
// g(x); f(x)` keeps the call and tells the library something first.

// The call made for its effect comes first, its literal takes the type of the
// parameter it is passed to, and the rebuilt call to the matched callee is
// marked so the rule does not match it again and add the call without end.
module {
  tessera.define private @lib.solve(i64) -> i32 attributes {argModes = [unit], pure = false}
  tessera.define private @lib.mark(i64, i32) -> i32 attributes {argModes = [unit, unit], pure = false}

  tessera.optimizations {
    tessera.optimization "lib.solve(x) -> lib.mark(x, 15); lib.solve(x)"
  }

  // CHECK-LABEL: llvm.func @effect_first
  // CHECK-NEXT: %[[C:.*]] = llvm.mlir.constant(15 : i32) : i32
  // CHECK-NEXT: tessera.call @lib.mark(%arg0, %[[C]]) : (i64, i32) -> i32
  // CHECK-NEXT: %[[R:.*]] = tessera.call @lib.solve(%arg0) {tessera.applied_rules = ["lib.solve(x) -> lib.mark(x, 15); lib.solve(x)"]} : (i64) -> i32
  // CHECK-NEXT: llvm.return %[[R]]
  llvm.func @effect_first(%x: i64) -> i32 {
    %0 = tessera.call @lib.solve(%x) : (i64) -> i32
    llvm.return %0 : i32
  }
}

// -----

// Only a call that repeats the matched one is marked. One to the same callee
// with other arguments is progress, and the rule goes on applying to it:
// here it strips both negations, not just the outer one.
module {
  tessera.define private @lib.f(f64) -> f64 attributes {argModes = [unit], pure = true}
  tessera.define private @lib.neg(f64) -> f64 attributes {argModes = [unit], pure = true}

  tessera.optimizations {
    tessera.optimization "lib.f(lib.neg(x)) -> lib.f(x)"
  }

  // CHECK-LABEL: llvm.func @progress
  // CHECK-NEXT: %[[F:.*]] = tessera.call @lib.f(%arg0) : (f64) -> f64
  // CHECK-NEXT: llvm.return %[[F]]
  llvm.func @progress(%x: f64) -> f64 {
    %0 = tessera.call @lib.neg(%x) : (f64) -> f64
    %1 = tessera.call @lib.neg(%0) : (f64) -> f64
    %2 = tessera.call @lib.f(%1) : (f64) -> f64
    llvm.return %2 : f64
  }
}

// -----

// Several calls run in the order written, and the replacement need not call
// the matched callee at all, in which case nothing is marked.
module {
  tessera.define private @lib.f(i64) -> i64 attributes {argModes = [unit], pure = false}
  tessera.define private @lib.a(i64) -> i64 attributes {argModes = [unit], pure = false}
  tessera.define private @lib.b(i64) -> i64 attributes {argModes = [unit], pure = false}
  tessera.define private @lib.g(i64) -> i64 attributes {argModes = [unit], pure = false}

  tessera.optimizations {
    tessera.optimization "lib.f(x) -> lib.a(x); lib.b(x); lib.g(x)"
  }

  // CHECK-LABEL: llvm.func @in_order
  // CHECK-NEXT: tessera.call @lib.a(%arg0) : (i64) -> i64
  // CHECK-NEXT: tessera.call @lib.b(%arg0) : (i64) -> i64
  // CHECK-NEXT: %[[G:.*]] = tessera.call @lib.g(%arg0) : (i64) -> i64
  // CHECK-NEXT: llvm.return %[[G]]
  llvm.func @in_order(%x: i64) -> i64 {
    %0 = tessera.call @lib.f(%x) : (i64) -> i64
    llvm.return %0 : i64
  }
}

// -----

// Under a guard, the whole sequence is the specialized path, and the rebuilt
// call in it is not wrapped in a guard of its own.
module {
  tessera.define private @lib.solve(i64) -> i32 attributes {argModes = [unit], pure = false}
  tessera.define private @lib.mark(i64, i32) -> i32 attributes {argModes = [unit, unit], pure = false}

  tessera.optimizations {
    tessera.optimization "if n > 0, lib.solve(n) -> lib.mark(n, 1); lib.solve(n)"
  }

  // CHECK-LABEL: llvm.func @guarded
  // CHECK-NEXT: %[[C:.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK-NEXT: tessera.guard "n > 0"
  // CHECK-NEXT: tessera.call @lib.mark(%arg0, %[[C]]) : (i64, i32) -> i32
  // CHECK-NEXT: %[[T:.*]] = tessera.call @lib.solve(%arg0) {tessera.applied_rules = {{.*}}} : (i64) -> i32
  // CHECK-NEXT: tessera.yield %[[T]] : i32
  // CHECK-NOT: tessera.guard
  // CHECK: llvm.return
  llvm.func @guarded(%n: i64) -> i32 {
    %0 = tessera.call @lib.solve(%n) : (i64) -> i32
    llvm.return %0 : i32
  }
}

// -----

// Only the last expression has a value that is used, so anything before a ';'
// must be a call.
module {
  tessera.define private @lib.f(i64) -> i64 attributes {argModes = [unit], pure = false}
  tessera.define private @lib.g(i64) -> i64 attributes {argModes = [unit], pure = false}

  tessera.optimizations {
    // expected-warning @+1 {{'x' is followed by ';' in an optimization rule, but only a call can be}}
    tessera.optimization "lib.f(x) -> x; lib.g(x)"
  }

  // CHECK-LABEL: llvm.func @not_a_call
  // CHECK-NEXT: tessera.call @lib.f(%arg0)
  llvm.func @not_a_call(%x: i64) -> i64 {
    %0 = tessera.call @lib.f(%x) : (i64) -> i64
    llvm.return %0 : i64
  }
}

// -----

// A call made for its effect is checked like any other: one with the wrong
// number of arguments is reported, and the rule left out.
module {
  tessera.define private @lib.solve(i64) -> i32 attributes {argModes = [unit], pure = false}
  tessera.define private @lib.mark(i64, i32) -> i32 attributes {argModes = [unit, unit], pure = false}

  tessera.optimizations {
    tessera.optimization "lib.solve(x) -> lib.mark(x); lib.solve(x)"
  }

  // CHECK-LABEL: llvm.func @bad_effect
  // CHECK-NEXT: tessera.call @lib.solve(%arg0) : (i64) -> i32
  llvm.func @bad_effect(%x: i64) -> i32 {
    // expected-warning @+1 {{optimization rule 'lib.solve(x) -> lib.mark(x); lib.solve(x)' cannot be applied: its right-hand side calls 'lib.mark' with 1 argument(s), but it takes 2}}
    %0 = tessera.call @lib.solve(%x) : (i64) -> i32
    llvm.return %0 : i32
  }
}
