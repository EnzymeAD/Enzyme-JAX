// RUN: enzymexlamlir-opt %s -parse-optimization-rules -tessera-propagate-properties -tessera-apply-pdl -split-input-file -verify-diagnostics | FileCheck %s

// A chain rule matches a nested producer as well as the root. The producer has
// to move with the root: if it stayed where it was, it would still run
// unconditionally, and on the specialized path a producer with side effects --
// one accumulating into an output buffer, as every FEM element kernel does --
// would run on top of its replacement and the result would be wrong.

// The producer runs in the else region only, never ahead of the guard.
module {
  tessera.define private @exa.btcb(!llvm.ptr, !llvm.ptr, !llvm.ptr, i64) -> !llvm.ptr attributes {argModes = [unit, unit, unit, unit], pure = false}
  tessera.define private @exa.btcb_packed(!llvm.ptr, !llvm.ptr, !llvm.ptr, i64) -> !llvm.ptr attributes {argModes = [unit, unit, unit, unit], pure = false}
  tessera.define private @exa.assemble(!llvm.ptr, !llvm.ptr, !llvm.ptr, i64) -> f64 attributes {argModes = [unit, unit, unit, unit], pure = false}
  tessera.define private @exa.assemble_packed(!llvm.ptr, !llvm.ptr, !llvm.ptr, i64) -> f64 attributes {argModes = [unit, unit, unit, unit], pure = false}

  tessera.optimizations {
    tessera.optimization "if symmetric(C), exa.assemble(exa.btcb(B, C, K, n), C, G, n) -> exa.assemble_packed(exa.btcb_packed(B, C, K, n), C, G, n)"
  }

  // CHECK-LABEL: llvm.func @site
  // CHECK-NOT: tessera.call @exa.btcb(
  // CHECK: tessera.guard "symmetric(C)"
  // CHECK-NEXT: %[[P:.*]] = tessera.call @exa.btcb_packed(%arg0, %arg1, %arg2, %arg4)
  // CHECK-NEXT: %[[A:.*]] = tessera.call @exa.assemble_packed(%[[P]], %arg1, %arg3, %arg4)
  // CHECK-NEXT: tessera.yield %[[A]]
  // CHECK-NEXT: } else {
  // CHECK-NEXT: %[[B:.*]] = tessera.call @exa.btcb(%arg0, %arg1, %arg2, %arg4)
  // CHECK-NEXT: %[[O:.*]] = tessera.call @exa.assemble(%[[B]], %arg1, %arg3, %arg4)
  // CHECK-NEXT: tessera.yield %[[O]]
  // CHECK-NEXT: }
  // CHECK-NEXT: llvm.return
  llvm.func @site(%B: !llvm.ptr, %C: !llvm.ptr, %K: !llvm.ptr, %G: !llvm.ptr, %n: i64) -> f64 {
    %0 = tessera.call @exa.btcb(%B, %C, %K, %n) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64) -> !llvm.ptr
    %1 = tessera.call @exa.assemble(%0, %C, %G, %n) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64) -> f64
    llvm.return %1 : f64
  }
}

// -----

// A pure producer used only by the chain moves too. That is not needed for
// correctness -- left behind it would merely be dead on one path -- but it
// keeps the specialized path from paying for it.
module {
  tessera.define private @lib.g(!llvm.ptr) -> !llvm.ptr attributes {argModes = [unit], pure = true}
  tessera.define private @lib.f(!llvm.ptr) -> f64 attributes {argModes = [unit], pure = true}
  tessera.define private @lib.fast(!llvm.ptr) -> f64 attributes {argModes = [unit], pure = true}

  tessera.optimizations {
    tessera.optimization "if symmetric(x), lib.f(lib.g(x)) -> lib.fast(x)"
  }

  // CHECK-LABEL: llvm.func @pure_producer
  // CHECK-NOT: tessera.call @lib.g(
  // CHECK: tessera.guard "symmetric(x)"
  // CHECK-NEXT: tessera.call @lib.fast(%arg0)
  // CHECK: } else {
  // CHECK-NEXT: %[[G:.*]] = tessera.call @lib.g(%arg0)
  // CHECK-NEXT: tessera.call @lib.f(%[[G]])
  llvm.func @pure_producer(%x: !llvm.ptr) -> f64 {
    %0 = tessera.call @lib.g(%x) : (!llvm.ptr) -> !llvm.ptr
    %1 = tessera.call @lib.f(%0) : (!llvm.ptr) -> f64
    llvm.return %1 : f64
  }
}

// -----

// A pure producer with another user stays where it is and is shared: it
// computes the same thing whichever path runs.
module {
  tessera.define private @lib.g(!llvm.ptr) -> !llvm.ptr attributes {argModes = [unit], pure = true}
  tessera.define private @lib.f(!llvm.ptr) -> f64 attributes {argModes = [unit], pure = true}
  tessera.define private @lib.fast(!llvm.ptr) -> f64 attributes {argModes = [unit], pure = true}

  tessera.optimizations {
    tessera.optimization "if symmetric(x), lib.f(lib.g(x)) -> lib.fast(x)"
  }

  // CHECK-LABEL: llvm.func @shared_pure_producer
  // CHECK: %[[G:.*]] = tessera.call @lib.g(%arg0)
  // CHECK-NEXT: tessera.guard "symmetric(x)"
  // CHECK: } else {
  // CHECK-NEXT: tessera.call @lib.f(%[[G]])
  // CHECK: llvm.store %[[G]], %arg1
  llvm.func @shared_pure_producer(%x: !llvm.ptr, %out: !llvm.ptr) -> f64 {
    %0 = tessera.call @lib.g(%x) : (!llvm.ptr) -> !llvm.ptr
    %1 = tessera.call @lib.f(%0) : (!llvm.ptr) -> f64
    llvm.store %0, %out : !llvm.ptr, !llvm.ptr
    llvm.return %1 : f64
  }
}

// -----

// A producer with side effects that something else also uses cannot move:
// the other user still needs it, and the replacement would then run in
// addition to it. The rule declines rather than miscompiling -- and declining
// in the match leaves the IR untouched, so the greedy driver does not loop.
module {
  tessera.define private @exa.btcb(!llvm.ptr, !llvm.ptr, !llvm.ptr, i64) -> !llvm.ptr attributes {argModes = [unit, unit, unit, unit], pure = false}
  tessera.define private @exa.btcb_packed(!llvm.ptr, !llvm.ptr, !llvm.ptr, i64) -> !llvm.ptr attributes {argModes = [unit, unit, unit, unit], pure = false}
  tessera.define private @exa.assemble(!llvm.ptr, !llvm.ptr, !llvm.ptr, i64) -> f64 attributes {argModes = [unit, unit, unit, unit], pure = false}
  tessera.define private @exa.assemble_packed(!llvm.ptr, !llvm.ptr, !llvm.ptr, i64) -> f64 attributes {argModes = [unit, unit, unit, unit], pure = false}

  tessera.optimizations {
    tessera.optimization "if symmetric(C), exa.assemble(exa.btcb(B, C, K, n), C, G, n) -> exa.assemble_packed(exa.btcb_packed(B, C, K, n), C, G, n)"
  }

  // CHECK-LABEL: llvm.func @shared_effectful_producer
  // CHECK-NOT: tessera.guard
  // CHECK: tessera.call @exa.btcb(
  // CHECK-NEXT: tessera.call @exa.assemble(
  llvm.func @shared_effectful_producer(%B: !llvm.ptr, %C: !llvm.ptr, %K: !llvm.ptr, %G: !llvm.ptr, %n: i64) -> f64 {
    %0 = tessera.call @exa.btcb(%B, %C, %K, %n) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64) -> !llvm.ptr
    // expected-warning @+1 {{was not applied here: the matched call to 'exa.btcb' has side effects and its results are also used outside the matched expression}}
    %1 = tessera.call @exa.assemble(%0, %C, %G, %n) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64) -> f64
    llvm.store %0, %G : !llvm.ptr, !llvm.ptr
    llvm.return %1 : f64
  }
}

// -----

// Moving a producer with side effects down to the root is only sound when
// nothing between them has side effects of its own.
module {
  tessera.define private @exa.btcb(!llvm.ptr, !llvm.ptr, !llvm.ptr, i64) -> !llvm.ptr attributes {argModes = [unit, unit, unit, unit], pure = false}
  tessera.define private @exa.btcb_packed(!llvm.ptr, !llvm.ptr, !llvm.ptr, i64) -> !llvm.ptr attributes {argModes = [unit, unit, unit, unit], pure = false}
  tessera.define private @exa.assemble(!llvm.ptr, !llvm.ptr, !llvm.ptr, i64) -> f64 attributes {argModes = [unit, unit, unit, unit], pure = false}
  tessera.define private @exa.assemble_packed(!llvm.ptr, !llvm.ptr, !llvm.ptr, i64) -> f64 attributes {argModes = [unit, unit, unit, unit], pure = false}

  tessera.optimizations {
    tessera.optimization "if symmetric(C), exa.assemble(exa.btcb(B, C, K, n), C, G, n) -> exa.assemble_packed(exa.btcb_packed(B, C, K, n), C, G, n)"
  }

  // CHECK-LABEL: llvm.func @effect_in_between
  // CHECK-NOT: tessera.guard
  llvm.func @effect_in_between(%B: !llvm.ptr, %C: !llvm.ptr, %K: !llvm.ptr, %G: !llvm.ptr, %n: i64, %v: f64) -> f64 {
    %0 = tessera.call @exa.btcb(%B, %C, %K, %n) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64) -> !llvm.ptr
    llvm.store %v, %K : f64, !llvm.ptr
    // expected-warning @+1 {{was not applied here: the matched call to 'exa.btcb' has side effects and would have to move past 'llvm.store', which has side effects of its own}}
    %1 = tessera.call @exa.assemble(%0, %C, %G, %n) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64) -> f64
    llvm.return %1 : f64
  }
}

// -----

// Two levels of nesting: the else region recomputes the whole chain, in the
// order the program ran it.
module {
  tessera.define private @lib.h(!llvm.ptr) -> !llvm.ptr attributes {argModes = [unit], pure = false}
  tessera.define private @lib.g(!llvm.ptr) -> !llvm.ptr attributes {argModes = [unit], pure = false}
  tessera.define private @lib.f(!llvm.ptr) -> f64 attributes {argModes = [unit], pure = false}
  tessera.define private @lib.fast(!llvm.ptr) -> f64 attributes {argModes = [unit], pure = false}

  tessera.optimizations {
    tessera.optimization "if symmetric(x), lib.f(lib.g(lib.h(x))) -> lib.fast(x)"
  }

  // CHECK-LABEL: llvm.func @two_levels
  // CHECK-NOT: tessera.call @lib.h(
  // CHECK-NOT: tessera.call @lib.g(
  // CHECK: tessera.guard "symmetric(x)"
  // CHECK: } else {
  // CHECK-NEXT: %[[H:.*]] = tessera.call @lib.h(%arg0)
  // CHECK-NEXT: %[[G:.*]] = tessera.call @lib.g(%[[H]])
  // CHECK-NEXT: tessera.call @lib.f(%[[G]])
  llvm.func @two_levels(%x: !llvm.ptr) -> f64 {
    %0 = tessera.call @lib.h(%x) : (!llvm.ptr) -> !llvm.ptr
    %1 = tessera.call @lib.g(%0) : (!llvm.ptr) -> !llvm.ptr
    %2 = tessera.call @lib.f(%1) : (!llvm.ptr) -> f64
    llvm.return %2 : f64
  }
}

// -----

// One producer feeding the root twice is cloned once.
module {
  tessera.define private @lib.g(!llvm.ptr) -> !llvm.ptr attributes {argModes = [unit], pure = false}
  tessera.define private @lib.pair(!llvm.ptr, !llvm.ptr) -> f64 attributes {argModes = [unit, unit], pure = false}
  tessera.define private @lib.fast(!llvm.ptr) -> f64 attributes {argModes = [unit], pure = false}

  tessera.optimizations {
    tessera.optimization "if symmetric(x), lib.pair(lib.g(x), lib.g(x)) -> lib.fast(x)"
  }

  // CHECK-LABEL: llvm.func @used_twice
  // CHECK: } else {
  // CHECK-NEXT: %[[G:.*]] = tessera.call @lib.g(%arg0)
  // CHECK-NEXT: tessera.call @lib.pair(%[[G]], %[[G]])
  // CHECK-NOT: tessera.call @lib.g(
  llvm.func @used_twice(%x: !llvm.ptr) -> f64 {
    %0 = tessera.call @lib.g(%x) : (!llvm.ptr) -> !llvm.ptr
    %1 = tessera.call @lib.pair(%0, %0) : (!llvm.ptr, !llvm.ptr) -> f64
    llvm.return %1 : f64
  }
}

// -----

// An unconditional chain rule. Its nested right-hand side could not be built
// before, and once it can, the producer must not be left behind either.
module {
  tessera.define private @exa.btcb(!llvm.ptr, !llvm.ptr, !llvm.ptr, i64) -> !llvm.ptr attributes {argModes = [unit, unit, unit, unit], pure = false}
  tessera.define private @exa.btcb_packed(!llvm.ptr, !llvm.ptr, !llvm.ptr, i64) -> !llvm.ptr attributes {argModes = [unit, unit, unit, unit], pure = false}
  tessera.define private @exa.assemble(!llvm.ptr, !llvm.ptr, !llvm.ptr, i64) -> f64 attributes {argModes = [unit, unit, unit, unit], pure = false}
  tessera.define private @exa.assemble_packed(!llvm.ptr, !llvm.ptr, !llvm.ptr, i64) -> f64 attributes {argModes = [unit, unit, unit, unit], pure = false}

  tessera.optimizations {
    tessera.optimization "exa.assemble(exa.btcb(B, C, K, n), C, G, n) -> exa.assemble_packed(exa.btcb_packed(B, C, K, n), C, G, n)"
  }

  // CHECK-LABEL: llvm.func @unconditional
  // CHECK-NEXT: %[[P:.*]] = tessera.call @exa.btcb_packed(%arg0, %arg1, %arg2, %arg4)
  // CHECK-NEXT: %[[A:.*]] = tessera.call @exa.assemble_packed(%[[P]], %arg1, %arg3, %arg4)
  // CHECK-NEXT: llvm.return %[[A]]
  llvm.func @unconditional(%B: !llvm.ptr, %C: !llvm.ptr, %K: !llvm.ptr, %G: !llvm.ptr, %n: i64) -> f64 {
    %0 = tessera.call @exa.btcb(%B, %C, %K, %n) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64) -> !llvm.ptr
    %1 = tessera.call @exa.assemble(%0, %C, %G, %n) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64) -> f64
    llvm.return %1 : f64
  }
}

// -----

// A proven condition rewrites outright, and the producer goes with the root
// here too.
module {
  tessera.define private @exa.build() -> (!llvm.ptr {tessera.property = ["symmetric"]}) attributes {argModes = [], pure = false}
  tessera.define private @exa.btcb(!llvm.ptr, !llvm.ptr, !llvm.ptr, i64) -> !llvm.ptr attributes {argModes = [unit, unit, unit, unit], pure = false}
  tessera.define private @exa.btcb_packed(!llvm.ptr, !llvm.ptr, !llvm.ptr, i64) -> !llvm.ptr attributes {argModes = [unit, unit, unit, unit], pure = false}
  tessera.define private @exa.assemble(!llvm.ptr, !llvm.ptr, !llvm.ptr, i64) -> f64 attributes {argModes = [unit, unit, unit, unit], pure = false}
  tessera.define private @exa.assemble_packed(!llvm.ptr, !llvm.ptr, !llvm.ptr, i64) -> f64 attributes {argModes = [unit, unit, unit, unit], pure = false}

  tessera.optimizations {
    tessera.optimization "if symmetric(C), exa.assemble(exa.btcb(B, C, K, n), C, G, n) -> exa.assemble_packed(exa.btcb_packed(B, C, K, n), C, G, n)"
  }

  // CHECK-LABEL: llvm.func @proven
  // CHECK-NEXT: %[[C:.*]] = tessera.call @exa.build()
  // CHECK-NEXT: %[[P:.*]] = tessera.call @exa.btcb_packed(%arg0, %[[C]], %arg1, %arg3)
  // CHECK-NEXT: %[[A:.*]] = tessera.call @exa.assemble_packed(%[[P]], %[[C]], %arg2, %arg3)
  // CHECK-NEXT: llvm.return %[[A]]
  llvm.func @proven(%B: !llvm.ptr, %K: !llvm.ptr, %G: !llvm.ptr, %n: i64) -> f64 {
    %C = tessera.call @exa.build() : () -> !llvm.ptr
    %0 = tessera.call @exa.btcb(%B, %C, %K, %n) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64) -> !llvm.ptr
    %1 = tessera.call @exa.assemble(%0, %C, %G, %n) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64) -> f64
    llvm.return %1 : f64
  }
}
