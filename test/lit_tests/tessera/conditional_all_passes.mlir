// RUN: enzymexlamlir-opt %s -parse-optimization-rules -tessera-apply-pdl -tessera-lower-guards -tessera-to-llvm -split-input-file | FileCheck %s

// A conditional rule the whole way through: parsed into a pattern, applied
// into a guard, lowered into a branch, and finally out of the tessera dialect
// altogether. What this pins is that nothing tessera-specific survives and
// that BOTH branches are ordinary calls -- guards are lowered before
// tessera-to-llvm precisely so the call in each one still gets the usual
// treatment rather than needing its own.

module {
  tessera.define @lib.foo(%a: !llvm.ptr {tessera.layout = {elem = f64, rows = 2 : i64, cols = 2 : i64, row_major = true}}) -> f64 attributes {byRefTypes = [unit], pure = true, tessera.original_name = "foo"} {
    %c = llvm.mlir.constant(0.0 : f64) : f64
    tessera.return %c : f64
  }
  tessera.define @lib.symmetric_foo(%a: !llvm.ptr) -> f64 attributes {byRefTypes = [unit], pure = true, tessera.original_name = "symmetric_foo"} {
    %c = llvm.mlir.constant(0.0 : f64) : f64
    tessera.return %c : f64
  }

  tessera.optimizations {
    tessera.optimization "if symmetric(x), lib.foo(x) -> lib.symmetric_foo(x)"
  }

  // CHECK-LABEL: llvm.func @main
  llvm.func @main(%x: !llvm.ptr) -> f64 {
    // The synthesized check, reading the same memory the callee will read.
    // CHECK: %[[G1:.*]] = llvm.getelementptr %arg0[1]
    // CHECK: %[[E1:.*]] = llvm.load %[[G1]] : !llvm.ptr -> f64
    // CHECK: %[[G2:.*]] = llvm.getelementptr %arg0[2]
    // CHECK: %[[E2:.*]] = llvm.load %[[G2]] : !llvm.ptr -> f64
    // CHECK: %[[P:.*]] = llvm.fcmp "oeq" %[[E1]], %[[E2]] : f64
    // CHECK: llvm.cond_br %[[P]], ^[[THEN:.*]], ^[[ELSE:.*]]

    // Both branches end up as plain llvm.call, under the names the defines
    // declared, and the phi is a block argument.
    // CHECK: ^[[THEN]]:
    // CHECK: %[[S:.*]] = llvm.call @symmetric_foo(%arg0)
    // CHECK: llvm.br ^[[TAIL:.*]](%[[S]] : f64)
    // CHECK: ^[[ELSE]]:
    // CHECK: %[[O:.*]] = llvm.call @foo(%arg0)
    // CHECK: llvm.br ^[[TAIL]](%[[O]] : f64)
    // CHECK: ^[[TAIL]](%[[PHI:.*]]: f64):
    // CHECK: llvm.return %[[PHI]]
    %0 = tessera.call @lib.foo(%x) : (!llvm.ptr) -> f64
    llvm.return %0 : f64
  }

  // Nothing from the tessera dialect is left, and neither is the bookkeeping
  // the rewriter used to keep from matching its own output.
  // CHECK-NOT: tessera.guard
  // CHECK-NOT: tessera.call
  // CHECK-NOT: tessera.define
  // CHECK-NOT: tessera.applied_rules
}

// -----

// The same rule where the condition is already known to hold: the check is
// elided outright, so what comes out the far end is a single direct call with
// no branch at all.
module {
  tessera.define @lib.build_cov() -> (!llvm.ptr {tessera.guarantees = ["symmetric"]}) attributes {byRefTypes = [], pure = true, tessera.original_name = "build_cov"} {
    %c = llvm.mlir.zero : !llvm.ptr
    tessera.return %c : !llvm.ptr
  }
  tessera.define @lib.foo(%a: !llvm.ptr {tessera.layout = {elem = f64, rows = 2 : i64, cols = 2 : i64, row_major = true}}) -> f64 attributes {byRefTypes = [unit], pure = true, tessera.original_name = "foo"} {
    %c = llvm.mlir.constant(0.0 : f64) : f64
    tessera.return %c : f64
  }
  tessera.define @lib.symmetric_foo(%a: !llvm.ptr) -> f64 attributes {byRefTypes = [unit], pure = true, tessera.original_name = "symmetric_foo"} {
    %c = llvm.mlir.constant(0.0 : f64) : f64
    tessera.return %c : f64
  }

  tessera.optimizations {
    tessera.optimization "if symmetric(x), lib.foo(x) -> lib.symmetric_foo(x)"
  }

  // CHECK-LABEL: llvm.func @proven
  llvm.func @proven() -> f64 {
    // CHECK: %[[M:.*]] = llvm.call @build_cov()
    // CHECK-NEXT: %[[R:.*]] = llvm.call @symmetric_foo(%[[M]])
    // CHECK-NEXT: llvm.return %[[R]]
    // CHECK-NOT: llvm.cond_br
    // CHECK-NOT: llvm.fcmp
    %0 = tessera.call @lib.build_cov() : () -> !llvm.ptr
    %1 = tessera.call @lib.foo(%0) : (!llvm.ptr) -> f64
    llvm.return %1 : f64
  }
}
