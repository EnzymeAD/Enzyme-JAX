// RUN: enzymexlamlir-opt %s -parse-optimization-rules -tessera-apply-pdl -split-input-file | FileCheck %s

// When a condition cannot be proven, the rewrite is recorded as a
// tessera.guard rather than applied: the specialized call in the first region,
// the original in the second. This is the default path -- proving a condition
// is an optimization over it, not a precondition for it.

module {
  tessera.define @lib.qux(%a: i512, %n: i64) -> f32 attributes {byRefTypes = [unit, unit], pure = true} {
    %c = llvm.mlir.constant(0.0 : f32) : f32
    tessera.return %c : f32
  }
  tessera.define @lib.tiled_qux(%a: i512, %n: i64) -> f32 attributes {byRefTypes = [unit, unit], pure = true} {
    %c = llvm.mlir.constant(0.0 : f32) : f32
    tessera.return %c : f32
  }

  tessera.optimizations {
    tessera.optimization "if n > 64, lib.qux(x, n) -> lib.tiled_qux(x, n)"
  }

  // CHECK-LABEL: llvm.func @main
  llvm.func @main(%x: i512, %n: i64) -> f32 {
    // The guard carries the condition alone, not the whole rule, and names the
    // values it refers to.
    // CHECK: %[[G:.*]] = tessera.guard "n > 64" args(%arg0, %arg1) {argNames = ["x", "n"]} : (i512, i64) -> f32 {
    // CHECK-NEXT: %[[S:.*]] = tessera.call @lib.tiled_qux(%arg0, %arg1)
    // CHECK-NEXT: tessera.yield %[[S]] : f32
    // CHECK-NEXT: } else {

    // The original call is kept verbatim, tagged with the rule that already
    // fired on it so the same pattern cannot match it again.
    // CHECK-NEXT: %[[O:.*]] = tessera.call @lib.qux(%arg0, %arg1) {tessera.applied_rules = ["if n > 64, lib.qux(x, n) -> lib.tiled_qux(x, n)"]}
    // CHECK-NEXT: tessera.yield %[[O]] : f32
    // CHECK-NEXT: }
    %0 = tessera.call @lib.qux(%x, %n) : (i512, i64) -> f32
    // CHECK: llvm.return %[[G]]
    llvm.return %0 : f32
  }
}

// -----

// Two rules matching the same call. The second must still apply to the copy
// the first one's guard kept, so the tagging has to be per rule rather than a
// blanket "already rewritten" mark -- otherwise only one rule would ever fire.

module {
  tessera.define @lib.qux(%a: i512, %n: i64) -> f32 attributes {byRefTypes = [unit, unit], pure = true} {
    %c = llvm.mlir.constant(0.0 : f32) : f32
    tessera.return %c : f32
  }
  tessera.define @lib.tiled_qux(%a: i512, %n: i64) -> f32 attributes {byRefTypes = [unit, unit], pure = true} {
    %c = llvm.mlir.constant(0.0 : f32) : f32
    tessera.return %c : f32
  }
  tessera.define @lib.small_qux(%a: i512, %n: i64) -> f32 attributes {byRefTypes = [unit, unit], pure = true} {
    %c = llvm.mlir.constant(0.0 : f32) : f32
    tessera.return %c : f32
  }

  tessera.optimizations {
    tessera.optimization "if n > 64, lib.qux(x, n) -> lib.tiled_qux(x, n)"
    tessera.optimization "if n < 4, lib.qux(x, n) -> lib.small_qux(x, n)"
  }

  // CHECK-LABEL: llvm.func @two_rules
  // CHECK: tessera.guard
  // CHECK: tessera.call @lib.small_qux
  // CHECK: tessera.guard
  // CHECK: tessera.call @lib.tiled_qux
  llvm.func @two_rules(%x: i512, %n: i64) -> f32 {
    %0 = tessera.call @lib.qux(%x, %n) : (i512, i64) -> f32
    llvm.return %0 : f32
  }
}
