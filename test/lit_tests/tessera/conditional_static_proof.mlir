// RUN: enzymexlamlir-opt %s -parse-optimization-rules -tessera-apply-pdl -split-input-file | FileCheck %s

// When the condition is already known to hold, the rewrite is applied outright
// and no check is emitted. This is a saving over the guarded form, never a
// precondition for it: an unprovable condition still works, it just costs a
// check at run time.

// A producing function declares what it returns. Written once on the
// declaration, it holds at every call site.
module {
  tessera.define @lib.build_cov(%n: i64) -> (i512 {tessera.guarantees = ["symmetric"]}) attributes {byRefTypes = [unit], pure = true} {
    %c = llvm.mlir.constant(0 : i512) : i512
    tessera.return %c : i512
  }
  tessera.define @lib.foo(%a: i512) -> f32 attributes {byRefTypes = [unit], pure = true} {
    %c = llvm.mlir.constant(0.0 : f32) : f32
    tessera.return %c : f32
  }
  tessera.define @lib.symmetric_foo(%a: i512) -> f32 attributes {byRefTypes = [unit], pure = true} {
    %c = llvm.mlir.constant(0.0 : f32) : f32
    tessera.return %c : f32
  }

  tessera.optimizations {
    tessera.optimization "if symmetric(x), lib.foo(x) -> lib.symmetric_foo(x)"
  }

  // CHECK-LABEL: llvm.func @guaranteed_result
  llvm.func @guaranteed_result(%n: i64) -> f32 {
    // CHECK: %[[M:.*]] = tessera.call @lib.build_cov
    // CHECK-NEXT: %[[R:.*]] = tessera.call @lib.symmetric_foo(%[[M]])
    // CHECK-NEXT: llvm.return %[[R]]
    // CHECK-NOT: tessera.guard
    %0 = tessera.call @lib.build_cov(%n) : (i64) -> i512
    %1 = tessera.call @lib.foo(%0) : (i512) -> f32
    llvm.return %1 : f32
  }
}

// -----

// The same matrix without the guarantee gets a guard, which is what makes the
// case above a genuine elision rather than the only path.
module {
  tessera.define @lib.build_any(%n: i64) -> i512 attributes {byRefTypes = [unit], pure = true} {
    %c = llvm.mlir.constant(0 : i512) : i512
    tessera.return %c : i512
  }
  tessera.define @lib.foo(%a: i512) -> f32 attributes {byRefTypes = [unit], pure = true} {
    %c = llvm.mlir.constant(0.0 : f32) : f32
    tessera.return %c : f32
  }
  tessera.define @lib.symmetric_foo(%a: i512) -> f32 attributes {byRefTypes = [unit], pure = true} {
    %c = llvm.mlir.constant(0.0 : f32) : f32
    tessera.return %c : f32
  }

  tessera.optimizations {
    tessera.optimization "if symmetric(x), lib.foo(x) -> lib.symmetric_foo(x)"
  }

  // CHECK-LABEL: llvm.func @no_guarantee
  // CHECK: tessera.guard "symmetric(x)"
  llvm.func @no_guarantee(%n: i64) -> f32 {
    %0 = tessera.call @lib.build_any(%n) : (i64) -> i512
    %1 = tessera.call @lib.foo(%0) : (i512) -> f32
    llvm.return %1 : f32
  }
}

// -----

// A property can also be stated on the operation that produced the value.
module {
  tessera.define @lib.foo(%a: i512) -> f32 attributes {byRefTypes = [unit], pure = true} {
    %c = llvm.mlir.constant(0.0 : f32) : f32
    tessera.return %c : f32
  }
  tessera.define @lib.diagonal_foo(%a: i512) -> f32 attributes {byRefTypes = [unit], pure = true} {
    %c = llvm.mlir.constant(0.0 : f32) : f32
    tessera.return %c : f32
  }

  tessera.optimizations {
    tessera.optimization "if diagonal(x), lib.foo(x) -> lib.diagonal_foo(x)"
  }

  // CHECK-LABEL: llvm.func @property_on_value
  // CHECK: tessera.call @lib.diagonal_foo
  // CHECK-NOT: tessera.guard
  llvm.func @property_on_value() -> f32 {
    %0 = llvm.mlir.constant(0 : i512) {"tessera.property.diagonal"} : i512
    %1 = tessera.call @lib.foo(%0) : (i512) -> f32
    llvm.return %1 : f32
  }
}

// -----

// A comparison against a constant folds at compile time, so a rule guarded on
// a size that is already known costs nothing.
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

  // CHECK-LABEL: llvm.func @constant_folds_true
  // CHECK: tessera.call @lib.tiled_qux
  // CHECK-NOT: tessera.guard
  // CHECK-NOT: llvm.icmp
  llvm.func @constant_folds_true(%x: i512) -> f32 {
    %n = llvm.mlir.constant(128 : i64) : i64
    %0 = tessera.call @lib.qux(%x, %n) : (i512, i64) -> f32
    llvm.return %0 : f32
  }
}

// -----

// The same rule with a non-constant size still guards.
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

  // CHECK-LABEL: llvm.func @dynamic_size_guards
  // CHECK: tessera.guard "n > 64"
  llvm.func @dynamic_size_guards(%x: i512, %n: i64) -> f32 {
    %0 = tessera.call @lib.qux(%x, %n) : (i512, i64) -> f32
    llvm.return %0 : f32
  }
}

// -----

// Conjunction: both sides provable, so the whole condition is.
module {
  tessera.define @lib.build_cov(%n: i64) -> (i512 {tessera.guarantees = ["symmetric"]}) attributes {byRefTypes = [unit], pure = true} {
    %c = llvm.mlir.constant(0 : i512) : i512
    tessera.return %c : i512
  }
  tessera.define @lib.qux(%a: i512, %n: i64) -> f32 attributes {byRefTypes = [unit, unit], pure = true} {
    %c = llvm.mlir.constant(0.0 : f32) : f32
    tessera.return %c : f32
  }
  tessera.define @lib.special_qux(%a: i512, %n: i64) -> f32 attributes {byRefTypes = [unit, unit], pure = true} {
    %c = llvm.mlir.constant(0.0 : f32) : f32
    tessera.return %c : f32
  }

  tessera.optimizations {
    tessera.optimization "if symmetric(x) && n > 64, lib.qux(x, n) -> lib.special_qux(x, n)"
  }

  // CHECK-LABEL: llvm.func @conjunction_both_proven
  // CHECK: tessera.call @lib.special_qux
  // CHECK-NOT: tessera.guard
  llvm.func @conjunction_both_proven(%k: i64) -> f32 {
    %n = llvm.mlir.constant(128 : i64) : i64
    %m = tessera.call @lib.build_cov(%k) : (i64) -> i512
    %0 = tessera.call @lib.qux(%m, %n) : (i512, i64) -> f32
    llvm.return %0 : f32
  }
}
