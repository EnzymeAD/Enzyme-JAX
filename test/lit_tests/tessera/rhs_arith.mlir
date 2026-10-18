// RUN: enzymexlamlir-opt %s -parse-optimization-rules -tessera-propagate-properties -tessera-apply-pdl -split-input-file -verify-diagnostics | FileCheck %s

// Arithmetic on the right-hand side computes an argument from values the rule
// matched, as `2 * p + 1` computes a quadrature order from a polynomial order.
// A literal operand takes the type of the other operand, so the arithmetic is
// done in p's type.

module {
  tessera.define private @lib.f(f64, i32) -> f64 attributes {argModes = [unit, unit], pure = true}
  tessera.define private @lib.g(f64, i32) -> f64 attributes {argModes = [unit, unit], pure = true}

  tessera.optimizations {
    tessera.optimization "lib.f(x, p) -> lib.g(x, 2 * p + 1)"
  }

  // The driver puts constants first and on the right of a commutative op.
  // CHECK-LABEL: llvm.func @linear
  // CHECK-DAG: %[[TWO:.*]] = llvm.mlir.constant(2 : i32) : i32
  // CHECK-DAG: %[[ONE:.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK: %[[M:.*]] = llvm.mul %arg1, %[[TWO]] : i32
  // CHECK: %[[A:.*]] = llvm.add %[[M]], %[[ONE]] : i32
  // CHECK: tessera.call @lib.g(%arg0, %[[A]])
  // CHECK-NOT: tessera.call @lib.f
  llvm.func @linear(%x: f64, %p: i32) -> f64 {
    %0 = tessera.call @lib.f(%x, %p) : (f64, i32) -> f64
    llvm.return %0 : f64
  }
}

// -----

// C's precedence, left-to-right grouping, and parentheses. `p -1` is a
// subtraction, as in C, although the lexer reads `-1` as a signed number.
module {
  tessera.define private @lib.f(f64, i32) -> f64 attributes {argModes = [unit, unit], pure = true}
  tessera.define private @lib.g(f64, i32) -> f64 attributes {argModes = [unit, unit], pure = true}

  tessera.optimizations {
    tessera.optimization "lib.f(x, p) -> lib.g(x, p -1 - p * (p + 1))"
  }

  // CHECK-LABEL: llvm.func @grouping
  // CHECK: %[[S:.*]] = llvm.sub %arg1, %{{.*}} : i32
  // CHECK: %[[A:.*]] = llvm.add %arg1, %{{.*}} : i32
  // CHECK: %[[M:.*]] = llvm.mul %arg1, %[[A]] : i32
  // CHECK: %[[R:.*]] = llvm.sub %[[S]], %[[M]] : i32
  // CHECK: tessera.call @lib.g(%arg0, %[[R]])
  llvm.func @grouping(%x: f64, %p: i32) -> f64 {
    %0 = tessera.call @lib.f(%x, %p) : (f64, i32) -> f64
    llvm.return %0 : f64
  }
}

// -----

// Float arithmetic, and a call's result as an operand.
module {
  tessera.define private @lib.f(f64) -> f64 attributes {argModes = [unit], pure = true}
  tessera.define private @lib.sq(f64) -> f64 attributes {argModes = [unit], pure = true}

  tessera.optimizations {
    tessera.optimization "lib.f(x) -> lib.sq(x) * 0.5 / x"
  }

  // CHECK-LABEL: llvm.func @floats
  // CHECK-DAG: %[[H:.*]] = llvm.mlir.constant(5.000000e-01 : f64) : f64
  // CHECK-DAG: %[[S:.*]] = tessera.call @lib.sq(%arg0)
  // CHECK: %[[M:.*]] = llvm.fmul %[[S]], %[[H]] : f64
  // CHECK: %[[D:.*]] = llvm.fdiv %[[M]], %arg0 : f64
  // CHECK: llvm.return %[[D]]
  llvm.func @floats(%x: f64) -> f64 {
    %0 = tessera.call @lib.f(%x) : (f64) -> f64
    llvm.return %0 : f64
  }
}

// -----

// Arithmetic in a condition. A comparison of constants is settled at compile
// time; otherwise the guard carries the condition as text that parses back to
// the same tree, parentheses included.
module {
  tessera.define private @lib.f(f64, i32) -> f64 attributes {argModes = [unit, unit], pure = true}
  tessera.define private @lib.g(f64, i32) -> f64 attributes {argModes = [unit, unit], pure = true}

  tessera.optimizations {
    tessera.optimization "if (p + 1) * 2 == 8, lib.f(x, p) -> lib.g(x, p)"
  }

  // CHECK-LABEL: llvm.func @settled_true
  // CHECK-NOT: tessera.guard
  // CHECK: tessera.call @lib.g
  llvm.func @settled_true(%x: f64) -> f64 {
    %p = llvm.mlir.constant(3 : i32) : i32
    %0 = tessera.call @lib.f(%x, %p) : (f64, i32) -> f64
    llvm.return %0 : f64
  }

  // CHECK-LABEL: llvm.func @settled_false
  // CHECK-NOT: tessera.guard
  // CHECK-NOT: lib.g
  llvm.func @settled_false(%x: f64) -> f64 {
    %p = llvm.mlir.constant(2 : i32) : i32
    // expected-remark @+1 {{its condition is false here}}
    %0 = tessera.call @lib.f(%x, %p) : (f64, i32) -> f64
    llvm.return %0 : f64
  }

  // CHECK-LABEL: llvm.func @guarded
  // CHECK: tessera.guard "(p + 1) * 2 == 8"
  llvm.func @guarded(%x: f64, %p: i32) -> f64 {
    %0 = tessera.call @lib.f(%x, %p) : (f64, i32) -> f64
    llvm.return %0 : f64
  }
}

// -----

// Operands of different types are not converted.
module {
  tessera.define private @lib.f(f64, i32) -> f64 attributes {argModes = [unit, unit], pure = true}
  tessera.define private @lib.g(f64, i32) -> f64 attributes {argModes = [unit, unit], pure = true}

  tessera.optimizations {
    tessera.optimization "lib.f(x, p) -> lib.g(x, p + x)"
  }

  // CHECK-LABEL: llvm.func @mixed
  // CHECK: tessera.call @lib.f
  llvm.func @mixed(%x: f64, %p: i32) -> f64 {
    // expected-warning @+1 {{operands of '+' have different types (i32 and f64)}}
    %0 = tessera.call @lib.f(%x, %p) : (f64, i32) -> f64
    llvm.return %0 : f64
  }
}

// -----

// Nor is a literal zero divided by.
module {
  tessera.define private @lib.f(f64, i32) -> f64 attributes {argModes = [unit, unit], pure = true}
  tessera.define private @lib.g(f64, i32) -> f64 attributes {argModes = [unit, unit], pure = true}

  tessera.optimizations {
    tessera.optimization "lib.f(x, p) -> lib.g(x, p % 0)"
  }

  // CHECK-LABEL: llvm.func @by_zero
  // CHECK: tessera.call @lib.f
  llvm.func @by_zero(%x: f64, %p: i32) -> f64 {
    // expected-warning @+1 {{'p % 0' divides by zero}}
    %0 = tessera.call @lib.f(%x, %p) : (f64, i32) -> f64
    llvm.return %0 : f64
  }
}

// -----

// There is nothing for arithmetic to match on the left.
module {
  tessera.optimizations {
    // expected-warning @+1 {{arithmetic can only be used on the right-hand side or in a condition of an optimization rule, not in what it matches}}
    tessera.optimization "lib.f(x, p + 1) -> lib.g(x, p)"
  }
}

// -----

// Or to test in a predicate.
module {
  tessera.optimizations {
    // expected-warning @+1 {{arithmetic cannot be used in an argument of predicate 'power_of_two'}}
    tessera.optimization "if power_of_two(p + 1), lib.f(x, p) -> lib.g(x, p)"
  }
}
