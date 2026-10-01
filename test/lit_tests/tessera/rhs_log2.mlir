// RUN: enzymexlamlir-opt %s -parse-optimization-rules -tessera-propagate-properties -tessera-apply-pdl -split-input-file -verify-diagnostics | FileCheck %s

// log2(n) on the right-hand side gives the exponent of a power of two, for a
// library function that takes a shift count where the matched call took the
// divisor. It is what MPFR's own mpfr_div_ui macro computes with
// __builtin_ctzl. On a constant it folds, so a rule applied to a literal
// leaves nothing to compute at run time; otherwise it is a count of trailing
// zeros, which the rule's condition makes exact.

module {
  tessera.define private @lib.div_ui(f64, i64) -> f64 attributes {argModes = [unit, unit], pure = true}
  tessera.define private @lib.div_2ui(f64, i64) -> f64 attributes {argModes = [unit, unit], pure = true}

  tessera.optimizations {
    tessera.optimization "if power_of_two(n), lib.div_ui(x, n) -> lib.div_2ui(x, log2(n))"
  }

  // Both the condition and the exponent are settled at compile time.
  // CHECK-LABEL: llvm.func @constant
  // CHECK-NOT: tessera.guard
  // CHECK-NOT: llvm.intr.cttz
  // CHECK: %[[K:.*]] = llvm.mlir.constant(5 : i64) : i64
  // CHECK-NEXT: %[[R:.*]] = tessera.call @lib.div_2ui(%arg0, %[[K]])
  // CHECK-NEXT: llvm.return %[[R]]
  llvm.func @constant(%x: f64) -> f64 {
    %n = llvm.mlir.constant(32 : i64) : i64
    %0 = tessera.call @lib.div_ui(%x, %n) : (f64, i64) -> f64
    llvm.return %0 : f64
  }

  // A constant that is not a power of two keeps the original call.
  // CHECK-LABEL: llvm.func @constant_not_power_of_two
  // CHECK-NOT: tessera.guard
  // CHECK-NOT: lib.div_2ui
  // CHECK: tessera.call @lib.div_ui
  llvm.func @constant_not_power_of_two(%x: f64) -> f64 {
    %n = llvm.mlir.constant(24 : i64) : i64
    // expected-remark @+1 {{its condition is false here}}
    %0 = tessera.call @lib.div_ui(%x, %n) : (f64, i64) -> f64
    llvm.return %0 : f64
  }

  // Otherwise the exponent is computed inside the guarded path, only once the
  // check has passed.
  // CHECK-LABEL: llvm.func @dynamic
  // CHECK: tessera.guard "power_of_two(n)"
  // CHECK-NEXT: %[[K:.*]] = "llvm.intr.cttz"(%arg1) <{is_zero_poison = false}> : (i64) -> i64
  // CHECK-NEXT: %[[R:.*]] = tessera.call @lib.div_2ui(%arg0, %[[K]])
  // CHECK-NEXT: tessera.yield %[[R]]
  // CHECK-NEXT: } else {
  // CHECK-NEXT: tessera.call @lib.div_ui(%arg0, %arg1)
  llvm.func @dynamic(%x: f64, %n: i64) -> f64 {
    %0 = tessera.call @lib.div_ui(%x, %n) : (f64, i64) -> f64
    llvm.return %0 : f64
  }
}

// -----

// log2 of a value that is not an integer cannot be built, and is reported.
module {
  tessera.define private @lib.scale(f64, f64) -> f64 attributes {argModes = [unit, unit], pure = true}
  tessera.define private @lib.shift(f64, i64) -> f64 attributes {argModes = [unit, unit], pure = true}

  tessera.optimizations {
    tessera.optimization "lib.scale(x, y) -> lib.shift(x, log2(y))"
  }

  // CHECK-LABEL: llvm.func @not_an_integer
  // CHECK: tessera.call @lib.scale
  llvm.func @not_an_integer(%x: f64, %y: f64) -> f64 {
    // expected-warning @+1 {{log2 takes an integer, but is given a value of type 'f64'}}
    %0 = tessera.call @lib.scale(%x, %y) : (f64, f64) -> f64
    llvm.return %0 : f64
  }
}

// -----

// log2 is the only built-in function.
module {
  tessera.optimizations {
    // expected-warning @+1 {{unknown function 'log10' in optimization rule}}
    tessera.optimization "lib.f(x, n) -> lib.g(x, log10(n))"
  }
}

// -----

// It computes a value for the right-hand side; there is nothing for it to
// match on the left.
module {
  tessera.optimizations {
    // expected-warning @+1 {{'log2(...)' can only be used on the right-hand side of an optimization rule}}
    tessera.optimization "lib.f(x, log2(n)) -> lib.g(x, n)"
  }
}

// -----

// Nor is it a call that can be made for its effect.
module {
  tessera.optimizations {
    // expected-warning @+1 {{'log2(n)' is followed by ';' in an optimization rule, but only a call can be}}
    tessera.optimization "lib.f(x, n) -> log2(n); lib.g(x, n)"
  }
}
