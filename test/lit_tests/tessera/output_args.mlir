// RUN: enzymexlamlir-opt %s -parse-optimization-rules -tessera-apply-pdl -split-input-file -verify-diagnostics | FileCheck %s

// A rule passes a call's C arguments, including one the function only writes
// (val=out), though a tessera.call has no operand for it. Such an argument
// names where the result goes.
module {
  tessera.define private @lib.inv(!llvm.ptr, !llvm.ptr) attributes {argModes = [{dir = #tessera.dir<in>, type = f64}, {dir = #tessera.dir<out>, type = f64}], pure = true}
  tessera.define private @lib.fast_inv(!llvm.ptr, !llvm.ptr) attributes {argModes = [{dir = #tessera.dir<in>, type = f64}, {dir = #tessera.dir<out>, type = f64}], pure = true}

  tessera.optimizations {
    tessera.optimization "lib.inv(a, b) -> lib.fast_inv(a, b)"
  }

  // CHECK-LABEL: llvm.func @output
  // CHECK-NEXT: %[[F:.*]] = tessera.call @lib.fast_inv(%arg0) : (f64) -> f64
  // CHECK-NEXT: llvm.return %[[F]]
  llvm.func @output(%a: f64) -> f64 {
    %0 = tessera.call @lib.inv(%a) : (f64) -> f64
    llvm.return %0 : f64
  }
}

// -----

// Once written, an output can be read and modified like any variable.
module {
  tessera.define private @lib.inv(!llvm.ptr, !llvm.ptr) attributes {argModes = [{dir = #tessera.dir<in>, type = f64}, {dir = #tessera.dir<out>, type = f64}], pure = true}
  tessera.define private @lib.fast_inv(!llvm.ptr, !llvm.ptr) attributes {argModes = [{dir = #tessera.dir<in>, type = f64}, {dir = #tessera.dir<out>, type = f64}], pure = true}
  tessera.define private @lib.scale(!llvm.ptr) attributes {argModes = [{dir = #tessera.dir<inout>, type = f64}], pure = true}

  tessera.optimizations {
    tessera.optimization "lib.inv(a, b) -> lib.fast_inv(a, b); lib.scale(b)"
  }

  // CHECK-LABEL: llvm.func @output_then_modify
  // CHECK-NEXT: %[[F:.*]] = tessera.call @lib.fast_inv(%arg0) : (f64) -> f64
  // CHECK-NEXT: %[[S:.*]] = tessera.call @lib.scale(%[[F]]) : (f64) -> f64
  // CHECK-NEXT: llvm.return %[[S]]
  llvm.func @output_then_modify(%a: f64) -> f64 {
    %0 = tessera.call @lib.inv(%a) : (f64) -> f64
    llvm.return %0 : f64
  }
}

// -----

// An output has no value until something writes it.
module {
  tessera.define private @lib.inv(!llvm.ptr, !llvm.ptr) attributes {argModes = [{dir = #tessera.dir<in>, type = f64}, {dir = #tessera.dir<out>, type = f64}], pure = true}
  tessera.define private @lib.fast_inv(!llvm.ptr, !llvm.ptr) attributes {argModes = [{dir = #tessera.dir<in>, type = f64}, {dir = #tessera.dir<out>, type = f64}], pure = true}
  tessera.define private @lib.scale(!llvm.ptr) attributes {argModes = [{dir = #tessera.dir<inout>, type = f64}], pure = true}
  tessera.define private @lib.other(f64) -> f64 attributes {argModes = [unit], pure = true}

  tessera.optimizations {
    tessera.optimization "lib.inv(a, b) -> lib.scale(b); lib.fast_inv(a, b)"
    tessera.optimization "lib.inv(a, b) -> lib.other(a)"
    tessera.optimization "lib.inv(a, b) -> lib.fast_inv(a, 1.0)"
  }

  // CHECK-LABEL: llvm.func @unwritten
  // CHECK-NEXT: tessera.call @lib.inv(%arg0)
  llvm.func @unwritten(%a: f64) -> f64 {
    // expected-warning @+3 {{optimization rule 'lib.inv(a, b) -> lib.scale(b); lib.fast_inv(a, b)' cannot be applied: its right-hand side reads 'b' before anything writes it}}
    // expected-warning @+2 {{optimization rule 'lib.inv(a, b) -> lib.other(a)' cannot be applied: the call it replaces writes 'b', but its right-hand side never does}}
    // expected-warning @+1 {{optimization rule 'lib.inv(a, b) -> lib.fast_inv(a, 1.0)' cannot be applied: argument 1 of 'lib.fast_inv' is where it writes its result, so the rule must pass a variable}}
    %0 = tessera.call @lib.inv(%a) : (f64) -> f64
    llvm.return %0 : f64
  }
}

// -----

// What a rule matches must pass every argument, and an output only names
// where the result goes.
module {
  tessera.define private @lib.inv(!llvm.ptr, !llvm.ptr) attributes {argModes = [{dir = #tessera.dir<in>, type = f64}, {dir = #tessera.dir<out>, type = f64}], pure = true}
  tessera.define private @lib.fast_inv(!llvm.ptr, !llvm.ptr) attributes {argModes = [{dir = #tessera.dir<in>, type = f64}, {dir = #tessera.dir<out>, type = f64}], pure = true}

  tessera.optimizations {
    // expected-warning @+1 {{optimization rule ignored: 'lib.inv' takes 2 argument(s), but the rule passes it 1}}
    tessera.optimization "lib.inv(a) -> lib.fast_inv(a)"
    // expected-warning @+1 {{optimization rule ignored: argument 1 of 'lib.inv' is where it writes its result, so the rule must pass a variable}}
    tessera.optimization "lib.inv(a, 0.0) -> lib.fast_inv(a, b)"
    // expected-warning @+1 {{optimization rule ignored: 'b' is where a call writes its result, so it can't also be matched or tested}}
    tessera.optimization "if b > 0.0, lib.inv(a, b) -> lib.fast_inv(a, b)"
    // expected-warning @+1 {{optimization rule ignored: 'a' is where a call writes its result, so it can't also be matched or tested}}
    tessera.optimization "lib.inv(a, a) -> lib.fast_inv(a, a)"
  }

  // CHECK-LABEL: llvm.func @bad_lhs
  // CHECK-NEXT: tessera.call @lib.inv(%arg0)
  llvm.func @bad_lhs(%a: f64) -> f64 {
    %0 = tessera.call @lib.inv(%a) : (f64) -> f64
    llvm.return %0 : f64
  }
}
