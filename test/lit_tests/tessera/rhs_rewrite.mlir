// RUN: enzymexlamlir-opt %s -parse-optimization-rules -tessera-apply-pdl -split-input-file | FileCheck %s

// The right-hand side of a rule is built once the callees' tessera.defines
// exist, so each call it builds takes its result types from its callee, and
// each literal takes its type from where it is used.

// A call nested on the right-hand side. PDL cannot build one of these
// declaratively, since nothing in the pattern says what type it produces.
module {
  tessera.define private @lib.norm(f64) -> f64 attributes {argModes = [unit], pure = true}
  tessera.define private @lib.dot(f64, f64) -> f64 attributes {argModes = [unit, unit], pure = true}
  tessera.define private @lib.sqrt(f64) -> f64 attributes {argModes = [unit], pure = true}

  tessera.optimizations {
    tessera.optimization "lib.norm(x) -> lib.sqrt(lib.dot(x, x))"
  }

  // CHECK-LABEL: llvm.func @nested
  // CHECK-NEXT: %[[D:.*]] = tessera.call @lib.dot(%arg0, %arg0) : (f64, f64) -> f64
  // CHECK-NEXT: %[[S:.*]] = tessera.call @lib.sqrt(%[[D]]) : (f64) -> f64
  // CHECK-NEXT: llvm.return %[[S]]
  llvm.func @nested(%x: f64) -> f64 {
    %0 = tessera.call @lib.norm(%x) : (f64) -> f64
    llvm.return %0 : f64
  }
}

// -----

// An integer literal takes the type of the parameter it is passed to, not the
// narrowest type its magnitude fits in.
module {
  tessera.define private @lib.twice(i64) -> i64 attributes {argModes = [unit], pure = true}
  tessera.define private @lib.shift(i64, i64) -> i64 attributes {argModes = [unit, unit], pure = true}

  tessera.optimizations {
    tessera.optimization "lib.twice(x) -> lib.shift(x, 1)"
  }

  // CHECK-LABEL: llvm.func @int_literal
  // CHECK-NEXT: %[[C:.*]] = llvm.mlir.constant(1 : i64) : i64
  // CHECK-NEXT: tessera.call @lib.shift(%arg0, %[[C]]) : (i64, i64) -> i64
  llvm.func @int_literal(%x: i64) -> i64 {
    %0 = tessera.call @lib.twice(%x) : (i64) -> i64
    llvm.return %0 : i64
  }
}

// -----

// Likewise a float literal: 0.5 is exact in f32, but the parameter is f64.
module {
  tessera.define private @lib.half(f64) -> f64 attributes {argModes = [unit], pure = true}
  tessera.define private @lib.mulf(f64, f64) -> f64 attributes {argModes = [unit, unit], pure = true}

  tessera.optimizations {
    tessera.optimization "lib.half(x) -> lib.mulf(x, 0.5)"
  }

  // CHECK-LABEL: llvm.func @float_literal
  // CHECK-NEXT: %[[C:.*]] = llvm.mlir.constant(5.000000e-01 : f64) : f64
  // CHECK-NEXT: tessera.call @lib.mulf(%arg0, %[[C]]) : (f64, f64) -> f64
  llvm.func @float_literal(%x: f64) -> f64 {
    %0 = tessera.call @lib.half(%x) : (f64) -> f64
    llvm.return %0 : f64
  }
}

// -----

// A literal replacing the matched call takes that call's result type.
module {
  tessera.define private @lib.count_zero(i64) -> i64 attributes {argModes = [unit], pure = true}

  tessera.optimizations {
    tessera.optimization "lib.count_zero(x) -> 0"
  }

  // CHECK-LABEL: llvm.func @literal_root
  // CHECK-NEXT: %[[C:.*]] = llvm.mlir.constant(0 : i64) : i64
  // CHECK-NEXT: llvm.return %[[C]]
  llvm.func @literal_root(%x: i64) -> i64 {
    %0 = tessera.call @lib.count_zero(%x) : (i64) -> i64
    llvm.return %0 : i64
  }
}

// -----

// A callee with a written argument returns that argument's value as a leading
// result, which its function_type does not show. The guarded path used to
// build the call from the function_type alone and so get the results wrong.
module {
  tessera.define private @lib.inv(!llvm.ptr, !llvm.ptr) attributes {argModes = [{dir = #tessera.dir<in>, type = f64}, {dir = #tessera.dir<out>, type = f64}], pure = true}
  tessera.define private @lib.inv_fast(!llvm.ptr, !llvm.ptr) attributes {argModes = [{dir = #tessera.dir<in>, type = f64}, {dir = #tessera.dir<out>, type = f64}], pure = true}

  tessera.optimizations {
    tessera.optimization "if x > 0.0, lib.inv(x) -> lib.inv_fast(x)"
  }

  // CHECK-LABEL: llvm.func @written_argument
  // CHECK: tessera.guard "x > 0.0"
  // CHECK-NEXT: %[[F:.*]] = tessera.call @lib.inv_fast(%arg0) : (f64) -> f64
  // CHECK-NEXT: tessera.yield %[[F]] : f64
  llvm.func @written_argument(%x: f64) -> f64 {
    %0 = tessera.call @lib.inv(%x) : (f64) -> f64
    llvm.return %0 : f64
  }
}
