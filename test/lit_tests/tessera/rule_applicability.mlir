// RUN: enzymexlamlir-opt %s -parse-optimization-rules -tessera-apply-pdl -split-input-file -verify-diagnostics | FileCheck %s

// When a rule matches but must not, or cannot, be applied. Each of these used
// to fail the whole module, recurse at run time, or do nothing without saying
// so.

// The replacement has no tessera.define -- typically an inline helper nothing
// in this translation unit caused to be emitted. Building the call anyway
// would fail verification of the whole module, so the rule is skipped and
// that is reported: otherwise it silently never fires.
module {
  tessera.define private @lib.invert(f64) -> f64 attributes {argModes = [unit], pure = false}

  tessera.optimizations {
    tessera.optimization "lib.invert(x) -> lib.calc_inverse(x)"
  }

  // CHECK-LABEL: llvm.func @missing_helper
  // CHECK-NEXT: tessera.call @lib.invert(%arg0)
  llvm.func @missing_helper(%x: f64) -> f64 {
    // expected-warning @+1 {{optimization rule 'lib.invert(x) -> lib.calc_inverse(x)' cannot be applied: its right-hand side calls 'lib.calc_inverse', which has no tessera.define in this module}}
    %0 = tessera.call @lib.invert(%x) : (f64) -> f64
    llvm.return %0 : f64
  }
}

// -----

// The two sides disagree about how an argument is passed: the matched call
// takes the matrix as a loaded value, the replacement as a plain pointer.
module {
  tessera.define private @lib.foo(!llvm.ptr) -> f32 attributes {argModes = [{dir = #tessera.dir<in>, type = !llvm.struct<(array<4 x f32>)>}], pure = true}
  tessera.define private @lib.bar(!llvm.ptr) -> f32 attributes {argModes = [unit], pure = true}

  tessera.optimizations {
    tessera.optimization "lib.foo(m) -> lib.bar(m)"
  }

  // CHECK-LABEL: llvm.func @mode_mismatch
  // CHECK-NEXT: tessera.call @lib.foo(%arg0)
  llvm.func @mode_mismatch(%m: !llvm.struct<(array<4 x f32>)>) -> f32 {
    // expected-warning @+1 {{cannot be applied: argument 0 of 'lib.bar' expects !llvm.ptr, but the rule passes it a value of type !llvm.struct<(array<4 x f32>)>}}
    %0 = tessera.call @lib.foo(%m) : (!llvm.struct<(array<4 x f32>)>) -> f32
    llvm.return %0 : f32
  }
}

// -----

// The replacement produces something other than what it replaces.
module {
  tessera.define private @lib.foo(f32) -> f32 attributes {argModes = [unit], pure = true}
  tessera.define private @lib.bar(f32) -> f64 attributes {argModes = [unit], pure = true}

  tessera.optimizations {
    tessera.optimization "lib.foo(x) -> lib.bar(x)"
  }

  llvm.func @result_mismatch(%x: f32) -> f32 {
    // expected-warning @+1 {{cannot be applied: 'lib.bar' produces (f64), but the call it replaces produces (f32)}}
    %0 = tessera.call @lib.foo(%x) : (f32) -> f32
    llvm.return %0 : f32
  }
}

// -----

// A literal that does not fit the parameter it is passed to, signed or
// unsigned. (3000000000 would fit an i32 as an unsigned bit pattern.)
module {
  tessera.define private @lib.foo(i32) -> i32 attributes {argModes = [unit], pure = true}
  tessera.define private @lib.shift32(i32, i32) -> i32 attributes {argModes = [unit, unit], pure = true}

  tessera.optimizations {
    tessera.optimization "lib.foo(x) -> lib.shift32(x, 5000000000)"
  }

  llvm.func @literal_too_wide(%x: i32) -> i32 {
    // expected-warning @+1 {{cannot be applied: literal 5000000000 passed as argument 1 of 'lib.shift32' does not fit in its parameter type}}
    %0 = tessera.call @lib.foo(%x) : (i32) -> i32
    llvm.return %0 : i32
  }
}

// -----

// A replacement's fallback path commonly calls the very op it replaces.
// Rewriting that call would turn it into a call to the replacement itself,
// recursing forever on exactly the inputs the fallback is there for, so no
// rule touches the body of any op some rule rewrites onto.
module {
  tessera.define @lib.invert(%a: f64) -> f64 attributes {argModes = [unit], pure = false} {
    tessera.return %a : f64
  }
  tessera.define @lib.calc_inverse(%a: f64) -> f64 attributes {argModes = [unit], pure = false} {
    %0 = tessera.call @lib.invert(%a) : (f64) -> f64
    tessera.return %0 : f64
  }

  tessera.optimizations {
    tessera.optimization "lib.invert(x) -> lib.calc_inverse(x)"
  }

  // CHECK: tessera.define @lib.calc_inverse
  // CHECK-NEXT: tessera.call @lib.invert(%arg0)

  // Everywhere else the rule applies as usual.
  // CHECK-LABEL: llvm.func @caller
  // CHECK-NEXT: tessera.call @lib.calc_inverse(%arg0)
  llvm.func @caller(%x: f64) -> f64 {
    %0 = tessera.call @lib.invert(%x) : (f64) -> f64
    llvm.return %0 : f64
  }
}

// -----

// The same protection, asked for explicitly: lift-tessera-annotations puts
// tessera.no_rewrite on a function annotated tessera_no_rewrite.
module {
  tessera.define private @lib.invert(f64) -> f64 attributes {argModes = [unit], pure = false}
  tessera.define private @lib.calc_inverse(f64) -> f64 attributes {argModes = [unit], pure = false}

  tessera.optimizations {
    tessera.optimization "lib.invert(x) -> lib.calc_inverse(x)"
  }

  // CHECK-LABEL: llvm.func @fallback
  // CHECK-NEXT: tessera.call @lib.invert(%arg0)
  llvm.func @fallback(%x: f64) -> f64 attributes {tessera.no_rewrite} {
    %0 = tessera.call @lib.invert(%x) : (f64) -> f64
    llvm.return %0 : f64
  }
}

// -----

// A name the right-hand side uses but the left does not bind has no value to
// stand for. That is a mistake in the rule, reported when it is parsed.
module {
  tessera.optimizations {
    // expected-error @+1 {{optimization rule uses 'y' on the right-hand side, but it is not bound on the left}}
    tessera.optimization "lib.foo(x) -> lib.bar(x, y)"
  }
}
