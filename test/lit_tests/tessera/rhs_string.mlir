// RUN: enzymexlamlir-opt %s -parse-optimization-rules -tessera-apply-pdl -split-input-file -verify-diagnostics | FileCheck %s

// A string literal on the right-hand side is passed as C passes one: the
// address of a NUL-terminated constant. It is how a library that names things
// by string, as PETSc names its solvers, is told which one to use.

// The string becomes a private constant global, shared by every rewrite that
// passes it, and the call is given its address.
module {
  tessera.define private @lib.solve(!llvm.ptr) -> i32 attributes {argModes = [unit], pure = false}
  tessera.define private @lib.set_type(!llvm.ptr, !llvm.ptr) -> i32 attributes {argModes = [unit, unit], pure = false}

  tessera.optimizations {
    tessera.optimization "lib.solve(k) -> lib.set_type(k, 'cg'); lib.solve(k)"
  }

  // CHECK: llvm.mlir.global private constant @[[S:__tessera_str_[0-9A-F]+]]("cg\00")
  // CHECK-NOT: llvm.mlir.global

  // CHECK-LABEL: llvm.func @two_sites
  // CHECK-NEXT: %[[A:.*]] = llvm.mlir.addressof @[[S]] : !llvm.ptr
  // CHECK-NEXT: tessera.call @lib.set_type(%arg0, %[[A]]) : (!llvm.ptr, !llvm.ptr) -> i32
  // CHECK-NEXT: tessera.call @lib.solve(%arg0)
  // CHECK-NEXT: tessera.call @lib.set_type(%arg1, %[[A]]) : (!llvm.ptr, !llvm.ptr) -> i32
  // CHECK-NEXT: tessera.call @lib.solve(%arg1)
  llvm.func @two_sites(%a: !llvm.ptr, %b: !llvm.ptr) -> i32 {
    %0 = tessera.call @lib.solve(%a) : (!llvm.ptr) -> i32
    %1 = tessera.call @lib.solve(%b) : (!llvm.ptr) -> i32
    llvm.return %1 : i32
  }
}

// -----

// A string passed where the callee does not take a pointer is reported, and
// the rule left out.
module {
  tessera.define private @lib.f(i64) -> i64 attributes {argModes = [unit], pure = true}
  tessera.define private @lib.g(i64, i32) -> i64 attributes {argModes = [unit, unit], pure = true}

  tessera.optimizations {
    tessera.optimization "lib.f(x) -> lib.g(x, 'cg')"
  }

  // CHECK-LABEL: llvm.func @not_a_pointer
  // CHECK-NEXT: tessera.call @lib.f(%arg0)
  llvm.func @not_a_pointer(%x: i64) -> i64 {
    // expected-warning @+1 {{cannot be applied: argument 1 of 'lib.g' expects i32, but the rule passes it a value of type !llvm.ptr}}
    %0 = tessera.call @lib.f(%x) : (i64) -> i64
    llvm.return %0 : i64
  }
}

// -----

// There is nothing to match a string against on the left-hand side.
module {
  tessera.optimizations {
    // expected-warning @+1 {{a string can only be used on the right-hand side of an optimization rule}}
    tessera.optimization "lib.f(x, 'cg') -> lib.g(x)"
  }
}

// -----

// Nor anything to compare one with in a condition.
module {
  tessera.optimizations {
    // expected-warning @+1 {{a string can only be used on the right-hand side of an optimization rule}}
    tessera.optimization "if x == 'cg', lib.f(x) -> lib.g(x)"
  }
}

// -----

// A string that is never closed.
module {
  tessera.optimizations {
    // expected-warning @+1 {{unterminated string in optimization rule}}
    tessera.optimization "lib.f(x) -> lib.g(x, 'cg)"
  }
}
