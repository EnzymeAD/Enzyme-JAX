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

  // CHECK: llvm.mlir.global private constant @[[$S:__tessera_str_[0-9A-F]+]]("cg\00")
  // CHECK-NOT: llvm.mlir.global

  // CHECK-LABEL: llvm.func @two_sites
  // CHECK-NEXT: %[[A:.*]] = llvm.mlir.addressof @[[$S]] : !llvm.ptr
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

// A string in what a rule matches matches a C string literal, the address of
// a constant holding that text, as PETSc's KSPSetType(ksp, KSPGMRES) passes
// "gmres". Text after the first NUL doesn't count, as C reads none of it.
module {
  llvm.mlir.global private unnamed_addr constant @".str"("gmres\00") {addr_space = 0 : i32}
  llvm.mlir.global private unnamed_addr constant @".str.1"("bcgs\00") {addr_space = 0 : i32}
  llvm.mlir.global private unnamed_addr constant @".str.2"("gmres\00\00") {addr_space = 0 : i32}
  llvm.mlir.global internal @buffer("gmres\00") {addr_space = 0 : i32}

  tessera.define private @lib.set_type(!llvm.ptr, !llvm.ptr) -> i32 attributes {argModes = [unit, unit], pure = false}

  tessera.optimizations {
    tessera.optimization "lib.set_type(k, 'gmres') -> lib.set_type(k, 'cg')"
  }

  // CHECK-LABEL: llvm.func @literal
  // CHECK: %[[A:.*]] = llvm.mlir.addressof @__tessera_str_{{[0-9A-F]+}} : !llvm.ptr
  // CHECK: tessera.call @lib.set_type(%arg0, %[[A]])
  llvm.func @literal(%k: !llvm.ptr) -> i32 {
    %s = llvm.mlir.addressof @".str" : !llvm.ptr
    %0 = tessera.call @lib.set_type(%k, %s) : (!llvm.ptr, !llvm.ptr) -> i32
    llvm.return %0 : i32
  }

  // CHECK-LABEL: llvm.func @padded
  // CHECK: llvm.mlir.addressof @__tessera_str_
  llvm.func @padded(%k: !llvm.ptr) -> i32 {
    %s = llvm.mlir.addressof @".str.2" : !llvm.ptr
    %0 = tessera.call @lib.set_type(%k, %s) : (!llvm.ptr, !llvm.ptr) -> i32
    llvm.return %0 : i32
  }

  // CHECK-LABEL: llvm.func @other_text
  // CHECK-NOT: __tessera_str_
  // CHECK: llvm.return
  llvm.func @other_text(%k: !llvm.ptr) -> i32 {
    %s = llvm.mlir.addressof @".str.1" : !llvm.ptr
    %0 = tessera.call @lib.set_type(%k, %s) : (!llvm.ptr, !llvm.ptr) -> i32
    llvm.return %0 : i32
  }

  // A global that could be written to by then is not a literal.
  // CHECK-LABEL: llvm.func @not_constant
  // CHECK-NOT: __tessera_str_
  // CHECK: llvm.return
  llvm.func @not_constant(%k: !llvm.ptr) -> i32 {
    %s = llvm.mlir.addressof @buffer : !llvm.ptr
    %0 = tessera.call @lib.set_type(%k, %s) : (!llvm.ptr, !llvm.ptr) -> i32
    llvm.return %0 : i32
  }

  // CHECK-LABEL: llvm.func @unknown
  // CHECK-NOT: __tessera_str_
  // CHECK: llvm.return
  llvm.func @unknown(%k: !llvm.ptr, %s: !llvm.ptr) -> i32 {
    %0 = tessera.call @lib.set_type(%k, %s) : (!llvm.ptr, !llvm.ptr) -> i32
    llvm.return %0 : i32
  }
}

// -----

// A condition compares a string by its text, as `if t == KSPGMRES` does once
// the clang plugin has put in PETSc's "gmres". There is no runtime check, so,
// as with a declared property like SPD, a pointer whose text the IR doesn't
// show doesn't satisfy it.
module {
  llvm.mlir.global private unnamed_addr constant @".str"("gmres\00") {addr_space = 0 : i32}
  llvm.mlir.global private unnamed_addr constant @".str.1"("bcgs\00") {addr_space = 0 : i32}

  tessera.define private @lib.set_type(!llvm.ptr, !llvm.ptr) -> i32 attributes {argModes = [unit, unit], pure = false}
  tessera.define private @lib.set_type_fast(!llvm.ptr) -> i32 attributes {argModes = [unit], pure = false}

  tessera.optimizations {
    tessera.optimization "if t == 'gmres', lib.set_type(k, t) -> lib.set_type_fast(k)"
  }

  // CHECK-LABEL: llvm.func @same_text
  // CHECK-NOT: tessera.guard
  // CHECK: tessera.call @lib.set_type_fast(%arg0)
  llvm.func @same_text(%k: !llvm.ptr) -> i32 {
    %t = llvm.mlir.addressof @".str" : !llvm.ptr
    %0 = tessera.call @lib.set_type(%k, %t) : (!llvm.ptr, !llvm.ptr) -> i32
    llvm.return %0 : i32
  }

  // CHECK-LABEL: llvm.func @other_text
  // CHECK: tessera.call @lib.set_type(
  llvm.func @other_text(%k: !llvm.ptr) -> i32 {
    %t = llvm.mlir.addressof @".str.1" : !llvm.ptr
    // expected-remark @+1 {{its condition is false here}}
    %0 = tessera.call @lib.set_type(%k, %t) : (!llvm.ptr, !llvm.ptr) -> i32
    llvm.return %0 : i32
  }

  // CHECK-LABEL: llvm.func @unknown_text
  // CHECK-NOT: tessera.guard
  // CHECK: tessera.call @lib.set_type(
  llvm.func @unknown_text(%k: !llvm.ptr, %t: !llvm.ptr) -> i32 {
    // expected-remark @+1 {{could not show t == 'gmres'}}
    %0 = tessera.call @lib.set_type(%k, %t) : (!llvm.ptr, !llvm.ptr) -> i32
    llvm.return %0 : i32
  }
}

// -----

// Text has no order.
module {
  tessera.optimizations {
    // expected-warning @+1 {{a string or global can only be compared with '==' or '!=', not '<'}}
    tessera.optimization "if x < 'cg', lib.f(x) -> lib.g(x)"
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
