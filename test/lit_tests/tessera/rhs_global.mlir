// RUN: enzymexlamlir-opt %s -parse-optimization-rules -tessera-apply-pdl -split-input-file -verify-diagnostics | FileCheck %s

// A global on the right-hand side, `@sym`, passes the address of a global
// variable, as a library passes its own global object: MFEM's integration
// rules live in the global `mfem::IntRules`. The clang plugin writes `@sym`
// for a C++ name the rule uses, so a rule in a header never spells a mangled
// name.

module {
  llvm.mlir.global external @_ZN4mfem8IntRulesE() {addr_space = 0 : i32} : !llvm.struct<(i32, i32)>

  tessera.define private @lib.solve(!llvm.ptr) -> i32 attributes {argModes = [unit], pure = false}
  tessera.define private @lib.rules_get(!llvm.ptr, i32) -> !llvm.ptr attributes {argModes = [unit, unit], pure = true}
  tessera.define private @lib.set_rule(!llvm.ptr, !llvm.ptr) -> i32 attributes {argModes = [unit, unit], pure = false}

  tessera.optimizations {
    tessera.optimization "lib.solve(k) -> lib.set_rule(k, lib.rules_get(@_ZN4mfem8IntRulesE, 7)); lib.solve(k)"
  }

  // CHECK-LABEL: llvm.func @uses_global
  // CHECK: %[[G:.*]] = llvm.mlir.addressof @_ZN4mfem8IntRulesE : !llvm.ptr
  // CHECK: %[[R:.*]] = tessera.call @lib.rules_get(%[[G]], %{{.*}}) : (!llvm.ptr, i32) -> !llvm.ptr
  // CHECK: tessera.call @lib.set_rule(%arg0, %[[R]])
  // CHECK: tessera.call @lib.solve(%arg0)
  llvm.func @uses_global(%a: !llvm.ptr) -> i32 {
    %0 = tessera.call @lib.solve(%a) : (!llvm.ptr) -> i32
    llvm.return %0 : i32
  }
}

// -----

// A global the module does not have is reported, and the call kept.
module {
  tessera.define private @lib.solve(!llvm.ptr) -> i32 attributes {argModes = [unit], pure = false}
  tessera.define private @lib.set_rule(!llvm.ptr, !llvm.ptr) -> i32 attributes {argModes = [unit, unit], pure = false}

  tessera.optimizations {
    tessera.optimization "lib.solve(k) -> lib.set_rule(k, @missing); lib.solve(k)"
  }

  // CHECK-LABEL: llvm.func @missing_global
  // CHECK-NOT: lib.set_rule
  llvm.func @missing_global(%a: !llvm.ptr) -> i32 {
    // expected-warning @+1 {{global '@missing' is not in this module}}
    %0 = tessera.call @lib.solve(%a) : (!llvm.ptr) -> i32
    llvm.return %0 : i32
  }
}

// -----

// A global in what a rule matches matches its address, and no other.
module {
  llvm.mlir.global external @g() {addr_space = 0 : i32} : i32
  llvm.mlir.global external @h() {addr_space = 0 : i32} : i32

  tessera.define private @lib.f(!llvm.ptr) -> i32 attributes {argModes = [unit], pure = true}
  tessera.define private @lib.g(i32) -> i32 attributes {argModes = [unit], pure = true}

  tessera.optimizations {
    tessera.optimization "lib.f(@g) -> lib.g(1)"
  }

  // CHECK-LABEL: llvm.func @matched
  // CHECK: tessera.call @lib.g
  // CHECK-NOT: tessera.call @lib.f
  llvm.func @matched() -> i32 {
    %a = llvm.mlir.addressof @g : !llvm.ptr
    %0 = tessera.call @lib.f(%a) : (!llvm.ptr) -> i32
    llvm.return %0 : i32
  }

  // CHECK-LABEL: llvm.func @other_global
  // CHECK: tessera.call @lib.f
  llvm.func @other_global() -> i32 {
    %a = llvm.mlir.addressof @h : !llvm.ptr
    %0 = tessera.call @lib.f(%a) : (!llvm.ptr) -> i32
    llvm.return %0 : i32
  }

  // CHECK-LABEL: llvm.func @unknown
  // CHECK: tessera.call @lib.f
  llvm.func @unknown(%a: !llvm.ptr) -> i32 {
    %0 = tessera.call @lib.f(%a) : (!llvm.ptr) -> i32
    llvm.return %0 : i32
  }
}

// -----

// A condition compares a global by its address. The same global settles it.
// Two different globals may still share an address (an alias, or constants
// the linker merges), so otherwise the guard compares at run time.
module {
  llvm.mlir.global external @g() {addr_space = 0 : i32} : i32
  llvm.mlir.global external @h() {addr_space = 0 : i32} : i32

  tessera.define private @lib.f(!llvm.ptr) -> i32 attributes {argModes = [unit], pure = true}
  tessera.define private @lib.g(!llvm.ptr) -> i32 attributes {argModes = [unit], pure = true}

  tessera.optimizations {
    tessera.optimization "if r == @g, lib.f(r) -> lib.g(r)"
  }

  // CHECK-LABEL: llvm.func @same_global
  // CHECK-NOT: tessera.guard
  // CHECK: tessera.call @lib.g
  llvm.func @same_global() -> i32 {
    %r = llvm.mlir.addressof @g : !llvm.ptr
    %0 = tessera.call @lib.f(%r) : (!llvm.ptr) -> i32
    llvm.return %0 : i32
  }

  // CHECK-LABEL: llvm.func @other_global
  // CHECK: tessera.guard "r == @g"
  llvm.func @other_global() -> i32 {
    %r = llvm.mlir.addressof @h : !llvm.ptr
    %0 = tessera.call @lib.f(%r) : (!llvm.ptr) -> i32
    llvm.return %0 : i32
  }

  // CHECK-LABEL: llvm.func @unknown
  // CHECK: tessera.guard "r == @g"
  llvm.func @unknown(%r: !llvm.ptr) -> i32 {
    %0 = tessera.call @lib.f(%r) : (!llvm.ptr) -> i32
    llvm.return %0 : i32
  }
}

// -----

// Addresses have no order.
module {
  tessera.optimizations {
    // expected-warning @+1 {{a string or global can only be compared with '==' or '!=', not '>='}}
    tessera.optimization "if r >= @g, lib.f(r) -> lib.g(r)"
  }
}

// -----

// A bare '@' is not a global.
module {
  tessera.optimizations {
    // expected-warning @+1 {{unrecognized character '@' in optimization rule}}
    tessera.optimization "lib.f(x) -> lib.g(@)"
  }
}
