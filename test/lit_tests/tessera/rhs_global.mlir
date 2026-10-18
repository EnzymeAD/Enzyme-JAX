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

// There is nothing to bind a global to in what a rule matches.
module {
  tessera.optimizations {
    // expected-warning @+1 {{a global, '@g', can only be used on the right-hand side of an optimization rule}}
    tessera.optimization "lib.f(@g) -> lib.g(1)"
  }
}

// -----

// Or in a condition.
module {
  tessera.optimizations {
    // expected-warning @+1 {{a global, '@g', can only be used on the right-hand side of an optimization rule}}
    tessera.optimization "if lib.h(@g) == 1, lib.f(x) -> lib.g(x)"
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
