// RUN: enzymexlamlir-opt %s -tessera-lower-guards -split-input-file -verify-diagnostics | FileCheck %s

// Arithmetic in a guard's condition is computed ahead of the comparison, in
// the type of the value it is computed from; a query call can be an operand.

module {
  tessera.define private @lib.order(!llvm.ptr) -> i32 attributes {argModes = [unit], pure = true}
  tessera.define private @lib.f(!llvm.ptr, i32) -> f32 attributes {argModes = [unit, unit], pure = true}
  tessera.define private @lib.g(!llvm.ptr, i32) -> f32 attributes {argModes = [unit, unit], pure = true}

  // CHECK-LABEL: llvm.func @arith_compare
  llvm.func @arith_compare(%a: !llvm.ptr, %n: i32) -> f32 {
    // CHECK: %[[P:.*]] = tessera.call @lib.order(%arg0)
    // CHECK: %[[ONE:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK: %[[S:.*]] = llvm.add %[[P]], %[[ONE]] : i32
    // CHECK: %[[TWO:.*]] = llvm.mlir.constant(2 : i32) : i32
    // CHECK: %[[M:.*]] = llvm.mul %[[S]], %[[TWO]] : i32
    // CHECK: %[[C:.*]] = llvm.icmp "eq" %[[M]], %arg1 : i32
    // CHECK: llvm.cond_br %[[C]]
    %0 = tessera.guard "(lib.order(a) + 1) * 2 == n" args(%a, %n) {argNames = ["a", "n"]} : (!llvm.ptr, i32) -> f32 {
      %1 = tessera.call @lib.g(%a, %n) : (!llvm.ptr, i32) -> f32
      tessera.yield %1 : f32
    } else {
      %2 = tessera.call @lib.f(%a, %n) : (!llvm.ptr, i32) -> f32
      tessera.yield %2 : f32
    }
    llvm.return %0 : f32
  }
}

// -----

// Operands of different types are not converted; the guard keeps the
// original call.
module {
  tessera.define private @lib.f(f64, i32) -> f32 attributes {argModes = [unit, unit], pure = true}
  tessera.define private @lib.g(f64, i32) -> f32 attributes {argModes = [unit, unit], pure = true}

  // CHECK-LABEL: llvm.func @mixed
  // CHECK-NOT: tessera.call @lib.g
  // CHECK: tessera.call @lib.f
  llvm.func @mixed(%x: f64, %n: i32) -> f32 {
    // expected-warning @+1 {{operands of '*' in a condition have different types}}
    %0 = tessera.guard "x * n > 4" args(%x, %n) {argNames = ["x", "n"]} : (f64, i32) -> f32 {
      %1 = tessera.call @lib.g(%x, %n) : (f64, i32) -> f32
      tessera.yield %1 : f32
    } else {
      %2 = tessera.call @lib.f(%x, %n) : (f64, i32) -> f32
      tessera.yield %2 : f32
    }
    llvm.return %0 : f32
  }
}
