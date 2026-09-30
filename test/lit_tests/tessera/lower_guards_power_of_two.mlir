// RUN: enzymexlamlir-opt %s -tessera-lower-guards -split-input-file -verify-diagnostics | FileCheck %s

// power_of_two(n) on a value only known at run time becomes the bit test
// n != 0 && (n & (n - 1)) == 0, at the width of n.

module {
  tessera.define @lib.div_ui(%x: f64, %n: i64) -> f64 attributes {argModes = [unit, unit], pure = true} {
    tessera.return %x : f64
  }
  tessera.define @lib.shift(%x: f64, %n: i64) -> f64 attributes {argModes = [unit, unit], pure = true} {
    tessera.return %x : f64
  }
  tessera.define @lib.div_u32(%x: f64, %n: i32) -> f64 attributes {argModes = [unit, unit], pure = true} {
    tessera.return %x : f64
  }
  tessera.define @lib.shift_u32(%x: f64, %n: i32) -> f64 attributes {argModes = [unit, unit], pure = true} {
    tessera.return %x : f64
  }

  // CHECK-LABEL: llvm.func @runtime_check
  llvm.func @runtime_check(%x: f64, %n: i64) -> f64 {
    // CHECK: %[[Z:.*]] = llvm.mlir.constant(0 : i64) : i64
    // CHECK: %[[O:.*]] = llvm.mlir.constant(1 : i64) : i64
    // CHECK: %[[B:.*]] = llvm.sub %arg1, %[[O]] : i64
    // CHECK: %[[S:.*]] = llvm.and %arg1, %[[B]] : i64
    // CHECK: %[[NZ:.*]] = llvm.icmp "ne" %arg1, %[[Z]] : i64
    // CHECK: %[[ONE:.*]] = llvm.icmp "eq" %[[S]], %[[Z]] : i64
    // CHECK: %[[P:.*]] = llvm.and %[[NZ]], %[[ONE]] : i1
    // CHECK: llvm.cond_br %[[P]], ^[[THEN:.*]], ^[[ELSE:.*]]
    // CHECK: ^[[THEN]]:
    // CHECK: tessera.call @lib.shift
    // CHECK: ^[[ELSE]]:
    // CHECK: tessera.call @lib.div_ui
    %0 = tessera.guard "power_of_two(n)" args(%x, %n) {argNames = ["x", "n"]} : (f64, i64) -> f64 {
      %1 = tessera.call @lib.shift(%x, %n) : (f64, i64) -> f64
      tessera.yield %1 : f64
    } else {
      %2 = tessera.call @lib.div_ui(%x, %n) : (f64, i64) -> f64
      tessera.yield %2 : f64
    }
    llvm.return %0 : f64
  }

  // The constants follow the operand's width.
  // CHECK-LABEL: llvm.func @runtime_check_i32
  llvm.func @runtime_check_i32(%x: f64, %n: i32) -> f64 {
    // CHECK: %[[O:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK: llvm.sub %arg1, %[[O]] : i32
    // CHECK: llvm.cond_br
    %0 = tessera.guard "power_of_two(n)" args(%x, %n) {argNames = ["x", "n"]} : (f64, i32) -> f64 {
      %1 = tessera.call @lib.shift_u32(%x, %n) : (f64, i32) -> f64
      tessera.yield %1 : f64
    } else {
      %2 = tessera.call @lib.div_u32(%x, %n) : (f64, i32) -> f64
      tessera.yield %2 : f64
    }
    llvm.return %0 : f64
  }

  // It combines with the other conditions like any predicate.
  // CHECK-LABEL: llvm.func @with_compare
  llvm.func @with_compare(%x: f64, %n: i64) -> f64 {
    // CHECK: llvm.sub %arg1
    // CHECK: llvm.icmp "sgt" %arg1
    // CHECK: llvm.cond_br
    %0 = tessera.guard "power_of_two(n) && n > 1" args(%x, %n) {argNames = ["x", "n"]} : (f64, i64) -> f64 {
      %1 = tessera.call @lib.shift(%x, %n) : (f64, i64) -> f64
      tessera.yield %1 : f64
    } else {
      %2 = tessera.call @lib.div_ui(%x, %n) : (f64, i64) -> f64
      tessera.yield %2 : f64
    }
    llvm.return %0 : f64
  }

  // CHECK-NOT: tessera.guard
}

// -----

// Only an integer can be tested. Anything else keeps the original call.

module {
  tessera.define @lib.div(%x: f64, %y: f64) -> f64 attributes {argModes = [unit, unit], pure = true} {
    tessera.return %x : f64
  }
  tessera.define @lib.shift(%x: f64, %y: f64) -> f64 attributes {argModes = [unit, unit], pure = true} {
    tessera.return %x : f64
  }

  // CHECK-LABEL: llvm.func @not_an_integer
  llvm.func @not_an_integer(%x: f64, %y: f64) -> f64 {
    // CHECK-NOT: llvm.cond_br
    // CHECK: %[[R:.*]] = tessera.call @lib.div
    // CHECK-NEXT: llvm.return %[[R]]
    // expected-warning @+1 {{optimization rule not applied here, the original call is kept: predicate 'power_of_two' takes an integer, but got 'f64'}}
    %0 = tessera.guard "power_of_two(y)" args(%x, %y) {argNames = ["x", "y"]} : (f64, f64) -> f64 {
      %1 = tessera.call @lib.shift(%x, %y) : (f64, f64) -> f64
      tessera.yield %1 : f64
    } else {
      %2 = tessera.call @lib.div(%x, %y) : (f64, f64) -> f64
      tessera.yield %2 : f64
    }
    llvm.return %0 : f64
  }
}
