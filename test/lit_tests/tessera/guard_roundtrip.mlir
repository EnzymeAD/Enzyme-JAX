// RUN: enzymexlamlir-opt %s | enzymexlamlir-opt | FileCheck %s

// tessera.guard holds the two outcomes of a conditional optimization rule
// whose condition could not be proven statically: the specialized rewrite in
// the first region, the original computation in the second. These pin that it
// parses and prints back unchanged.

module {
  tessera.define @lib.foo(%arg0: f32) -> f32 attributes {byRefTypes = [unit], pure = true} {
    tessera.return %arg0 : f32
  }

  tessera.define @lib.symmetric_foo(%arg0: f32) -> f32 attributes {byRefTypes = [unit], pure = true} {
    tessera.return %arg0 : f32
  }

  // CHECK-LABEL: llvm.func @single_result
  llvm.func @single_result(%x: f32) -> f32 {
    // CHECK: %[[R:.*]] = tessera.guard "symmetric(x)" args(%arg0) {argNames = ["x"]} : (f32) -> f32 {
    // CHECK-NEXT: %[[S:.*]] = tessera.call @lib.symmetric_foo(%arg0)
    // CHECK-NEXT: tessera.yield %[[S]] : f32
    // CHECK-NEXT: } else {
    // CHECK-NEXT: %[[O:.*]] = tessera.call @lib.foo(%arg0)
    // CHECK-NEXT: tessera.yield %[[O]] : f32
    // CHECK-NEXT: }
    %0 = tessera.guard "symmetric(x)" args(%x) {argNames = ["x"]} : (f32) -> f32 {
      %1 = tessera.call @lib.symmetric_foo(%x) : (f32) -> f32
      tessera.yield %1 : f32
    } else {
      %2 = tessera.call @lib.foo(%x) : (f32) -> f32
      tessera.yield %2 : f32
    }
    // CHECK: llvm.return %[[R]]
    llvm.return %0 : f32
  }

  // A condition over several variables, resolved positionally through
  // argNames. The names in the condition need not match the SSA names.
  // CHECK-LABEL: llvm.func @multiple_args
  llvm.func @multiple_args(%a: f32, %b: f32) -> f32 {
    // CHECK: tessera.guard "symmetric(m) && n > 64" args(%arg0, %arg1) {argNames = ["m", "n"]} : (f32, f32) -> f32
    %0 = tessera.guard "symmetric(m) && n > 64" args(%a, %b) {argNames = ["m", "n"]} : (f32, f32) -> f32 {
      %1 = tessera.call @lib.symmetric_foo(%a) : (f32) -> f32
      tessera.yield %1 : f32
    } else {
      %2 = tessera.call @lib.foo(%a) : (f32) -> f32
      tessera.yield %2 : f32
    }
    llvm.return %0 : f32
  }

  // A guard producing no results at all: both regions yield nothing.
  // CHECK-LABEL: llvm.func @no_results
  llvm.func @no_results(%x: f32) {
    // CHECK: tessera.guard "diagonal(x)" args(%arg0) {argNames = ["x"]} : (f32) -> ()
    // CHECK: tessera.yield
    tessera.guard "diagonal(x)" args(%x) {argNames = ["x"]} : (f32) -> () {
      %0 = tessera.call @lib.symmetric_foo(%x) : (f32) -> f32
      tessera.yield
    } else {
      %1 = tessera.call @lib.foo(%x) : (f32) -> f32
      tessera.yield
    }
    llvm.return
  }

  // Several results from one guard.
  // CHECK-LABEL: llvm.func @multiple_results
  llvm.func @multiple_results(%x: f32) -> f32 {
    // CHECK: tessera.guard "symmetric(x)" args(%arg0) {argNames = ["x"]} : (f32) -> (f32, f32)
    %0:2 = tessera.guard "symmetric(x)" args(%x) {argNames = ["x"]} : (f32) -> (f32, f32) {
      %1 = tessera.call @lib.symmetric_foo(%x) : (f32) -> f32
      tessera.yield %1, %1 : f32, f32
    } else {
      %2 = tessera.call @lib.foo(%x) : (f32) -> f32
      tessera.yield %2, %2 : f32, f32
    }
    llvm.return %0#0 : f32
  }
}
