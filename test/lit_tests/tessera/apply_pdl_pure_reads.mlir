// RUN: enzymexlamlir-opt %s -parse-optimization-rules -tessera-propagate-properties -tessera-apply-pdl -split-input-file | FileCheck %s

// A pure op reads the object its handle names, so a rule matching a chain
// through it does not move it past a write to that object.

module {
  tessera.define private @lib.g(!llvm.ptr) -> !llvm.ptr attributes {argModes = [unit], pure = true}
  tessera.define private @lib.f(!llvm.ptr, !llvm.ptr) -> f64 attributes {argModes = [unit, unit], pure = false}
  tessera.define private @lib.fast(!llvm.ptr, !llvm.ptr) -> f64 attributes {argModes = [unit, unit], pure = false}
  llvm.func @mat_create(!llvm.ptr) -> i32
  llvm.func @fill_coo(!llvm.ptr {tessera.establishes = ["SPD"]}, !llvm.ptr) -> i32
  llvm.func @mat_scale(!llvm.ptr, f64) -> i32

  tessera.optimizations {
    tessera.optimization "if SPD(a), lib.f(lib.g(a), b) -> lib.fast(a, b)"
  }

  // CHECK-LABEL: llvm.func @write_between
  // CHECK-NOT: tessera.call @lib.fast
  // CHECK: llvm.return
  llvm.func @write_between(%x: !llvm.ptr, %b: !llvm.ptr, %s: f64) -> f64 {
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %slot = llvm.alloca %c1 x !llvm.ptr : (i32) -> !llvm.ptr
    %e0 = llvm.call @mat_create(%slot) : (!llvm.ptr) -> i32
    %a0 = llvm.load %slot : !llvm.ptr -> !llvm.ptr
    %e1 = llvm.call @fill_coo(%a0, %x) : (!llvm.ptr, !llvm.ptr) -> i32
    %a1 = llvm.load %slot : !llvm.ptr -> !llvm.ptr
    %g = tessera.call @lib.g(%a1) : (!llvm.ptr) -> !llvm.ptr
    %a2 = llvm.load %slot : !llvm.ptr -> !llvm.ptr
    %e2 = llvm.call @mat_scale(%a2, %s) : (!llvm.ptr, f64) -> i32
    %r = tessera.call @lib.f(%g, %b) : (!llvm.ptr, !llvm.ptr) -> f64
    llvm.return %r : f64
  }

  // CHECK-LABEL: llvm.func @no_write_between
  // CHECK: tessera.call @lib.fast
  llvm.func @no_write_between(%x: !llvm.ptr, %b: !llvm.ptr) -> f64 {
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %slot = llvm.alloca %c1 x !llvm.ptr : (i32) -> !llvm.ptr
    %e0 = llvm.call @mat_create(%slot) : (!llvm.ptr) -> i32
    %a0 = llvm.load %slot : !llvm.ptr -> !llvm.ptr
    %e1 = llvm.call @fill_coo(%a0, %x) : (!llvm.ptr, !llvm.ptr) -> i32
    %a1 = llvm.load %slot : !llvm.ptr -> !llvm.ptr
    %g = tessera.call @lib.g(%a1) : (!llvm.ptr) -> !llvm.ptr
    %r = tessera.call @lib.f(%g, %b) : (!llvm.ptr, !llvm.ptr) -> f64
    llvm.return %r : f64
  }
}
