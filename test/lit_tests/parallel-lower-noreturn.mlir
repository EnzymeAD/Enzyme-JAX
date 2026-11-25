// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(parallel-lower)" --split-input-file | FileCheck %s

// The inlined callee holds device intrinsics, so parallel-lower wraps the
// call in an execute_region that yields the call's results -- but this
// callee never returns and ends in llvm.unreachable, which the inliner
// preserves inside the execute_region.

module {
  llvm.func @abort()
  func.func private @dies() -> i32 {
    %t = gpu.thread_id x
    llvm.call @abort() : () -> ()
    llvm.unreachable
  }
  func.func @main() -> i32 {
    %v = func.call @dies() : () -> i32
    return %v : i32
  }
}

// CHECK-LABEL: func.func @main
// CHECK: scf.execute_region -> i32 {
// CHECK-NEXT: llvm.call @abort()
// CHECK-NEXT: llvm.unreachable
// CHECK-NEXT: }

// -----

// The multi-result form: the callee's unreachable carries no operands while
// the call carries two results, so the inliner preserves llvm.unreachable
// inside the execute_region.

module {
  llvm.func @abort()
  func.func private @dies2() -> (i32, f32) {
    %t = gpu.thread_id x
    llvm.call @abort() : () -> ()
    llvm.unreachable
  }
  func.func @main2() -> (i32, f32) {
    %v:2 = func.call @dies2() : () -> (i32, f32)
    return %v#0, %v#1 : i32, f32
  }
}

// CHECK-LABEL: func.func @main2
// CHECK: scf.execute_region -> (i32, f32) {
// CHECK-NEXT: llvm.call @abort()
// CHECK-NEXT: llvm.unreachable
// CHECK-NEXT: }
