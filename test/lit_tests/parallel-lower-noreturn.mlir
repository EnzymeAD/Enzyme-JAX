// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(parallel-lower)" | FileCheck %s

// The inlined callee holds device intrinsics, so parallel-lower wraps the
// call in an execute_region that yields the call's results -- but this
// callee never returns, so inlining leaves those results with nothing to
// replace them and the yield with nothing to forward. Control cannot reach
// the yield; its operands are said as poison instead of erasing the call
// out from under its uses.

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
// CHECK: %[[P:.+]] = ub.poison : i32
// CHECK: scf.execute_region
// CHECK: llvm.call @abort()
// CHECK-NEXT: scf.yield %[[P]] : i32
