// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(parallel-lower)" --split-input-file | FileCheck %s

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

// -----

// The multi-result form: the callee's unreachable carries no operands while
// the call carries two results, so the inliner's terminator handler must
// bound its replacement loop (debug builds asserted on the mismatch); the
// repair then fills every result with poison.

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
// CHECK-DAG: %[[PI:.+]] = ub.poison : i32
// CHECK-DAG: %[[PF:.+]] = ub.poison : f32
// CHECK: scf.execute_region
// CHECK: llvm.call @abort()
// CHECK-NEXT: scf.yield %[[PI]], %[[PF]] : i32, f32
