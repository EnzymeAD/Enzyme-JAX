// RUN: enzymexlamlir-opt %s --split-input-file --pass-pipeline='builtin.module(canonicalize-scf-for)' --verify-diagnostics | FileCheck %s

// __enzyme_set_checkpointing(schedule, budget) takes the schedules of
// enzyme/checkpoint_schedule.h, as Enzyme's Clang loop annotations
// ([[enzyme::checkpoint("binomial", 4)]], `#pragma enzyme checkpoint`) emit
// them: periodic 1, revolve 2, store_all 3, binomial 4. Both binomial
// schedules become Enzyme-MLIR's binomial one, storing every step is no
// checkpointing, and a budget of 0 or less asks for the default.

module {
  llvm.func local_unnamed_addr @body(!llvm.ptr)

  llvm.func local_unnamed_addr @periodic(%n: i32, %p: !llvm.ptr) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %s = llvm.mlir.constant(1 : i64) : i64
    %b = llvm.mlir.constant(6 : i64) : i64
    scf.for %i = %c0 to %n step %c1  : i32 {
      llvm.call @_Z26__enzyme_set_checkpointingmm(%s, %b) : (i64, i64) -> ()
      llvm.call @body(%p) : (!llvm.ptr) -> ()
    }
    llvm.return
  }

  llvm.func local_unnamed_addr @_Z26__enzyme_set_checkpointingmm(i64, i64)
}

// CHECK-LABEL: llvm.func local_unnamed_addr @periodic
// CHECK-NOT:     __enzyme_set_checkpointing
// CHECK:         scf.for
// CHECK:           llvm.call @body
// CHECK:         } {enzyme.checkpoint_period = 6 : i64, enzyme.enable_checkpointing = true}

// -----

module {
  llvm.func local_unnamed_addr @body(!llvm.ptr)

  llvm.func local_unnamed_addr @periodic_default(%n: i32, %p: !llvm.ptr) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %s = llvm.mlir.constant(1 : i64) : i64
    %b = llvm.mlir.constant(-1 : i64) : i64
    scf.for %i = %c0 to %n step %c1  : i32 {
      llvm.call @_Z26__enzyme_set_checkpointingmm(%s, %b) : (i64, i64) -> ()
      llvm.call @body(%p) : (!llvm.ptr) -> ()
    }
    llvm.return
  }

  llvm.func local_unnamed_addr @_Z26__enzyme_set_checkpointingmm(i64, i64)
}

// CHECK-LABEL: llvm.func local_unnamed_addr @periodic_default
// CHECK-NOT:     __enzyme_set_checkpointing
// CHECK:         scf.for
// CHECK:           llvm.call @body
// CHECK-NOT:     enzyme.checkpoint_period
// CHECK:         } {enzyme.enable_checkpointing = true}

// -----

module {
  llvm.func local_unnamed_addr @body(!llvm.ptr)

  llvm.func local_unnamed_addr @revolve(%n: i32, %p: !llvm.ptr) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %s = llvm.mlir.constant(2 : i64) : i64
    %b = llvm.mlir.constant(4 : i64) : i64
    scf.for %i = %c0 to %n step %c1  : i32 {
      llvm.call @_Z26__enzyme_set_checkpointingmm(%s, %b) : (i64, i64) -> ()
      llvm.call @body(%p) : (!llvm.ptr) -> ()
    }
    llvm.return
  }

  llvm.func local_unnamed_addr @_Z26__enzyme_set_checkpointingmm(i64, i64)
}

// CHECK-LABEL: llvm.func local_unnamed_addr @revolve
// CHECK-NOT:     __enzyme_set_checkpointing
// CHECK:         scf.for
// CHECK:           llvm.call @body
// CHECK:         } {enzyme.binomial_checkpointing, enzyme.checkpoint_period = 4 : i64, enzyme.enable_checkpointing = true}

// -----

module {
  llvm.func local_unnamed_addr @body(!llvm.ptr)

  llvm.func local_unnamed_addr @store_all(%n: i32, %p: !llvm.ptr) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %s = llvm.mlir.constant(3 : i64) : i64
    %b = llvm.mlir.constant(-1 : i64) : i64
    scf.for %i = %c0 to %n step %c1  : i32 {
      llvm.call @_Z26__enzyme_set_checkpointingmm(%s, %b) : (i64, i64) -> ()
      llvm.call @body(%p) : (!llvm.ptr) -> ()
    }
    llvm.return
  }

  llvm.func local_unnamed_addr @_Z26__enzyme_set_checkpointingmm(i64, i64)
}

// CHECK-LABEL: llvm.func local_unnamed_addr @store_all
// CHECK-NOT:     __enzyme_set_checkpointing
// CHECK:         scf.for
// CHECK:           llvm.call @body
// CHECK-NOT:     enzyme.binomial_checkpointing
// CHECK:         } {enzyme.enable_checkpointing = false}

// -----

module {
  llvm.func local_unnamed_addr @body(!llvm.ptr)

  llvm.func local_unnamed_addr @binomial(%n: i32, %p: !llvm.ptr) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %s = llvm.mlir.constant(4 : i64) : i64
    %b = llvm.mlir.constant(3 : i64) : i64
    scf.for %i = %c0 to %n step %c1  : i32 {
      llvm.call @_Z26__enzyme_set_checkpointingmm(%s, %b) : (i64, i64) -> ()
      llvm.call @body(%p) : (!llvm.ptr) -> ()
    }
    llvm.return
  }

  llvm.func local_unnamed_addr @_Z26__enzyme_set_checkpointingmm(i64, i64)
}

// CHECK-LABEL: llvm.func local_unnamed_addr @binomial
// CHECK-NOT:     __enzyme_set_checkpointing
// CHECK:         scf.for
// CHECK:           llvm.call @body
// CHECK:         } {enzyme.binomial_checkpointing, enzyme.checkpoint_period = 3 : i64, enzyme.enable_checkpointing = true}

// -----

module {
  llvm.func local_unnamed_addr @body(!llvm.ptr)

  llvm.func local_unnamed_addr @binomial_default(%n: i32, %p: !llvm.ptr) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %s = llvm.mlir.constant(4 : i64) : i64
    %b = llvm.mlir.constant(0 : i64) : i64
    scf.for %i = %c0 to %n step %c1  : i32 {
      llvm.call @_Z26__enzyme_set_checkpointingmm(%s, %b) : (i64, i64) -> ()
      llvm.call @body(%p) : (!llvm.ptr) -> ()
    }
    llvm.return
  }

  llvm.func local_unnamed_addr @_Z26__enzyme_set_checkpointingmm(i64, i64)
}

// CHECK-LABEL: llvm.func local_unnamed_addr @binomial_default
// CHECK-NOT:     __enzyme_set_checkpointing
// CHECK:         scf.for
// CHECK:           llvm.call @body
// CHECK-NOT:     enzyme.checkpoint_period
// CHECK:         } {enzyme.binomial_checkpointing, enzyme.enable_checkpointing = true}

// -----

// A schedule checkpoint_schedule.h does not name is left alone, call and all.

module {
  llvm.func local_unnamed_addr @body(!llvm.ptr)

  llvm.func local_unnamed_addr @unknown(%n: i32, %p: !llvm.ptr) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %s = llvm.mlir.constant(9 : i64) : i64
    %b = llvm.mlir.constant(2 : i64) : i64
    scf.for %i = %c0 to %n step %c1  : i32 {
      // expected-warning @+1 {{unknown checkpointing schedule 9}}
      llvm.call @_Z26__enzyme_set_checkpointingmm(%s, %b) : (i64, i64) -> ()
      llvm.call @body(%p) : (!llvm.ptr) -> ()
    }
    llvm.return
  }

  llvm.func local_unnamed_addr @_Z26__enzyme_set_checkpointingmm(i64, i64)
}

// CHECK-LABEL: llvm.func local_unnamed_addr @unknown
// CHECK:         scf.for
// CHECK:           llvm.call @_Z26__enzyme_set_checkpointingmm
// CHECK-NOT:     enzyme.enable_checkpointing
