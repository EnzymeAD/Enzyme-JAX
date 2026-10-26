// RUN: enzymexlamlir-opt %s --split-input-file --pass-pipeline='builtin.module(canonicalize-scf-for)' | FileCheck %s

// Reading the attribute off the call is what the call was there for, so it
// goes with what it said -- and so does the declaration it named, but only
// once nothing calls it any more.

module {
  llvm.func local_unnamed_addr @body(!llvm.ptr)

  llvm.func local_unnamed_addr @mincut_off(%n: i32, %p: !llvm.ptr) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %m = llvm.mlir.constant(0 : i64) : i64
    scf.for %i = %c0 to %n step %c1  : i32 {
      llvm.call @_Z26__enzyme_set_mincutm(%m) : (i64 {llvm.noundef}) -> ()
      llvm.call @body(%p) : (!llvm.ptr) -> ()
    }
    llvm.return
  }

  llvm.func local_unnamed_addr @_Z26__enzyme_set_mincutm(i64 {llvm.noundef})
}

// CHECK-LABEL: llvm.func local_unnamed_addr @mincut_off
// CHECK-NOT:     __enzyme_set_mincut
// CHECK:         scf.for
// CHECK:           llvm.call @body
// CHECK:         } {enzyme.disable_mincut}
// CHECK-NOT:   llvm.func local_unnamed_addr @_Z26__enzyme_set_mincutm

// -----

// Left on, there is no attribute to add, and the call still goes.

module {
  llvm.func local_unnamed_addr @body(!llvm.ptr)

  llvm.func local_unnamed_addr @mincut_on(%n: i32, %p: !llvm.ptr) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %m = llvm.mlir.constant(1 : i64) : i64
    scf.for %i = %c0 to %n step %c1  : i32 {
      llvm.call @_Z26__enzyme_set_mincutm(%m) : (i64 {llvm.noundef}) -> ()
      llvm.call @body(%p) : (!llvm.ptr) -> ()
    }
    llvm.return
  }

  llvm.func local_unnamed_addr @_Z26__enzyme_set_mincutm(i64 {llvm.noundef})
}

// CHECK-LABEL: llvm.func local_unnamed_addr @mincut_on
// CHECK-NOT:     __enzyme_set_mincut
// CHECK:         scf.for
// CHECK-NOT:     enzyme.disable_mincut

// -----

// A call whose setting cannot be read stays, so the declaration it names stays
// with it rather than leaving it pointing at nothing.

module {
  llvm.func local_unnamed_addr @body(!llvm.ptr)

  llvm.func local_unnamed_addr @mincut_dynamic(%n: i32, %p: !llvm.ptr, %d: i64) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    scf.for %i = %c0 to %n step %c1  : i32 {
      llvm.call @_Z26__enzyme_set_mincutm(%d) : (i64 {llvm.noundef}) -> ()
      llvm.call @body(%p) : (!llvm.ptr) -> ()
    }
    llvm.return
  }

  llvm.func local_unnamed_addr @_Z26__enzyme_set_mincutm(i64 {llvm.noundef})
}

// CHECK-LABEL: llvm.func local_unnamed_addr @mincut_dynamic
// CHECK:         llvm.call @_Z26__enzyme_set_mincutm
// CHECK:       llvm.func local_unnamed_addr @_Z26__enzyme_set_mincutm
