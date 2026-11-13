// RUN: enzymexlamlir-opt --mark-func-memory-effects %s | FileCheck %s

module {
  llvm.func @ext_callee(!llvm.ptr)

  // Reading/writing through GEPs and returning a loaded value does not capture.
  // CHECK-LABEL: llvm.func @nocapture_gep_load_store(
  // CHECK-SAME: %arg0: !llvm.ptr {enzymexla.memory_effects = ["read"], llvm.nocapture, llvm.nofree, llvm.readonly},
  // CHECK-SAME: %arg1: !llvm.ptr {enzymexla.memory_effects = ["write"], llvm.nocapture, llvm.nofree, llvm.writeonly})
  llvm.func @nocapture_gep_load_store(%in: !llvm.ptr, %out: !llvm.ptr) -> f64 {
    %c2 = llvm.mlir.constant(2 : i64) : i64
    %gep_in = llvm.getelementptr inbounds %in[%c2] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %v = llvm.load %gep_in : !llvm.ptr -> f64
    %gep_out = llvm.getelementptr inbounds %out[%c2] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %v, %gep_out : f64, !llvm.ptr
    llvm.return %v : f64
  }

  // Storing a pointer into memory captures the stored pointer (%p), but not the destination (%slot).
  // CHECK-LABEL: llvm.func @capture_stored_ptr(
  // CHECK-SAME: %arg0: !llvm.ptr {enzymexla.memory_effects = [], llvm.nofree},
  // CHECK-SAME: %arg1: !llvm.ptr {enzymexla.memory_effects = ["write"], llvm.nocapture, llvm.nofree, llvm.writeonly})
  llvm.func @capture_stored_ptr(%p: !llvm.ptr, %slot: !llvm.ptr) {
    llvm.store %p, %slot : !llvm.ptr, !llvm.ptr
    llvm.return
  }

  // Self-store `store %p, %p` captures %p because it is the stored value.
  // CHECK-LABEL: llvm.func @capture_self_store(
  // CHECK-SAME: %arg0: !llvm.ptr {enzymexla.memory_effects = ["write"], llvm.nofree, llvm.writeonly})
  llvm.func @capture_self_store(%p: !llvm.ptr) {
    llvm.store %p, %p : !llvm.ptr, !llvm.ptr
    llvm.return
  }

  // Returning a pointer captures it.
  // CHECK-LABEL: llvm.func @capture_returned_ptr(
  // CHECK-SAME: %arg0: !llvm.ptr {enzymexla.memory_effects = [], llvm.nofree})
  llvm.func @capture_returned_ptr(%p: !llvm.ptr) -> !llvm.ptr {
    %c1 = llvm.mlir.constant(1 : i64) : i64
    %g = llvm.getelementptr %p[%c1] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.return %g : !llvm.ptr
  }

  // ptrtoint captures the address bits.
  // CHECK-LABEL: llvm.func @capture_ptrtoint(
  // CHECK-SAME: %arg0: !llvm.ptr {enzymexla.memory_effects = [], llvm.nofree})
  llvm.func @capture_ptrtoint(%p: !llvm.ptr) -> i64 {
    %i = llvm.ptrtoint %p : !llvm.ptr to i64
    llvm.return %i : i64
  }

  // Calling an external function without known nocapture captures; calling a proven-nocapture callee does not.
  // CHECK-LABEL: llvm.func @caller_propagation(
  // CHECK-SAME: %arg0: !llvm.ptr {enzymexla.memory_effects = ["read"], llvm.nocapture, llvm.nofree, llvm.readonly},
  // CHECK-SAME: %arg1: !llvm.ptr {enzymexla.memory_effects = ["write"], llvm.nocapture, llvm.nofree, llvm.writeonly},
  // CHECK-SAME: %arg2: !llvm.ptr {enzymexla.memory_effects = ["read", "write", "allocate", "free"]})
  llvm.func @caller_propagation(%a: !llvm.ptr, %b: !llvm.ptr, %c: !llvm.ptr) -> f64 {
    %r = llvm.call @nocapture_gep_load_store(%a, %b) : (!llvm.ptr, !llvm.ptr) -> f64
    llvm.call @ext_callee(%c) : (!llvm.ptr) -> ()
    llvm.return %r : f64
  }
}
