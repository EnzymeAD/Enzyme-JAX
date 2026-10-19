// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(convert-parallel-to-gpu1)" | FileCheck %s

// The stream an async.execute waits on is given to the launch it contains.
// gpu.launch takes it the same way gpu.launch_func does, on its asyncObject
// operand: a dependency operand with no result token does not verify, and the
// launch built here returns no token.

module {
  func.func @async_launch(%stream: !llvm.ptr, %n: index, %out: memref<?xf64>, %v: f64) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %tok = "enzymexla.stream2token"(%stream) : (!llvm.ptr) -> !async.token
    %done = async.execute [%tok] {
      "enzymexla.gpu_wrapper"(%n, %c1, %c1, %c8, %c1, %c1) ({
        scf.parallel (%i) = (%c0) to (%n) step (%c1) {
          scf.parallel (%j) = (%c0) to (%c8) step (%c1) {
            memref.store %v, %out[%j] : memref<?xf64>
            scf.reduce
          }
          scf.reduce
        }
        "enzymexla.polygeist_yield"() : () -> ()
      }) : (index, index, index, index, index, index) -> index
      async.yield
    }
    return
  }
}

// CHECK-LABEL: func.func @async_launch(
// CHECK-SAME: %[[stream:[^ :]+]]: !llvm.ptr
// The stream is the launch's asyncObject, and no async token is asked for.
// CHECK: gpu.launch <%[[stream]] : !llvm.ptr> blocks
// CHECK-NOT: gpu.launch async
