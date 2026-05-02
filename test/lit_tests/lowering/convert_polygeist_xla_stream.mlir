// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(convert-polygeist-to-llvm{backend=xla-gpu})" | FileCheck %s

// CHECK-LABEL: llvm.func @rxla$raised_0
// CHECK-SAME: (!llvm.ptr)

// CHECK-LABEL: llvm.func @test_stream_async
// CHECK-SAME: (%arg0: !llvm.ptr, %arg1: !llvm.ptr)
// CHECK-NOT: enzymexla.stream2token
// CHECK-NOT: async.execute
// CHECK: llvm.call @rxla$raised_0(%arg1) : (!llvm.ptr) -> ()
// CHECK: llvm.return
module {
  func.func private @rxla$raised_0(memref<?xf64>) -> ()
  
  func.func @test_stream_async(%arg0: !llvm.ptr, %arg1: memref<?xf64>) {
    %0 = "enzymexla.stream2token"(%arg0) : (!llvm.ptr) -> !async.token
    %token = async.execute [%0] {
      func.call @rxla$raised_0(%arg1) : (memref<?xf64>) -> ()
      async.yield
    }
    return
  }
}
