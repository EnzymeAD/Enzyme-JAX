// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(lower-kernel{backend=cuda},parallel-lower{wrapParallelOps=1})" | FileCheck %s

// Verify parallel-lower handles gpu.launch async dependencies produced by
// enzymexla.get_stream (from lower-kernel{backend=cuda}) and rocdl barriers.
module {
  llvm.func ptx_kernelcc @stream_kern(%ptr: !llvm.ptr<1>) {
    %c = llvm.mlir.constant(1.0 : f32) : f32
    rocdl.s.barrier
    llvm.store %c, %ptr : f32, !llvm.ptr<1>
    llvm.return
  }
  func.func @main(%x: tensor<4xf32>) -> tensor<4xf32> {
    %c0 = stablehlo.constant dense<0> : tensor<i64>
    %c1 = stablehlo.constant dense<1> : tensor<i64>
    %c4 = stablehlo.constant dense<4> : tensor<i64>
    %res = enzymexla.kernel_call @stream_kern
      blocks in(%c1, %c1, %c1) threads in(%c4, %c1, %c1)
      shmem = %c0 (%x)
      <{output_operand_aliases = [#stablehlo.output_operand_alias<
        output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>]}>
      : (tensor<4xf32>) -> tensor<4xf32>
    return %res : tensor<4xf32>
  }
}

// CHECK-LABEL: func.func private @stream_kern$call$
// CHECK: %[[S:.*]] = "enzymexla.get_stream"() : () -> !llvm.ptr
// CHECK: %[[TOK:.*]] = "enzymexla.stream2token"(%[[S]]) : (!llvm.ptr) -> !async.token
// CHECK: async.execute [%[[TOK]]]
// CHECK: "enzymexla.barrier"
