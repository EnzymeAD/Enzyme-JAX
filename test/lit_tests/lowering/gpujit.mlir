// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(lower-jit{jit=false})" | FileCheck %s

module attributes {gpu.container_module} {
  gpu.module @gpumod_kern {
    llvm.func internal unnamed_addr fastcc @throw_boundserror_2676() attributes {dso_local, no_inline, sym_visibility = "private"} {
      llvm.unreachable
    }
    gpu.func @kern(%arg0: !llvm.ptr<1>) kernel {
      %0 = llvm.mlir.constant(63 : i32) : i32
      %1 = nvvm.read.ptx.sreg.tid.x : i32
      %2 = llvm.icmp "ugt" %1, %0 : i32
      llvm.cond_br %2, ^bb2, ^bb1
    ^bb1:  // pred: ^bb0
      %3 = llvm.zext %1 : i32 to i64
      %4 = llvm.getelementptr inbounds %arg0[%3] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i64
      %5 = llvm.load %4 {alignment = 1 : i64} : !llvm.ptr<1> -> i64
      %6 = llvm.mul %5, %5 : i64
      llvm.store %6, %4 {alignment = 1 : i64} : i64, !llvm.ptr<1>
      gpu.return
    ^bb2:  // pred: ^bb0
      llvm.call fastcc @throw_boundserror_2676() : () -> ()
      gpu.return
    }
  }
  func.func private @kern$call$1(%arg0: !llvm.ptr<1>) {
    %c1_i64 = arith.constant 1 : i64
    %c40_i64 = arith.constant 40 : i64
    %c0_i32 = arith.constant 0 : i32
    %0 = "enzymexla.get_stream"() : () -> !gpu.async.token
    %1 = gpu.launch_func async [%0] @gpumod_kern::@kern blocks in (%c1_i64, %c1_i64, %c1_i64) threads in (%c1_i64, %c1_i64, %c40_i64) : i64 dynamic_shared_memory_size %c0_i32 args(%arg0 : !llvm.ptr<1>)
    return
  }
  func.func @main(%arg0: tensor<64xi64>) -> tensor<64xi64> {
    %0 = enzymexla.jit_call @kern$call$1 (%arg0) {output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>]} : (tensor<64xi64>) -> tensor<64xi64>
    return %0 : tensor<64xi64>
  }
}

// CHECK: func.func @main(%arg0: tensor<64xi64>) -> tensor<64xi64> {
// CHECK-NEXT:    %0 = stablehlo.custom_call @enzymexla_compile_gpu(%arg0) {api_version = 4 : i32, backend_config = {attr = "\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00"}, output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>]} : (tensor<64xi64>) -> tensor<64xi64>
// CHECK-NEXT:    return %0 : tensor<64xi64>
// CHECK-NEXT:  }
// CHECK-NEXT:}
