// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(lower-kernel,canonicalize)" | FileCheck %s

module {
  llvm.func internal unnamed_addr fastcc @throw_boundserror_2676() attributes {dso_local, no_inline, sym_visibility = "private"} {
    llvm.unreachable
  }
  llvm.func internal ptx_kernelcc @kern(%arg0: !llvm.ptr<1>) {
    %0 = llvm.mlir.constant(63 : i32) : i32
    %1 = nvvm.read.ptx.sreg.tid.x : i32
    %2 = llvm.icmp "ugt" %1, %0 : i32
    llvm.cond_br %2, ^bb2, ^bb1
  ^bb1:  // pred: ^bb0
    %4 = llvm.zext %1 : i32 to i64
    %5 = llvm.getelementptr inbounds %arg0[%4] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i64
    %6 = llvm.load %5 {alignment = 1 : i64} : !llvm.ptr<1> -> i64
    %7 = llvm.mul %6, %6 : i64
    llvm.store %7, %5 {alignment = 1 : i64} : i64, !llvm.ptr<1>
    llvm.return
  ^bb2:  // pred: ^bb0
    llvm.call fastcc @throw_boundserror_2676() : () -> ()
    llvm.unreachable
  }
  func.func @main(%arg0: tensor<64xi64>) -> tensor<64xi64> {
    %c0 = stablehlo.constant dense<0> : tensor<i64>
    %c1 = stablehlo.constant dense<1> : tensor<i64>
    %c40 = stablehlo.constant dense<40> : tensor<i64>
    %0 = enzymexla.kernel_call @kern blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c40) shmem=%c0 (%arg0) {output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>]} : (tensor<64xi64>) -> tensor<64xi64>
    return %0 : tensor<64xi64>
  }
  func.func @main2(%arg0: tensor<64xi64>) -> tensor<64xi64> {
    %c0 = stablehlo.constant dense<0> : tensor<i64>
    %c1 = stablehlo.constant dense<1> : tensor<i64>
    %c42 = stablehlo.constant dense<42> : tensor<i64>
    %0 = enzymexla.kernel_call @kern blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c42) shmem=%c0 (%arg0) {output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>]} : (tensor<64xi64>) -> tensor<64xi64>
    return %0 : tensor<64xi64>
  }
}

// CHECK:  gpu.module @gpumod_kern {
// CHECK:    llvm.func internal unnamed_addr fastcc @throw_boundserror_2676() attributes {dso_local, no_inline, sym_visibility = "private"} {
// CHECK-NEXT:      llvm.unreachable
// CHECK-NEXT:    }
// CHECK:    gpu.func @kern(%arg0: !llvm.ptr<1>) kernel {
// CHECK-NEXT:      %0 = llvm.mlir.constant(63 : i32) : i32
// CHECK-NEXT:      %1 = nvvm.read.ptx.sreg.tid.x : i32
// CHECK-NEXT:      %2 = llvm.icmp "ugt" %1, %0 : i32
// CHECK-NEXT:      llvm.cond_br %2, ^bb2, ^bb1
// CHECK-NEXT:    ^bb1:  // pred: ^bb0
// CHECK-NEXT:      %3 = llvm.zext %1 : i32 to i64
// CHECK-NEXT:      %4 = llvm.getelementptr inbounds %arg0[%3] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i64
// CHECK-NEXT:      %5 = llvm.load %4 {alignment = 1 : i64} : !llvm.ptr<1> -> i64
// CHECK-NEXT:      %6 = llvm.mul %5, %5 : i64
// CHECK-NEXT:      llvm.store %6, %4 {alignment = 1 : i64} : i64, !llvm.ptr<1>
// CHECK-NEXT:      gpu.return
// CHECK-NEXT:    ^bb2:  // pred: ^bb0
// CHECK-NEXT:      llvm.call fastcc @throw_boundserror_2676() : () -> ()
// CHECK-NEXT:      gpu.return
// CHECK-NEXT:    }
// CHECK-NEXT:  }
// CHECK:  func.func private @kern$call$1(%arg0: !llvm.ptr<1>) {
// CHECK-NEXT:    %c1_i64 = arith.constant 1 : i64
// CHECK-NEXT:    %c40_i64 = arith.constant 40 : i64
// CHECK-NEXT:    %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:    %0 = "enzymexla.get_stream"() : () -> !gpu.async.token
// CHECK-NEXT:    %1 = gpu.launch_func async [%0] @gpumod_kern::@kern blocks in (%c1_i64, %c1_i64, %c1_i64) threads in (%c1_i64, %c1_i64, %c40_i64) : i64 dynamic_shared_memory_size %c0_i32 args(%arg0 : !llvm.ptr<1>)
// CHECK-NEXT:    return
// CHECK-NEXT:  }
// CHECK:  func.func private @kern$call$2(%arg0: !llvm.ptr<1>) {
// CHECK-NEXT:    %c1_i64 = arith.constant 1 : i64
// CHECK-NEXT:    %c42_i64 = arith.constant 42 : i64
// CHECK-NEXT:    %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:    %0 = "enzymexla.get_stream"() : () -> !gpu.async.token
// CHECK-NEXT:    %1 = gpu.launch_func async [%0] @gpumod_kern::@kern blocks in (%c1_i64, %c1_i64, %c1_i64) threads in (%c1_i64, %c1_i64, %c42_i64) : i64 dynamic_shared_memory_size %c0_i32 args(%arg0 : !llvm.ptr<1>)
// CHECK-NEXT:    return
// CHECK-NEXT:  }
// CHECK:  func.func @main(%arg0: tensor<64xi64>) -> tensor<64xi64> {
// CHECK-NEXT:    %0 = enzymexla.jit_call @kern$call$1 (%arg0) {output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>]} : (tensor<64xi64>) -> tensor<64xi64>
// CHECK-NEXT:    return %0 : tensor<64xi64>
// CHECK-NEXT:  }
// CHECK:  func.func @main2(%arg0: tensor<64xi64>) -> tensor<64xi64> {
// CHECK-NEXT:    %0 = enzymexla.jit_call @kern$call$2 (%arg0) {output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>]} : (tensor<64xi64>) -> tensor<64xi64>
// CHECK-NEXT:    return %0 : tensor<64xi64>
// CHECK-NEXT:  }
