// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(lower-kernel,canonicalize)" | FileCheck %s --check-prefix=GPU
// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(lower-kernel{backend=cpu},canonicalize)" | FileCheck %s --check-prefix=CPU

module {
  llvm.func internal ptx_kernelcc @kern(
      %in: !llvm.ptr<1> {llvm.align = 64 : i64, llvm.noalias, llvm.nocapture, llvm.nofree, llvm.readonly},
      %out: !llvm.ptr<1> {llvm.align = 64 : i64, llvm.noalias, llvm.nocapture, llvm.nofree, llvm.writeonly},
      %scale: f32) {
    %tid = nvvm.read.ptx.sreg.tid.x : i32
    %idx = llvm.zext %tid : i32 to i64
    %gep_in = llvm.getelementptr inbounds %in[%idx] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
    %val = llvm.load %gep_in <alignment = 4> : !llvm.ptr<1> -> f32
    %res = llvm.fmul %val, %scale : f32
    %gep_out = llvm.getelementptr inbounds %out[%idx] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
    llvm.store %res, %gep_out <alignment = 4> : f32, !llvm.ptr<1>
    llvm.return
  }

  func.func @main(%a: tensor<32xf32>, %b: tensor<32xf32>, %s: tensor<f32>) -> tensor<32xf32> {
    %c0 = stablehlo.constant dense<0> : tensor<i64>
    %c1 = stablehlo.constant dense<1> : tensor<i64>
    %c32 = stablehlo.constant dense<32> : tensor<i64>
    %0 = enzymexla.kernel_call @kern blocks in (%c1, %c1, %c1) threads in (%c32, %c1, %c1) shmem=%c0 (%a, %b, %s) <{output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 1, operand_tuple_indices = []>]}> : (tensor<32xf32>, tensor<32xf32>, tensor<f32>) -> tensor<32xf32>
    return %0 : tensor<32xf32>
  }
}

// GPU-LABEL: gpu.module @gpumod_kern {
// GPU: gpu.func @kern(
// GPU-SAME: %arg0: !llvm.ptr<1> {llvm.align = 64 : i64, llvm.noalias, llvm.nocapture, llvm.nofree, llvm.readonly},
// GPU-SAME: %arg1: !llvm.ptr<1> {llvm.align = 64 : i64, llvm.noalias, llvm.nocapture, llvm.nofree, llvm.writeonly},
// GPU-SAME: %arg2: f32) kernel
// GPU-LABEL: func.func private @kern$call$1(
// GPU-SAME: %arg0: !llvm.ptr<1> {llvm.align = 64 : i64, llvm.noalias, llvm.nocapture, llvm.nofree, llvm.readonly},
// GPU-SAME: %arg1: !llvm.ptr<1> {llvm.align = 64 : i64, llvm.noalias, llvm.nocapture, llvm.nofree, llvm.writeonly},
// GPU-SAME: %arg2: f32)

// CPU-LABEL: func.func private @kern$par0(
// CPU-SAME: %arg0: !llvm.ptr<1> {llvm.align = 64 : i64, llvm.noalias, llvm.nocapture, llvm.nofree, llvm.readonly},
// CPU-SAME: %arg1: !llvm.ptr<1> {llvm.align = 64 : i64, llvm.noalias, llvm.nocapture, llvm.nofree, llvm.writeonly},
// CPU-SAME: %arg2: f32)
