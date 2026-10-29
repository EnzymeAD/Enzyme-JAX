// RUN: enzymexlamlir-opt %s --nvvm-attach-target="chip=sm_80" | FileCheck %s --check-prefix=NVVM
// RUN: enzymexlamlir-opt %s --rocdl-attach-target="chip=gfx942" | FileCheck %s --check-prefix=ROCDL

// The new LLVM registers these two passes separately from the GPU passes
// (mlir/Dialect/GPU/Transforms/{NVVM,ROCDL}); the Julia pipelines use both.

// NVVM: gpu.module @kernels [#nvvm.target<chip = "sm_80">]
// ROCDL: gpu.module @kernels [#rocdl.target<chip = "gfx942">]
gpu.module @kernels {
  gpu.func @kernel() kernel {
    gpu.return
  }
}
