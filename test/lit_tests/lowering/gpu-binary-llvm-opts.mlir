// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(enzymexla-gpu-module-to-binary{format=isa})" | FileCheck %s --check-prefix=DEFAULT
// RUN: enzymexlamlir-opt %s --pass-pipeline='builtin.module(enzymexla-gpu-module-to-binary{format=isa llvm-opts="-nvptx-fma-level=0"})' | FileCheck %s --check-prefix=NOFMA
// RUN: enzymexlamlir-opt %s --nvptx-fma-level=0 --pass-pipeline="builtin.module(enzymexla-gpu-module-to-binary{format=isa})" | FileCheck %s --check-prefix=NOFMA
// RUN: enzymexlamlir-opt %s --nvptx-fma-level=0 --pass-pipeline='builtin.module(enzymexla-gpu-module-to-binary{format=isa llvm-opts="-nvptx-fma-level=1"})' | FileCheck %s --check-prefix=DEFAULT
// RUN: enzymexlamlir-opt %s --pass-pipeline='builtin.module(enzymexla-gpu-module-to-binary{format=isa llvm-opts="-nvptx-prec-divf32=0"})' | FileCheck %s --check-prefix=APPROXDIV
// RUN: enzymexlamlir-opt %s --pass-pipeline='builtin.module(enzymexla-gpu-module-to-binary{format=isa llvm-opts="-nvptx-fma-level=0 -nvptx-prec-divf32=0"})' | FileCheck %s --check-prefix=BOTH
// RUN: not enzymexlamlir-opt %s --pass-pipeline='builtin.module(enzymexla-gpu-module-to-binary{format=isa llvm-opts="-unknown-llvm-flag-xyz"})' 2>&1 | FileCheck %s --check-prefix=ERR

module attributes {gpu.container_module} {
  gpu.module @kmod [#nvvm.target<O = 3, chip = "sm_80", features = "+ptx80,+sm_80", flags = {}>] {
    llvm.func @fma_and_div(%out64: !llvm.ptr, %a: f64, %b: f64, %c: f64, %out32: !llvm.ptr, %x: f32, %y: f32) attributes {gpu.kernel, nvvm.kernel} {
      %mul = llvm.fmul %a, %b : f64
      %add = llvm.fadd %mul, %c : f64
      llvm.store %add, %out64 : f64, !llvm.ptr
      %div = llvm.fdiv %x, %y : f32
      llvm.store %div, %out32 : f32, !llvm.ptr
      llvm.return
    }
  }
}

// DEFAULT: gpu.binary @kmod
// DEFAULT-NOT: mul.rn.f64
// DEFAULT: fma.rn.f64
// DEFAULT: div.rn.f32
// DEFAULT-NOT: mul.rn.f64

// NOFMA: gpu.binary @kmod
// NOFMA-NOT: fma.rn.f64
// NOFMA: mul.rn.f64
// NOFMA: add.rn.f64
// NOFMA: div.rn.f32
// NOFMA-NOT: fma.rn.f64

// APPROXDIV: gpu.binary @kmod
// APPROXDIV: fma.rn.f64
// APPROXDIV: div.approx.f32

// BOTH: gpu.binary @kmod
// BOTH-NOT: fma.rn.f64
// BOTH: mul.rn.f64
// BOTH: add.rn.f64
// BOTH: div.approx.f32
// BOTH-NOT: fma.rn.f64

// ERR: Failed to parse LLVM options:
