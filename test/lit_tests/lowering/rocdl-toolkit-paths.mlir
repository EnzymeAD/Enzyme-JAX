// UNSUPPORTED: system-windows
// RUN: rm -rf %t && mkdir -p %t/std/llvm/bin %t/therock/lib/llvm/bin %t/therock/lib/llvm/amdgcn/bitcode %t/uenv/bin %t/uenv/lib/amdgcn/bitcode %t/pathdir %t/emptybin
// RUN: printf '#!/bin/sh\nwhile [ $# -gt 0 ]; do if [ "$1" = "-o" ]; then printf MOCK_STD > "$2"; exit 0; fi; shift; done\nexit 1\n' > %t/std/llvm/bin/ld.lld && chmod +x %t/std/llvm/bin/ld.lld
// RUN: printf '#!/bin/sh\nwhile [ $# -gt 0 ]; do if [ "$1" = "-o" ]; then printf MOCK_THEROCK > "$2"; exit 0; fi; shift; done\nexit 1\n' > %t/therock/lib/llvm/bin/ld.lld && chmod +x %t/therock/lib/llvm/bin/ld.lld
// RUN: printf '#!/bin/sh\nwhile [ $# -gt 0 ]; do if [ "$1" = "-o" ]; then printf MOCK_UENV > "$2"; exit 0; fi; shift; done\nexit 1\n' > %t/uenv/bin/ld.lld && chmod +x %t/uenv/bin/ld.lld
// RUN: printf '#!/bin/sh\nwhile [ $# -gt 0 ]; do if [ "$1" = "-o" ]; then printf MOCK_PATH > "$2"; exit 0; fi; shift; done\nexit 1\n' > %t/pathdir/ld.lld && chmod +x %t/pathdir/ld.lld
// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(enzymexla-gpu-module-to-binary{format=bin toolkit=%t/std})" | FileCheck %s --check-prefix=STD
// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(enzymexla-gpu-module-to-binary{format=bin toolkit=%t/therock})" | FileCheck %s --check-prefix=THEROCK
// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(enzymexla-gpu-module-to-binary{format=bin toolkit=%t/uenv})" | FileCheck %s --check-prefix=UENV
// RUN: env -u ROCM_PATH -u ROCM_ROOT -u ROCM_HOME PATH="%t/pathdir:$PATH" enzymexlamlir-opt %s --pass-pipeline="builtin.module(enzymexla-gpu-module-to-binary{format=bin})" | FileCheck %s --check-prefix=PATHFALLBACK
// RUN: OPT=$(command -v enzymexlamlir-opt) && not env -u ROCM_PATH -u ROCM_ROOT -u ROCM_HOME PATH="%t/emptybin" "$OPT" %s --pass-pipeline="builtin.module(enzymexla-gpu-module-to-binary{format=bin})" 2>&1 | FileCheck %s --check-prefix=ERR
// RUN: echo 'module attributes {gpu.container_module} { gpu.module @ocml [#rocdl.target<chip = "gfx942">] { llvm.func @__ocml_sin_f32(%arg0: f32) -> f32 { llvm.return %arg0 : f32 } } }' | enzymexlamlir-opt --pass-pipeline="builtin.module(enzymexla-gpu-module-to-binary{format=offloading})" > %t/ocml.mlir
// RUN: python -c 'import pathlib, re, sys; raw = re.sub(r"\\([0-9A-Fa-f]{2}|.)", lambda g: chr(int(g.group(1), 16)) if len(g.group(1)) == 2 else chr({"n": 10, "t": 9, "\\": 92, "\"": 34}[g.group(1)]), re.search(r"offload = \"((?:[^\"\\]|\\.)*)\"", pathlib.Path(sys.argv[1]).read_text(), re.S).group(1)).encode("latin1"); [pathlib.Path(q).write_bytes(raw) for q in sys.argv[2:]]' %t/ocml.mlir %t/therock/lib/llvm/amdgcn/bitcode/ocml.bc %t/uenv/lib/amdgcn/bitcode/ocml.bc
// RUN: echo 'module attributes {gpu.container_module} { gpu.module @kmod_bc [#rocdl.target<chip = "gfx942">] { llvm.func @__ocml_sin_f32(f32) -> f32 llvm.func @kern(%a0: !llvm.ptr, %a1: f32) attributes {gpu.kernel, rocdl.kernel} { %r0 = llvm.call @__ocml_sin_f32(%a1) : (f32) -> f32 llvm.store %r0, %a0 : f32, !llvm.ptr llvm.return } } }' > %t/use_ocml.mlir
// RUN: enzymexlamlir-opt %t/use_ocml.mlir --pass-pipeline="builtin.module(enzymexla-gpu-module-to-binary{format=bin toolkit=%t/therock})" | FileCheck %s --check-prefix=THEROCKBC
// RUN: enzymexlamlir-opt %t/use_ocml.mlir --pass-pipeline="builtin.module(enzymexla-gpu-module-to-binary{format=bin toolkit=%t/uenv})" | FileCheck %s --check-prefix=UENVBC

module attributes {gpu.container_module} {
  gpu.module @kmod [#rocdl.target<chip = "gfx942">] {
    llvm.func @noop(%out: !llvm.ptr, %x: f32) attributes {gpu.kernel, rocdl.kernel} {
      llvm.store %x, %out : f32, !llvm.ptr
      llvm.return
    }
  }
}

// STD: gpu.binary @kmod [#gpu.object<#rocdl.target<chip = "gfx942">, bin = "MOCK_STD">]
// THEROCK: gpu.binary @kmod [#gpu.object<#rocdl.target<chip = "gfx942">, bin = "MOCK_THEROCK">]
// UENV: gpu.binary @kmod [#gpu.object<#rocdl.target<chip = "gfx942">, bin = "MOCK_UENV">]
// PATHFALLBACK: gpu.binary @kmod [#gpu.object<#rocdl.target<chip = "gfx942">, bin = "MOCK_PATH">]
// ERR: invalid ROCm path, please set a valid path
// THEROCKBC: gpu.binary @kmod_bc [#gpu.object<#rocdl.target<chip = "gfx942">, bin = "MOCK_THEROCK">]
// UENVBC: gpu.binary @kmod_bc [#gpu.object<#rocdl.target<chip = "gfx942">, bin = "MOCK_UENV">]
