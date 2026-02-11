def if_llvm_amdgpu_available(if_true, if_false = []):
    """Returns if_true if AMDGPU is enabled through --define=with_amdgpu=true."""
    return select({
        "@//:with_amdgpu": if_true,
        "//conditions:default": if_false,
    })

AMDGPU_DEPS = [
    "@llvm-project//llvm:AMDGPUAsmParser",
    "@llvm-project//llvm:AMDGPUCodeGen",
]