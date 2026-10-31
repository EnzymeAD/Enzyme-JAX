#include <cstddef>

#include "src/enzyme_ad/jax/Utils.h"

#if defined(ENZYMEJAX_ROCM)
// Only the C entry points: the header's C++ templates (hipMalloc<T> and
// friends) would make the addresses below ambiguous.
#define __HIP_DISABLE_CPP_FUNCTIONS__
#include "rocm/include/hip/hip_runtime_api.h"

// Registration entry points of the HIP runtime, which clang's HIP host stubs
// call without any public declaration (hipamd/src/hip_platform.cpp).
extern "C" {
void **__hipRegisterFatBinary(const void *data);
void __hipUnregisterFatBinary(void **modules);
void __hipRegisterFunction(void **modules, const void *hostFunction,
                           char *deviceFunction, const char *deviceName,
                           unsigned int threadLimit, uint3 *tid, uint3 *bid,
                           dim3 *blockDim, dim3 *gridDim, int *wSize);
void __hipRegisterVar(void **modules, void *var, char *hostVar, char *deviceVar,
                      int ext, size_t size, int constant, int global);
}

namespace enzymexla {
namespace ffi_internal {

// Kernels jitted by lower-jit from convert-polygeist-to-llvm{backend=rocm}
// call into the HIP runtime. Hand their addresses to the JIT directly, as for
// the CUDA runtime: jitted kernels then share the libamdhip64 XLA uses, and
// the references keep these entry points linked into a library that would
// otherwise not use them.
void registerEnzymeJaXRocmRuntimeSymbols() {
#define MAP_HIP(name)                                                          \
  mlir::enzyme::EnzymeJaXMapSymbol(#name, reinterpret_cast<void *>(&name))
  MAP_HIP(__hipRegisterFatBinary);
  MAP_HIP(__hipUnregisterFatBinary);
  MAP_HIP(__hipRegisterFunction);
  MAP_HIP(__hipRegisterVar);
  MAP_HIP(hipLaunchKernel);
  MAP_HIP(hipMalloc);
  MAP_HIP(hipFree);
  MAP_HIP(hipMemcpy);
  MAP_HIP(hipMemcpy2D);
  MAP_HIP(hipMemset);
  MAP_HIP(hipDeviceSynchronize);
  MAP_HIP(hipGetLastError);
#undef MAP_HIP
  // The header declares a template overload of this one outside the
  // __HIP_DISABLE_CPP_FUNCTIONS__ guard, so pick the C entry point explicitly.
  mlir::enzyme::EnzymeJaXMapSymbol(
      "hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags",
      reinterpret_cast<void *>(
          static_cast<hipError_t (*)(int *, const void *, int, size_t,
                                     unsigned int)>(
              &hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags)));
}

} // namespace ffi_internal
} // namespace enzymexla

#else

namespace enzymexla {
namespace ffi_internal {

void registerEnzymeJaXRocmRuntimeSymbols() {}

} // namespace ffi_internal
} // namespace enzymexla

#endif
