#include <cstddef>

#include "src/enzyme_ad/jax/Utils.h"

#if defined(ENZYMEJAX_CUDA)
#include "third_party/gpus/cuda/include/cuda_runtime_api.h"

// Registration entry points of the CUDA runtime, normally declared by the
// nvcc-generated host stubs (crt/host_runtime.h).
extern "C" {
void **__cudaRegisterFatBinary(void *fatCubin);
void __cudaRegisterFatBinaryEnd(void **fatCubinHandle);
void __cudaUnregisterFatBinary(void **fatCubinHandle);
void __cudaRegisterFunction(void **fatCubinHandle, const char *hostFun,
                            char *deviceFun, const char *deviceName,
                            int thread_limit, uint3 *tid, uint3 *bid,
                            dim3 *bDim, dim3 *gDim, int *wSize);
void __cudaRegisterVar(void **fatCubinHandle, char *hostVar,
                       char *deviceAddress, const char *deviceName, int ext,
                       size_t size, int constant, int global);
}

namespace enzymexla {
namespace ffi_internal {

// Kernels jitted by lower-jit call into the CUDA runtime. The static libcudart
// linked in here hides its symbols, so hand their addresses to the JIT
// directly; jitted kernels then share the runtime instance XLA uses.
void registerEnzymeJaXCudaRuntimeSymbols() {
#define MAP_CUDART(name)                                                       \
  mlir::enzyme::EnzymeJaXMapSymbol(#name, reinterpret_cast<void *>(&name))
  MAP_CUDART(__cudaRegisterFatBinary);
  MAP_CUDART(__cudaRegisterFatBinaryEnd);
  MAP_CUDART(__cudaUnregisterFatBinary);
  MAP_CUDART(__cudaRegisterFunction);
  MAP_CUDART(__cudaRegisterVar);
  MAP_CUDART(cudaLaunchKernel);
  MAP_CUDART(cudaMalloc);
  MAP_CUDART(cudaFree);
  MAP_CUDART(cudaMemcpy);
  MAP_CUDART(cudaMemcpy2D);
  MAP_CUDART(cudaMemset);
  MAP_CUDART(cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags);
  MAP_CUDART(cudaDeviceSynchronize);
  MAP_CUDART(cudaGetLastError);
#undef MAP_CUDART
}

} // namespace ffi_internal
} // namespace enzymexla

#else

namespace enzymexla {
namespace ffi_internal {

void registerEnzymeJaXCudaRuntimeSymbols() {}

} // namespace ffi_internal
} // namespace enzymexla

#endif
