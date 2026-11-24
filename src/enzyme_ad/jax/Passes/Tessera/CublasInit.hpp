#include <cublas_v2.h>
#include <cuda_runtime.h>   

__attribute__((tessera_op("cublas.getHandle()")))
cublasHandle_t* getHandle() {
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    if (deviceCount > 0) {
        cublasHandle_t* handle = malloc(sizeof(cublasHandle_t));
        cublasCreate(handle);
        return handle;
    } else {
        return nullptr;
    }
}
  