#include <string>
#include <string_view>

#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"
#include "xla/ffi/ffi_api.h"

#if defined(ENZYMEJAX_CUDA)
#include "third_party/gpus/cuda/include/cuda.h"
#include "third_party/gpus/cuda/include/cuda_runtime_api.h"

namespace enzymexla {
namespace ffi_internal {

xla::ffi::Error xlaThrowErrorCUDA(CUstream stream,
                                  xla::ffi::BufferR0<xla::ffi::PRED> cond,
                                  std::string_view message) {
  bool host_cond;
  cudaError_t err =
      cudaMemcpyAsync(&host_cond, cond.untyped_data(), sizeof(bool),
                      cudaMemcpyDeviceToHost, stream);
  if (err != cudaSuccess) {
    return xla::ffi::Error(xla::ffi::ErrorCode::kInternal,
                           "cudaMemcpyAsync failed in xlaThrowError");
  }
  err = cudaStreamSynchronize(stream);
  if (err != cudaSuccess) {
    return xla::ffi::Error(xla::ffi::ErrorCode::kInternal,
                           "cudaStreamSynchronize failed in xlaThrowError");
  }
  if (host_cond) {
    return xla::ffi::Error(xla::ffi::ErrorCode::kInternal,
                           std::string(message));
  }
  return xla::ffi::Error::Success();
}

xla::ffi::Error xlaAlwaysThrowErrorCUDA(CUstream stream,
                                        std::string_view message) {
  (void)stream;
  return xla::ffi::Error(xla::ffi::ErrorCode::kInternal, std::string(message));
}

XLA_FFI_DEFINE_HANDLER(xlaThrowErrorHandlerCUDA, xlaThrowErrorCUDA,
                       xla::ffi::Ffi::Bind()
                           .Ctx<xla::ffi::PlatformStream<CUstream>>()
                           .Arg<xla::ffi::BufferR0<xla::ffi::PRED>>()
                           .Attr<std::string_view>("message"));

XLA_FFI_DEFINE_HANDLER(xlaAlwaysThrowErrorHandlerCUDA, xlaAlwaysThrowErrorCUDA,
                       xla::ffi::Ffi::Bind()
                           .Ctx<xla::ffi::PlatformStream<CUstream>>()
                           .Attr<std::string_view>("message"));

void registerEnzymeJaXXLACudaThrowErrorFFI() {
  XLA_FFI_REGISTER_HANDLER(xla::ffi::GetXlaFfiApi(), "xla_throw_error", "CUDA",
                           xlaThrowErrorHandlerCUDA);
  XLA_FFI_REGISTER_HANDLER(xla::ffi::GetXlaFfiApi(), "xla_always_throw_error",
                           "CUDA", xlaAlwaysThrowErrorHandlerCUDA);
}

} // namespace ffi_internal
} // namespace enzymexla

#else

namespace enzymexla {
namespace ffi_internal {

void registerEnzymeJaXXLACudaThrowErrorFFI() {}

} // namespace ffi_internal
} // namespace enzymexla

#endif
