#include <mutex>
#include <string>
#include <string_view>

#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"
#include "xla/ffi/ffi_api.h"

#if defined(GOOGLE_CUDA)
#include "third_party/gpus/cuda/include/cuda.h"
#include "third_party/gpus/cuda/include/cuda_runtime_api.h"
#endif

#if (defined(_WIN32) || defined(__CYGWIN__)) &&                                \
    !defined(MLIR_CAPI_ENABLE_WINDOWS_DLL_DECLSPEC)
// Visibility annotations disabled.
#define MLIR_CAPI_EXPORTED
#elif defined(_WIN32) || defined(__CYGWIN__)
// Windows visibility declarations.
#if MLIR_CAPI_BUILDING_LIBRARY
#define MLIR_CAPI_EXPORTED __declspec(dllexport)
#else
#define MLIR_CAPI_EXPORTED __declspec(dllimport)
#endif
#else
// Non-windows: use visibility attributes.
#define MLIR_CAPI_EXPORTED __attribute__((visibility("default")))
#endif

namespace enzymexla {
namespace ffi_internal {

xla::ffi::Error xlaThrowError(std::string_view message) {
  return xla::ffi::Error(xla::ffi::ErrorCode::kInternal, std::string(message));
}

xla::ffi::Error xlaThrowError(bool cond, std::string_view message) {
  if (cond) {
    return xlaThrowError(message);
  }
  return xla::ffi::Error::Success();
}

xla::ffi::Error xlaThrowErrorHost(xla::ffi::BufferR0<xla::ffi::PRED> cond,
                                  std::string_view message) {
  return xlaThrowError(cond.typed_data()[0], message);
}

xla::ffi::Error xlaAlwaysThrowErrorHost(std::string_view message) {
  return xlaThrowError(message);
}

XLA_FFI_DEFINE_HANDLER(xlaThrowErrorHandlerHost, xlaThrowErrorHost,
                       xla::ffi::Ffi::Bind()
                           .Arg<xla::ffi::BufferR0<xla::ffi::PRED>>()
                           .Attr<std::string_view>("message"));

XLA_FFI_DEFINE_HANDLER(xlaAlwaysThrowErrorHandlerHost, xlaAlwaysThrowErrorHost,
                       xla::ffi::Ffi::Bind().Attr<std::string_view>("message"));

#if defined(GOOGLE_CUDA)

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
  return xlaThrowError(host_cond, message);
}

xla::ffi::Error xlaAlwaysThrowErrorCUDA(CUstream stream,
                                        std::string_view message) {
  (void)stream;
  return xlaThrowError(message);
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

#endif

void registerEnzymeJaXXLAInternalFFI() {
  static std::once_flag once;
  std::call_once(once, []() {
    XLA_FFI_REGISTER_HANDLER(xla::ffi::GetXlaFfiApi(), "xla_throw_error",
                             "Host", xlaThrowErrorHandlerHost);
    XLA_FFI_REGISTER_HANDLER(xla::ffi::GetXlaFfiApi(),
                             "xla_always_throw_error", "Host",
                             xlaAlwaysThrowErrorHandlerHost);

#if defined(GOOGLE_CUDA)
    XLA_FFI_REGISTER_HANDLER(xla::ffi::GetXlaFfiApi(), "xla_throw_error",
                             "CUDA", xlaThrowErrorHandlerCUDA);
    XLA_FFI_REGISTER_HANDLER(xla::ffi::GetXlaFfiApi(),
                             "xla_always_throw_error", "CUDA",
                             xlaAlwaysThrowErrorHandlerCUDA);
#endif
  });
}

} // namespace ffi_internal
} // namespace enzymexla

extern "C" MLIR_CAPI_EXPORTED void registerEnzymeJaXXLAFFI() {
  enzymexla::ffi_internal::registerEnzymeJaXXLAInternalFFI();
}
