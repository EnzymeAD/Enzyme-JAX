#include <string>
#include <string_view>

#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"
#include "xla/ffi/ffi_api.h"

#if defined(ENZYMEJAX_ROCM)
#define __HIP_DISABLE_CPP_FUNCTIONS__
#include "rocm/include/hip/hip_runtime_api.h"

namespace enzymexla {
namespace ffi_internal {

xla::ffi::Error xlaThrowErrorROCM(hipStream_t stream,
                                  xla::ffi::BufferR0<xla::ffi::PRED> cond,
                                  std::string_view message) {
  bool host_cond;
  hipError_t err = hipMemcpyAsync(&host_cond, cond.untyped_data(), sizeof(bool),
                                  hipMemcpyDeviceToHost, stream);
  if (err != hipSuccess) {
    return xla::ffi::Error(xla::ffi::ErrorCode::kInternal,
                           "hipMemcpyAsync failed in xlaThrowError");
  }
  err = hipStreamSynchronize(stream);
  if (err != hipSuccess) {
    return xla::ffi::Error(xla::ffi::ErrorCode::kInternal,
                           "hipStreamSynchronize failed in xlaThrowError");
  }
  if (host_cond) {
    return xla::ffi::Error(xla::ffi::ErrorCode::kInternal,
                           std::string(message));
  }
  return xla::ffi::Error::Success();
}

xla::ffi::Error xlaAlwaysThrowErrorROCM(hipStream_t stream,
                                        std::string_view message) {
  (void)stream;
  return xla::ffi::Error(xla::ffi::ErrorCode::kInternal, std::string(message));
}

XLA_FFI_DEFINE_HANDLER(xlaThrowErrorHandlerROCM, xlaThrowErrorROCM,
                       xla::ffi::Ffi::Bind()
                           .Ctx<xla::ffi::PlatformStream<hipStream_t>>()
                           .Arg<xla::ffi::BufferR0<xla::ffi::PRED>>()
                           .Attr<std::string_view>("message"));

XLA_FFI_DEFINE_HANDLER(xlaAlwaysThrowErrorHandlerROCM, xlaAlwaysThrowErrorROCM,
                       xla::ffi::Ffi::Bind()
                           .Ctx<xla::ffi::PlatformStream<hipStream_t>>()
                           .Attr<std::string_view>("message"));

void registerEnzymeJaXXLARocmThrowErrorFFI() {
  XLA_FFI_REGISTER_HANDLER(xla::ffi::GetXlaFfiApi(), "xla_throw_error", "ROCM",
                           xlaThrowErrorHandlerROCM);
  XLA_FFI_REGISTER_HANDLER(xla::ffi::GetXlaFfiApi(), "xla_always_throw_error",
                           "ROCM", xlaAlwaysThrowErrorHandlerROCM);
}

} // namespace ffi_internal
} // namespace enzymexla

#else

namespace enzymexla {
namespace ffi_internal {

void registerEnzymeJaXXLARocmThrowErrorFFI() {}

} // namespace ffi_internal
} // namespace enzymexla

#endif
