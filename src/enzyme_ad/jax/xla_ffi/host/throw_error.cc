#include <string>
#include <string_view>

#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"
#include "xla/ffi/ffi_api.h"

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

void registerEnzymeJaXXLAHostThrowErrorFFI() {
  XLA_FFI_REGISTER_HANDLER(xla::ffi::GetXlaFfiApi(), "xla_throw_error", "Host",
                           xlaThrowErrorHandlerHost);
  XLA_FFI_REGISTER_HANDLER(xla::ffi::GetXlaFfiApi(), "xla_always_throw_error",
                           "Host", xlaAlwaysThrowErrorHandlerHost);
}

} // namespace ffi_internal
} // namespace enzymexla
