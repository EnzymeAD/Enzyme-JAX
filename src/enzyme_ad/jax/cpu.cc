#include <cstring>
#include <string_view>

#include "xla/ffi/api/ffi.h"
#include "xla/ffi/ffi_api.h"

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

namespace ffi = xla::ffi;

template <bool withError> struct CallInfo;

template <> struct CallInfo<false> {
  void (*run)(const void **);
};

template <> struct CallInfo<true> {
  char *(*run)(const void **);
};

template <bool withError>
static ffi::Error execute(std::string_view attr, ffi::RemainingArgs args,
                          ffi::RemainingRets) {
  // `attr` holds the raw bytes of the CallInfo built by lower-jit.
  if (attr.size() != sizeof(CallInfo<withError>))
    return ffi::Error::InvalidArgument(
        "enzymexla_compile_cpu: unexpected call info size");
  CallInfo<withError> cinfo;
  std::memcpy(&cinfo, attr.data(), sizeof(cinfo));

  // Results alias operands, so the jitted function only takes the operands.
  size_t numargs = args.size();
  const void *ptrs[numargs];
  for (size_t i = 0; i < numargs; i++) {
    auto buf = args.get<ffi::AnyBuffer>(i);
    if (!buf.has_value())
      return buf.error();
    ptrs[i] = buf->untyped_data();
  }

  if constexpr (withError) {
    if (char *err = cinfo.run(ptrs))
      return ffi::Error::Internal(err);
  } else {
    cinfo.run(ptrs);
  }

  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER(kExecute, execute<false>,
                       ffi::Ffi::Bind()
                           .Attr<std::string_view>("attr")
                           .RemainingArgs()
                           .RemainingRets());

XLA_FFI_DEFINE_HANDLER(kExecuteWithError, execute<true>,
                       ffi::Ffi::Bind()
                           .Attr<std::string_view>("attr")
                           .RemainingArgs()
                           .RemainingRets());

extern "C" MLIR_CAPI_EXPORTED void RegisterEnzymeXLACPUHandler() {
  XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "enzymexla_compile_cpu", "Host",
                           kExecute);
  XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(),
                           "enzymexla_compile_cpu_with_error", "Host",
                           kExecuteWithError);
}
