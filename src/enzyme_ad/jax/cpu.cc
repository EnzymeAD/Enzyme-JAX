#include "xla/service/custom_call_status.h"
#include "xla/service/custom_call_target_registry.h"
#include <cstring>

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

template <bool withError> struct CallInfo;

template <> struct CallInfo<false> {
  void (*run)(const void **);
};

template <> struct CallInfo<true> {
  char *(*run)(const void **);
};

template <bool withError>
void forwarding_custom_call(void *out, const void **in, const void *opaque_ptr,
                            size_t opaque_len, XlaCustomCallStatus *status) {
  const CallInfo<withError> *opaque =
      static_cast<const CallInfo<withError> *>(opaque_ptr);

  if constexpr (withError) {
    char *err = opaque->run(in);
    if (err) {
      XlaCustomCallStatusSetFailure(status, err, strlen(err));
    }
  } else {
    opaque->run(in);
  }
}

extern "C" MLIR_CAPI_EXPORTED void RegisterEnzymeXLACPUHandler() {
  xla::CustomCallTargetRegistry::Global()->Register(
      "enzymexla_compile_cpu", (void *)&forwarding_custom_call<false>, "Host");
  xla::CustomCallTargetRegistry::Global()->Register(
      "enzymexla_compile_cpu_with_error", (void *)&forwarding_custom_call<true>,
      "Host");
}
