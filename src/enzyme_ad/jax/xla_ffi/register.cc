#include <mutex>

#include "host/register.h"
#include "cuda/register.h"

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

extern "C" MLIR_CAPI_EXPORTED void registerEnzymeJaXXLAFFI() {
  static std::once_flag once;
  std::call_once(once, []() {
    enzymexla::ffi_internal::registerEnzymeJaXXLAHostFFI();
#if defined(GOOGLE_CUDA)
    enzymexla::ffi_internal::registerEnzymeJaXXLACudaFFI();
#endif
  });
}
