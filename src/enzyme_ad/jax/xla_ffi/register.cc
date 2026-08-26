#include "cuda/register.h"
#include "absl/base/call_once.h"
#include "export_macro.h"
#include "host/register.h"

extern "C" MLIR_CAPI_EXPORTED void registerEnzymeJaXXLAFFI() {
  static absl::once_flag once;
  absl::call_once(once, []() {
    enzymexla::ffi_internal::registerEnzymeJaXXLAHostFFI();
    enzymexla::ffi_internal::registerEnzymeJaXXLACudaFFI();
  });
}
