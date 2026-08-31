#include <cstdint>

#include "absl/strings/str_format.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"
#include "xla/ffi/ffi_api.h"

#include "../export_macro.h"

// libblastrampoline C API wrapper
extern "C" MLIR_CAPI_EXPORTED int32_t
enzymexla_lbt_forward(const char *libname, int32_t clear, int32_t verbose,
                      const char *suffix_hint) {
  return lbt_forward(libname, clear, verbose, suffix_hint);
}

extern "C" MLIR_CAPI_EXPORTED const lbt_config_t *enzymexla_lbt_get_config() {
  return lbt_get_config();
}

extern "C" MLIR_CAPI_EXPORTED int32_t enzymexla_lbt_get_num_threads() {
  return lbt_get_num_threads();
}

extern "C" MLIR_CAPI_EXPORTED void
enzymexla_lbt_set_num_threads(int32_t num_threads) {
  lbt_set_num_threads(num_threads);
}

extern "C" MLIR_CAPI_EXPORTED void
enzymexla_lbt_register_thread_interface(const char *getter,
                                        const char *setter) {
  lbt_register_thread_interface(getter, setter);
}

extern "C" MLIR_CAPI_EXPORTED void enzymxla_lbt_default_func_print_error() {
  lbt_default_func_print_error();
}

extern "C" MLIR_CAPI_EXPORTED void
enzymexla_lbt_default_func_print_error_and_exit() {
  lbt_default_func_print_error_and_exit();
}

extern "C" MLIR_CAPI_EXPORTED const void *enzymexla_lbt_get_default_func() {
  return lbt_get_default_func();
}

extern "C" MLIR_CAPI_EXPORTED void
enzymexla_lbt_set_default_func(const void *addr) {
  lbt_set_default_func(addr);
}

extern "C" MLIR_CAPI_EXPORTED uint8_t enzymexla_lbt_get_use_deepbind() {
  return lbt_get_use_deepbind();
}

extern "C" MLIR_CAPI_EXPORTED const void *
enzymexla_lbt_get_forward(const char *symbol_name, int32_t interface,
                          int32_t f2c) {
  return lbt_get_forward(symbol_name, interface, f2c);
}

extern "C" MLIR_CAPI_EXPORTED int32_t enzymexla_lbt_set_forward(
    const char *symbol_name, const void *addr, int32_t interface,
    int32_t complex_retstyle, int32_t f2c, int32_t verbose) {
  return lbt_set_forward(symbol_name, addr, interface, complex_retstyle, f2c,
                         verbose);
}
