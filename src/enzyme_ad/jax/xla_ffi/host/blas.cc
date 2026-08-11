#include "blas.h"
#include "libblastrampoline.h"

#include "absl/strings/str_format.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"
#include "xla/ffi/ffi_api.h"

namespace enzymexla::ffi_internal {} // namespace enzymexla::ffi_internal

int32_t enzymexla_lbt_forward(const char *libname, int32_t clear,
                              int32_t verbose, const char *suffix_hint) {
  return lbt_forward(libname, clear, verbose, suffix_hint);
}

int32_t enzymexla_lbt_get_num_threads() { return lbt_get_num_threads(); }

void enzymexla_lbt_set_num_threads(int32_t num_threads) {
  lbt_set_num_threads(num_threads);
}
