#include "register.h"

namespace enzymexla {
namespace ffi_internal {

void registerEnzymeJaXRocmRuntimeSymbols();
void registerEnzymeJaXXLARocmThrowErrorFFI();
void registerEnzymeJaXXLARocmSparseFFI();

void registerEnzymeJaXXLARocmFFI() {
  registerEnzymeJaXRocmRuntimeSymbols();
  registerEnzymeJaXXLARocmThrowErrorFFI();
  registerEnzymeJaXXLARocmSparseFFI();
}

} // namespace ffi_internal
} // namespace enzymexla
