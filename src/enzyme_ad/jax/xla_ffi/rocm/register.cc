#include "register.h"

namespace enzymexla {
namespace ffi_internal {

void registerEnzymeJaXRocmRuntimeSymbols();
void registerEnzymeJaXXLARocmThrowErrorFFI();

void registerEnzymeJaXXLARocmFFI() {
  registerEnzymeJaXRocmRuntimeSymbols();
  registerEnzymeJaXXLARocmThrowErrorFFI();
}

} // namespace ffi_internal
} // namespace enzymexla
