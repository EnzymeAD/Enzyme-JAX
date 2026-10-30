#include "register.h"

namespace enzymexla {
namespace ffi_internal {

void registerEnzymeJaXRocmRuntimeSymbols();

void registerEnzymeJaXXLARocmFFI() { registerEnzymeJaXRocmRuntimeSymbols(); }

} // namespace ffi_internal
} // namespace enzymexla
