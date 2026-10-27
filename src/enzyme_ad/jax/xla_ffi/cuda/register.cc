#include "register.h"

namespace enzymexla {
namespace ffi_internal {

void registerEnzymeJaXXLACudaBlasFFI();
void registerEnzymeJaXXLACudaThrowErrorFFI();
void registerEnzymeJaXCudaRuntimeSymbols();

void registerEnzymeJaXXLACudaFFI() {
  registerEnzymeJaXXLACudaThrowErrorFFI();
  registerEnzymeJaXXLACudaBlasFFI();
  registerEnzymeJaXCudaRuntimeSymbols();
}

} // namespace ffi_internal
} // namespace enzymexla
