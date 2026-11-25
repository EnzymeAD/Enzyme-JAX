#include "register.h"

namespace enzymexla {
namespace ffi_internal {

void registerEnzymeJaXXLACudaBlasFFI();
void registerEnzymeJaXXLACudaSparseFFI();
void registerEnzymeJaXXLACudaThrowErrorFFI();
void registerEnzymeJaXCudaRuntimeSymbols();

void registerEnzymeJaXXLACudaFFI() {
  registerEnzymeJaXXLACudaThrowErrorFFI();
  registerEnzymeJaXXLACudaBlasFFI();
  registerEnzymeJaXXLACudaSparseFFI();
  registerEnzymeJaXCudaRuntimeSymbols();
}

} // namespace ffi_internal
} // namespace enzymexla
