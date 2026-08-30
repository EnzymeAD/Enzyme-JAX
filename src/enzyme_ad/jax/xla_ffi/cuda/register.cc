#include "register.h"

namespace enzymexla {
namespace ffi_internal {

void registerEnzymeJaXXLACudaBlasFFI();
void registerEnzymeJaXXLACudaThrowErrorFFI();

void registerEnzymeJaXXLACudaFFI() {
  registerEnzymeJaXXLACudaThrowErrorFFI();
  registerEnzymeJaXXLACudaBlasFFI();
}

} // namespace ffi_internal
} // namespace enzymexla
