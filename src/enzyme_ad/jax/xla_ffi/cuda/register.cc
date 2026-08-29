#include "register.h"
#include "blas.h"
#include "throw_error.h"

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
