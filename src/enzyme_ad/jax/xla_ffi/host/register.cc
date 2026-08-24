#include "register.h"
#include "mpi_ffi.h"
#include "throw_error.h"

namespace enzymexla {
namespace ffi_internal {

void registerEnzymeJaXXLAHostFFI() {
  registerEnzymeJaXXLAHostThrowErrorFFI();
  registerEnzymeJaXXLAHostMPIFFI();
}

} // namespace ffi_internal
} // namespace enzymexla
