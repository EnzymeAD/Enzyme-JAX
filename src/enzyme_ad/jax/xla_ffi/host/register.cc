#include "register.h"
#include "throw_error.h"
#include "mpi.h"

namespace enzymexla {
namespace ffi_internal {

void registerEnzymeJaXXLAHostFFI() {
    registerEnzymeJaXXLAHostThrowErrorFFI();
    registerEnzymeJaXXLAHostMPIFFI();
}

} // namespace ffi_internal
} // namespace enzymexla
