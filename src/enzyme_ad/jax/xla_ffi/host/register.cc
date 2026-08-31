#include "register.h"

namespace enzymexla {
namespace ffi_internal {

void registerEnzymeJaXXLAHostThrowErrorFFI();
void registerEnzymeJaXXLAHostBLASFFI();
void registerEnzymeJaXXLAHostMPIFFI();

void registerEnzymeJaXXLAHostFFI() {
  registerEnzymeJaXXLAHostThrowErrorFFI();
  registerEnzymeJaXXLAHostBLASFFI();
#ifndef _WIN32
  // MPItrampoline is POSIX-only (dlfcn), so Windows builds skip MPI.
  registerEnzymeJaXXLAHostMPIFFI();
#endif
}

} // namespace ffi_internal
} // namespace enzymexla
