#include "register.h"
#include "mpi_ffi.h"
#include "throw_error.h"

namespace enzymexla {
namespace ffi_internal {

void registerEnzymeJaXXLAHostFFI() {
  registerEnzymeJaXXLAHostThrowErrorFFI();
#ifndef _WIN32
  // MPItrampoline is POSIX-only (dlfcn), so Windows builds skip MPI.
  registerEnzymeJaXXLAHostMPIFFI();
#endif
}

} // namespace ffi_internal
} // namespace enzymexla
