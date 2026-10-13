#ifndef ENZYMEXLA_COMM_DIALECT_OPS_TD
#define ENZYMEXLA_COMM_DIALECT_OPS_TD

// This file is a unifying header for all ops in the Comm dialect, to be mainly
// consumed by custom TableGen backends, like the Julia bindings generator.

include "MPIOps.td"
include "NCCLOps.td"

#endif // ENZYMEXLA_COMM_DIALECT_OPS_TD
