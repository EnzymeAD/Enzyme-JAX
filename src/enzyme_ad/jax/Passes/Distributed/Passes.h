#ifndef ENZYMEXLA_DISTRIBUTED_PASSES_H
#define ENZYMEXLA_DISTRIBUTED_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir::enzyme::distributed {

#define GEN_PASS_DECL
#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h.inc"

} // namespace mlir::enzyme::distributed

#endif // ENZYMEXLA_DISTRIBUTED_PASSES_H