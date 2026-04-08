#ifndef PACT_PASSES_H
#define PACT_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace enzyme {
namespace pact {

#define GEN_PASS_DECL
#include "src/enzyme_ad/jax/Passes/Pact/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "src/enzyme_ad/jax/Passes/Pact/Passes.h.inc"

} // namespace pact
} // namespace enzyme
} // namespace mlir

#endif
