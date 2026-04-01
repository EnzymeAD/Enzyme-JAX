#ifndef TESSERA_PASSES_H
#define TESSERA_PASSES_H

#include "mlir/Pass/Pass.h"
#include "src/enzyme_ad/jax/Dialect/Tessera/Dialect.h"

namespace mlir {
namespace enzyme {
namespace tessera {

#define GEN_PASS_DECL
#include "src/enzyme_ad/jax/Passes/Tessera/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "src/enzyme_ad/jax/Passes/Tessera/Passes.h.inc"

} // namespace tessera
} // namespace enzyme
} // namespace mlir

#endif
