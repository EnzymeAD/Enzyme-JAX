#ifndef TESSERA_PASSES_H
#define TESSERA_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace tessera {

#define GEN_PASS_DECLS
#include "Tessera/Passes/Tessera/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "Tessera/Passes/Tessera/Passes.h.inc"

} // namespace tessera
} // namespace mlir

#endif