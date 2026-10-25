#ifndef TESSERA_PASSES_H
#define TESSERA_PASSES_H

#include "mlir/Pass/Pass.h"
#include "src/enzyme_ad/jax/Dialect/Tessera/Dialect.h"

namespace mlir {
namespace enzyme {
namespace tessera {

// On a function with a tessera_op marking: for each argument (sret excluded),
// the pointee type a lifted argument is loaded as, or a unit attribute.
constexpr llvm::StringLiteral kArgTypesAttr = "tessera.arg_types";

#define GEN_PASS_DECL
#include "src/enzyme_ad/jax/Passes/Tessera/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "src/enzyme_ad/jax/Passes/Tessera/Passes.h.inc"

} // namespace tessera
} // namespace enzyme
} // namespace mlir

#endif
