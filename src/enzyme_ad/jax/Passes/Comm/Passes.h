#ifndef ENZYMEXLA_PASSES_COMM_PASSES_H
#define ENZYMEXLA_PASSES_COMM_PASSES_H

#include "mlir/Pass/Pass.h"
#include "src/enzyme_ad/jax/Dialect/Comm/Dialect.h"
#include "src/enzyme_ad/jax/Dialect/Dialect.h"

namespace mlir::comm {
#define GEN_PASS_DECL
#include "src/enzyme_ad/jax/Passes/Comm/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "src/enzyme_ad/jax/Passes/Comm/Passes.h.inc"
} // namespace mlir::comm

#endif // ENZYMEXLA_PASSES_COMM_PASSES_H
