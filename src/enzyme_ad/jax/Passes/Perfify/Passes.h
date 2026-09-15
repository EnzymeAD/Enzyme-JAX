#ifndef PERFIFY_PASSES_H
#define PERFIFY_PASSES_H

#include "mlir/Pass/Pass.h"
#include "src/enzyme_ad/jax/Dialect/Perfify/Dialect.h"

namespace mlir {
namespace enzyme {
namespace perfify {

#define GEN_PASS_DECL
#include "src/enzyme_ad/jax/Passes/Perfify/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "src/enzyme_ad/jax/Passes/Perfify/Passes.h.inc"

} // namespace perfify
} // namespace enzyme
} // namespace mlir

#endif
