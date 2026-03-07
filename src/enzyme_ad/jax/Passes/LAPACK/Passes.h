#ifndef ENZYME_AD_JAX_PASSES_LAPACK_PASSES_H
#define ENZYME_AD_JAX_PASSES_LAPACK_PASSES_H

#include "mlir/Pass/Pass.h"
#include "src/enzyme_ad/jax/Dialect/LAPACK/Dialect.h"

namespace mlir::lapack {

#define GEN_PASS_DECL
#include "src/enzyme_ad/jax/Passes/LAPACK/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "src/enzyme_ad/jax/Passes/LAPACK/Passes.h.inc"

} // namespace mlir::lapack

#endif // ENZYME_AD_JAX_PASSES_LAPACK_PASSES_H
