#ifndef ENZYME_AD_JAX_PASSES_BLAS_PASSES_H
#define ENZYME_AD_JAX_PASSES_BLAS_PASSES_H

#include "mlir/Pass/Pass.h"
#include "src/enzyme_ad/jax/Dialect/BLAS/Dialect.h"

namespace mlir::enzymexla::blas {

#define GEN_PASS_DECL
#include "src/enzyme_ad/jax/Passes/BLAS/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "src/enzyme_ad/jax/Passes/BLAS/Passes.h.inc"

} // namespace mlir::enzymexla::tessera

#endif // ENZYME_AD_JAX_PASSES_BLAS_PASSES_H
