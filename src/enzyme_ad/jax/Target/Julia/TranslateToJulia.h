#ifndef ENZYME_AD_JAX_TARGET_JULIA_TRANSLATETOJULIA_H
#define ENZYME_AD_JAX_TARGET_JULIA_TRANSLATETOJULIA_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::enzymexla::julia {

/// Emit standalone native Julia source code for `op` (either a `ModuleOp`
/// containing `julia.func` functions or a single `julia.func`).
LogicalResult translateToJulia(Operation *op, llvm::raw_ostream &os);

/// Register `--mlir-to-julia` with MLIR's translation registry.
void registerToJuliaTranslation();

} // namespace mlir::enzymexla::julia

#endif // ENZYME_AD_JAX_TARGET_JULIA_TRANSLATETOJULIA_H
