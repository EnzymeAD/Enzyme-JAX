#ifndef TESSERA_PASSES_H
#define TESSERA_PASSES_H

#include "mlir/Pass/Pass.h"
#include "src/enzyme_ad/jax/Dialect/Tessera/Dialect.h"

namespace mlir {
namespace enzyme {
namespace tessera {

std::unique_ptr<mlir::Pass> createTesseraAnnotationToAttributePass();
std::unique_ptr<mlir::Pass> createFuncAttrToTesseraAttrPass();
std::unique_ptr<mlir::Pass> createTesseraToFuncPass();
std::unique_ptr<mlir::Pass> createFuncToTesseraPass();

} // namespace tessera
} // namespace enzyme
} // namespace mlir

#define GEN_PASS_DECLS
#include "src/enzyme_ad/jax/Passes/Tessera/Passes.h.inc"

namespace mlir {
namespace enzyme {
namespace tessera {

#define GEN_PASS_REGISTRATION
#include "src/enzyme_ad/jax/Passes/Tessera/Passes.h.inc"

} // namespace tessera
} // namespace enzyme
} // namespace mlir

#endif
