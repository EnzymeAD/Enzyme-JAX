namespace mlir {
class RewritePatternSet;
class MLIRContext;
} // namespace mlir

#include "src/enzyme_ad/jax/Passes/EnzymeHLOPatterns.h.inc"

namespace mlir::transform {
void addPadDotGeneral(RewritePatternSet &patterns, bool postPad,
                      MLIRContext &context);
}
