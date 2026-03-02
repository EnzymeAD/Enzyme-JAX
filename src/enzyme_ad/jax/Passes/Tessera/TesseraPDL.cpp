//===----------------------------------------------------------------------===//
//
// This file implements a pass to raise the generic tessera_op attribute
// into a proper MLIR Tessera convert attribute on the operation.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/PDLInterp/IR/PDLInterp.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "src/enzyme_ad/jax/Dialect/Tessera/Dialect.h"
#include "src/enzyme_ad/jax/Passes/Tessera/Passes.h"

namespace mlir {
namespace enzyme {
namespace tessera {
#define GEN_PASS_DEF_TESSERAPDLPASS
#include "src/enzyme_ad/jax/Passes/Tessera/Passes.h.inc"
} // namespace tessera
} // namespace enzyme
} // namespace mlir

using namespace mlir;
using namespace mlir::enzyme;
using namespace mlir::enzyme::tessera;

namespace {

struct TesseraPDLPass
    : public enzyme::tessera::impl::TesseraPDLPassBase<TesseraPDLPass> {
  using TesseraPDLPassBase::TesseraPDLPassBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();

    ModuleOp patternModule = module.lookupSymbol<ModuleOp>(
        StringAttr::get(module->getContext(), "patterns"));
    ModuleOp irModule = module.lookupSymbol<ModuleOp>(
        StringAttr::get(module->getContext(), "ir"));

    if (!patternModule || !irModule)
      return;

    RewritePatternSet patternList(module->getContext());

    // Process the pattern module.
    patternModule.getOperation()->remove();
    PDLPatternModule pdlPattern(patternModule);

    patternList.add(std::move(pdlPattern));

    // Invoke the pattern driver with the provided patterns.
    (void)applyPatternsGreedily(irModule.getBodyRegion(),
                                std::move(patternList));
  }
};

} // namespace