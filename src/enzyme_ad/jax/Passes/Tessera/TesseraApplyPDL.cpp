//===----------------------------------------------------------------------===//
//
// This file implements a pass to apply the PDL patterns created from the
// tessera optimization rewrite rules to the IR.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/PDL/IR/PDL.h"
#include "mlir/Dialect/PDLInterp/IR/PDLInterp.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "src/enzyme_ad/jax/Dialect/Tessera/Dialect.h"
#include "src/enzyme_ad/jax/Passes/Tessera/Passes.h"

namespace mlir {
namespace enzyme {
namespace tessera {
#define GEN_PASS_DEF_TESSERAAPPLYPDLPASS
#include "src/enzyme_ad/jax/Passes/Tessera/Passes.h.inc"
} // namespace tessera
} // namespace enzyme
} // namespace mlir

using namespace mlir;
using namespace mlir::enzyme;
using namespace mlir::enzyme::tessera;

namespace {

static LogicalResult isConstantEqualTo(PatternRewriter &rewriter,
                                       PDLResultList &results,
                                       ArrayRef<PDLValue> args) {
  // args[0]: the matched Operation* that should be producing a constant
  // args[1]: the expected value, passed as a PDL attribute (IntegerAttr)
  Operation *constOp = args[0].cast<Operation *>();
  auto expectedAttr = args[1].cast<Attribute>();

  // The pattern side deliberately does not pin the op name, so this is where
  // "is it a constant?" is actually decided. Match through the constant
  // interface rather than a concrete op class: the same rule has to fire both
  // on llvm.mlir.constant and on arith.constant.
  if (constOp->getNumResults() != 1)
    return failure();

  llvm::APInt actual;
  if (!matchPattern(constOp->getResult(0), m_ConstantInt(&actual)))
    return failure();

  auto expectedIntAttr = dyn_cast<IntegerAttr>(expectedAttr);
  if (!expectedIntAttr)
    return failure();

  // Compare numeric value only, ignoring bit-width, so this stays robust
  // if the constant's width ever differs from what the annotation assumed.
  if (actual.getSExtValue() != expectedIntAttr.getValue().getSExtValue())
    return failure();

  return success();
}

struct TesseraApplyPDLPass
    : public enzyme::tessera::impl::TesseraApplyPDLPassBase<
          TesseraApplyPDLPass> {
  using TesseraApplyPDLPassBase::TesseraApplyPDLPassBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();

    ModuleOp patternModule = module.lookupSymbol<ModuleOp>(
        StringAttr::get(module->getContext(), "patterns"));

    if (!patternModule)
      return;

    if (patternModule.getBody()->getOperations().empty()) {
      patternModule.getOperation()->erase();
      return;
    }

    RewritePatternSet patternList(module->getContext());

    // Process the pattern module.
    patternModule.getOperation()->remove();
    PDLPatternModule pdlPattern(patternModule);

    // Register native constraints referenced by generated PDL patterns.
    pdlPattern.registerConstraintFunction("isConstantEqualTo",
                                          isConstantEqualTo);

    patternList.add(std::move(pdlPattern));

    // Invoke the pattern driver with the provided patterns.
    if (failed(applyPatternsGreedily(
            module, std::move(patternList),
            GreedyRewriteConfig().setRegionSimplificationLevel(
                GreedySimplifyRegionLevel::Normal)))) {
      llvm::errs() << "Failed to apply PDL patterns\n";
      signalPassFailure();
    }
  }
};

} // namespace