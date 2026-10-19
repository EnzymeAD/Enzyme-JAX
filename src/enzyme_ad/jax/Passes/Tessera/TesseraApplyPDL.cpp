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

static LogicalResult isFloatConstantEqualTo(PatternRewriter &rewriter,
                                            PDLResultList &results,
                                            ArrayRef<PDLValue> args) {
  // args[0]: the matched Operation* that should be producing a constant
  // args[1]: the expected value, passed as a PDL attribute (FloatAttr, always
  //          emitted as f64 by the rule parser)
  Operation *constOp = args[0].cast<Operation *>();
  auto expectedAttr = args[1].cast<Attribute>();

  // As in the integer case, the op name is not pinned on the pattern side, so
  // this is where "is it a float constant?" is decided — via the constant
  // interface, so the same rule fires on llvm.mlir.constant and arith.constant.
  if (constOp->getNumResults() != 1)
    return failure();

  llvm::APFloat actual(0.0);
  if (!matchPattern(constOp->getResult(0), m_ConstantFloat(&actual)))
    return failure();

  auto expectedFloatAttr = dyn_cast<FloatAttr>(expectedAttr);
  if (!expectedFloatAttr)
    return failure();

  // Round the expected value of the float into the semantics the matched
  // constant actually uses, then compare exactly.
  llvm::APFloat expected = expectedFloatAttr.getValue();
  bool losesInfo = false;
  expected.convert(actual.getSemantics(), llvm::APFloat::rmNearestTiesToEven,
                   &losesInfo);

  // bitwiseIsEqual rather than ==: it keeps +0.0 distinct from -0.0 and lets a
  // NaN literal match a NaN constant, where == would do the opposite on both.
  if (!actual.bitwiseIsEqual(expected))
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
    pdlPattern.registerConstraintFunction("isFloatConstantEqualTo",
                                          isFloatConstantEqualTo);

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