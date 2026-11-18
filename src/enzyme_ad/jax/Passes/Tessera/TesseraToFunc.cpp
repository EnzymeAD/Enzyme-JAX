//===----------------------------------------------------------------------===//
//
// This file implements patterns to convert operations in the Tessera dialect to
// operations in the Func dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "src/enzyme_ad/jax/Dialect/Tessera/Dialect.h"
#include "src/enzyme_ad/jax/Passes/Tessera/Passes.h"

namespace mlir {
namespace enzyme {
namespace tessera {
#define GEN_PASS_DEF_TESSERATOFUNCPASS
#include "src/enzyme_ad/jax/Passes/Tessera/Passes.h.inc"
} // namespace tessera
} // namespace enzyme
} // namespace mlir

using namespace mlir;
using namespace mlir::enzyme;
using namespace mlir::enzyme::tessera;

//===----------------------------------------------------------------------===//
// Rewrite Patterns
//===----------------------------------------------------------------------===//

namespace {

// Rewrite 'tessera.define' -> 'func.func'
class DefineOpRewrite final : public OpRewritePattern<tessera::DefineOp> {
public:
  using OpRewritePattern<tessera::DefineOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tessera::DefineOp defineOp,
                                PatternRewriter &rewriter) const override {
    auto funcNameAttr =
        defineOp->getAttrOfType<StringAttr>("tessera.original_name");
    if (!funcNameAttr)
      return failure();
    auto module = defineOp->getParentOfType<ModuleOp>();

    // func.func passes values, so there is nothing to put back into memory:
    // a define lifted from llvm.func that loads arguments from pointers
    // needs tessera-to-llvm.
    if (defineOp.getSretAttr() ||
        llvm::any_of(defineOp.getArgModeEntries(),
                     [](Attribute mode) { return !isa<UnitAttr>(mode); }))
      return rewriter.notifyMatchFailure(
          defineOp, "arguments lifted out of memory need tessera-to-llvm");

    // Replace tessera name with original function name
    if (failed(SymbolTable::replaceAllSymbolUses(defineOp.getSymNameAttr(),
                                                 funcNameAttr, module)))
      return failure();

    auto funcOp = func::FuncOp::create(rewriter, defineOp.getLoc(),
                                       funcNameAttr.getValue(),
                                       defineOp.getFunctionType());

    // Copy over attributes other than the function name and type, arg modes,
    // pure, and other attributes used only for tessera conversion
    for (const auto &namedAttr : defineOp->getAttrs()) {
      if (namedAttr.getName() != SymbolTable::getSymbolAttrName() &&
          namedAttr.getName() != defineOp.getFunctionTypeAttrName() &&
          namedAttr.getName() != defineOp.getArgModesAttrName() &&
          namedAttr.getName() != defineOp.getPureAttrName() &&
          namedAttr.getName() != "tessera.original_name")
        funcOp->setAttr(namedAttr.getName(), namedAttr.getValue());
    }

    if (!defineOp.isExternal())
      rewriter.inlineRegionBefore(defineOp.getBody(), funcOp.getBody(),
                                  funcOp.end());

    rewriter.eraseOp(defineOp);
    return success();
  }
};

// Rewrite 'tessera.call' -> 'func.call'
class CallOpRewrite final : public OpRewritePattern<tessera::CallOp> {
public:
  using OpRewritePattern<tessera::CallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tessera::CallOp callOp,
                                PatternRewriter &rewriter) const override {
    auto newCall =
        func::CallOp::create(rewriter, callOp.getLoc(), callOp.getCalleeAttr(),
                             callOp.getResultTypes(), callOp.getOperands());
    for (auto attr : callOp->getAttrs())
      if (attr.getName() != callOp.getCalleeAttrName())
        newCall->setAttr(attr.getName(), attr.getValue());
    rewriter.replaceOp(callOp, newCall);
    return success();
  }
};

// Rewrite 'tessera.return' -> 'func.return'
class ReturnOpRewrite final : public OpRewritePattern<tessera::ReturnOp> {
public:
  using OpRewritePattern<tessera::ReturnOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tessera::ReturnOp returnOp,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<func::ReturnOp>(returnOp,
                                                returnOp.getOperands());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass to convert Tessera operations into Func operations
//===----------------------------------------------------------------------===//

struct TesseraToFuncPass
    : public enzyme::tessera::impl::TesseraToFuncPassBase<TesseraToFuncPass> {
  using TesseraToFuncPassBase::TesseraToFuncPassBase;

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<DefineOpRewrite, CallOpRewrite, ReturnOpRewrite>(ctx);

    if (failed(applyPatternsGreedily(
            getOperation(), std::move(patterns),
            GreedyRewriteConfig().setRegionSimplificationLevel(
                GreedySimplifyRegionLevel::Normal)))) {
      llvm::errs() << "Failed to convert tessera dialect operations to func "
                      "dialect operations\n";
      return signalPassFailure();
    }

    // A define that could not be converted would leave an unknown op behind.
    WalkResult result = getOperation()->walk([](tessera::DefineOp defineOp) {
      defineOp.emitError("could not convert to func.func");
      return WalkResult::interrupt();
    });
    if (result.wasInterrupted())
      signalPassFailure();
  }
};
} // namespace
