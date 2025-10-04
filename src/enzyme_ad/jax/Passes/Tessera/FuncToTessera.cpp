//===----------------------------------------------------------------------===//
//
// This file implements patterns to convert the Func dialect to the Tessera
// dialect and from the Tessera dialect to the Func dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "src/enzyme_ad/jax/Dialect/Tessera/Dialect.h"
#include "src/enzyme_ad/jax/Passes/Passes.h"

using namespace mlir;
using namespace mlir::enzyme::tessera;

namespace {
} // namespace


//===----------------------------------------------------------------------===//
// Rewrite Patterns
//===----------------------------------------------------------------------===//

namespace {

// Rewrite 'func.func' -> 'tessera.define'
class FuncOpRewrite final : public OpRewritePattern<func::FuncOp> {
public:
  using OpRewritePattern<func::FuncOp>::OpRewritePattern;

  LogicalResult
  matchAndRewrite(func::FuncOp funcOp,
                  PatternRewriter &rewriter) const override {
    FunctionType fnType = funcOp.getFunctionType();

    if (fnType.getNumResults() > 1)
      return rewriter.notifyMatchFailure(
          funcOp, "only functions with zero or one result can be rewritten");


    // Create the `tessera.define` op
    auto tesseraDefineOp = rewriter.create<tessera::DefineOp>(
      funcOp.getLoc(), funcOp.getName(), fnType);
    

    // Copy over all attributes other than the function name and type.
    for (const auto &namedAttr : funcOp->getAttrs()) {
      if (namedAttr.getName() != funcOp.getFunctionTypeAttrName() &&
          namedAttr.getName() != SymbolTable::getSymbolAttrName())
        tesseraDefineOp->setAttr(namedAttr.getName(), namedAttr.getValue());
    }

    // Add `extern` to specifiers if `func.func` is declaration only.
    if (funcOp.isDeclaration()) {
      ArrayAttr specifiers = rewriter.getStrArrayAttr({"extern"});
      tesseraDefineOp.setSpecifiersAttr(specifiers);
    }

    // Add `static` to specifiers if `func.func` is private but not a
    // declaration.
    if (funcOp.isPrivate() && !funcOp.isDeclaration()) {
      ArrayAttr specifiers = rewriter.getStrArrayAttr({"static"});
      tesseraDefineOp.setSpecifiersAttr(specifiers);
    }

    if (!funcOp.isDeclaration()) {
      rewriter.inlineRegionBefore(funcOp.getBody(), tesseraDefineOp.getBody(),
                                  tesseraDefineOp.end());
    }


    rewriter.eraseOp(funcOp);

    return success();
  }
};

// Rewrite 'func.call' -> 'tessera.call'
class CallOpRewrite final : public OpRewritePattern<func::CallOp> {
public:
  using OpRewritePattern<func::CallOp>::OpRewritePattern;

  LogicalResult
  matchAndRewrite(func::CallOp callOp,
                  PatternRewriter &rewriter) const override {

    rewriter.replaceOpWithNewOp<tessera::CallOp>(callOp, callOp.getResultTypes(),
                                               callOp.getOperands(),
                                               callOp->getAttrs());

    return success();
  }
};

// Rewrite 'func.return' -> 'tessera.return'
class ReturnOpRewrite final : public OpRewritePattern<func::ReturnOp> {
public:
  using OpRewritePattern<func::ReturnOp>::OpRewritePattern;

  LogicalResult
  matchAndRewrite(func::ReturnOp returnOp,
                  PatternRewriter &rewriter) const override {

    rewriter.replaceOpWithNewOp<tessera::ReturnOp>(returnOp,
        returnOp.getOperands());
    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Pass to convert Func operations into Tessera operations
//===----------------------------------------------------------------------===//

struct FuncToTesseraPass
  : public PassWrapper<FuncToTesseraPass, OperationPass<ModuleOp>> {

  void runOnOperation() override {
    MLIRContext &ctx = patterns.getContext();
    RewritePatternSet patterns(&ctx);

    patterns.add<CallOpRewrite, FuncOpRewrite, ReturnOpRewrite>(&ctx);

    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns))))
      signalPassFailure();
  }
};

