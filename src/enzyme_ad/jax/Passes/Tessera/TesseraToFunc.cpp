//===----------------------------------------------------------------------===//
//
// This file implements patterns to convert operations in the Tessera dialect to 
// operations in the Func dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "src/enzyme_ad/jax/Dialect/Tessera/Dialect.h"
#include "src/enzyme_ad/jax/Passes/Tessera/Passes.h"
#include "src/enzyme_ad/jax/Passes/Passes.h"

using namespace mlir;
using namespace mlir::enzyme::tessera;

namespace {
} // namespace


//===----------------------------------------------------------------------===//
// Rewrite Patterns
//===----------------------------------------------------------------------===//

namespace {

// Rewrite 'tessera.define' -> 'func.func'
class DefineOpRewrite final : public OpRewritePattern<tessera::DefineOp> {
public:
  using OpRewritePattern<tessera::DefineOp>::OpRewritePattern;

  LogicalResult
  matchAndRewrite(tessera::DefineOp defineOp,
                  PatternRewriter &rewriter) const override {
    FunctionType fnType = defineOp.getFunctionType();


    // Create the `func.func` op
    auto funcOp = rewriter.create<tessera::DefineOp>(
      defineOp.getLoc(), defineOp.getName(), fnType);
    

    // Copy over all attributes other than the function name and type.
    for (const auto &namedAttr : defineOp->getAttrs()) {
      if (namedAttr.getName() != defineOp.getFunctionTypeAttrName() &&
          namedAttr.getName() != SymbolTable::getSymbolAttrName())
        funcOp->setAttr(namedAttr.getName(), namedAttr.getValue());
    }

    // Add `extern` to specifiers if `tessera.define` is declaration only.
    if (defineOp.isDeclaration()) {
      ArrayAttr specifiers = rewriter.getStrArrayAttr({"extern"});
      funcOp.setSpecifiersAttr(specifiers);
    }

    // Add `static` to specifiers if `tessera.define` is private but not a
    // declaration.
    if (defineOp.isPrivate() && !defineOp.isDeclaration()) {
      ArrayAttr specifiers = rewriter.getStrArrayAttr({"static"});
      funcOp.setSpecifiersAttr(specifiers);
    }

    if (!defineOp.isDeclaration()) {
      defineOp.getBody().cloneInto(&funcOp.getBody(),
                                  funcOp.getBody().end());
    }

    rewriter.eraseOp(defineOp);

    return success();
  }
};

// Rewrite 'tessera.call' -> 'func.call'
class CallOpRewrite final : public OpRewritePattern<tessera::CallOp> {
public:
  using OpRewritePattern<tessera::CallOp>::OpRewritePattern;

  LogicalResult
  matchAndRewrite(tessera::CallOp callOp,
                  PatternRewriter &rewriter) const override {

    rewriter.replaceOpWithNewOp<func::CallOp>(callOp, callOp.getResultTypes(),
                                               callOp.getOperands(),
                                               callOp->getAttrs());

    return success();
  }
};

// Rewrite 'tessera.return' -> 'func.return'
class ReturnOpRewrite final : public OpRewritePattern<tessera::ReturnOp> {
public:
  using OpRewritePattern<tessera::ReturnOp>::OpRewritePattern;

  LogicalResult
  matchAndRewrite(tessera::ReturnOp returnOp,
                  PatternRewriter &rewriter) const override {

    rewriter.replaceOpWithNewOp<func::ReturnOp>(returnOp,
        returnOp.getOperands());
    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Pass to convert Tessera operations into Func operations
//===----------------------------------------------------------------------===//

struct TesseraToFuncPass
  : public PassWrapper<TesseraToFuncPass, OperationPass<ModuleOp>> {

  void runOnOperation() override {
    MLIRContext &ctx = patterns.getContext();
    RewritePatternSet patterns(&ctx);

    patterns.add<DefineOpRewrite, CallOpRewrite, ReturnOpRewrite>(&ctx);

    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns))))
      signalPassFailure();
  }
};

