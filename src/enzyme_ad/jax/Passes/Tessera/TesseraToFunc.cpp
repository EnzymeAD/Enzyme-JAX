//===----------------------------------------------------------------------===//
//
// This file implements patterns to convert operations in the Tessera dialect to
// operations in the Func dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
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
    FunctionType fnType = defineOp.getFunctionType();

    // Create the `func.func` op
    auto funcOp = rewriter.create<func::FuncOp>(defineOp.getLoc(),
                                                defineOp.getName(), fnType);

    // Copy over all attributes other than the function name and type.
    for (const auto &namedAttr : defineOp->getAttrs()) {
      if (namedAttr.getName() != defineOp.getFunctionTypeAttrName() &&
          namedAttr.getName() != SymbolTable::getSymbolAttrName())
        funcOp->setAttr(namedAttr.getName(), namedAttr.getValue());
    }

    // Clone body of function
    if (!defineOp.isExternal()) {
      IRMapping mapper;
      defineOp.getBody().cloneInto(&funcOp.getBody(), funcOp.getBody().end(),
                                   mapper);
    }

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

    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      signalPassFailure();
      llvm::errs() << "Failed to convert tessera dialect operations to func "
                      "dialect operations\n";
    }
  }
};
} // namespace
