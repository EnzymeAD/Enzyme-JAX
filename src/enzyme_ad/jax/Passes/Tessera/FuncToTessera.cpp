//===----------------------------------------------------------------------===//
//
// This file implements patterns to convert operations in the Func dialect to 
// operations in the Tessera dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "src/enzyme_ad/jax/Dialect/Tessera/Ops.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/IR/BuiltinOps.h"
#include "src/enzyme_ad/jax/Dialect/Tessera/Dialect.h"
#include "src/enzyme_ad/jax/Passes/Tessera/Passes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::enzyme;

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


    // Create the `tessera.define` op
    auto tesseraDefineOp = rewriter.create<tessera::DefineOp>(
      funcOp.getLoc(), funcOp.getName(), fnType);
    
    // Copy over all attributes other than the function name and type.
    for (const auto &namedAttr : funcOp->getAttrs()) {
      if (namedAttr.getName() != funcOp.getFunctionTypeAttrName() &&
          namedAttr.getName() != SymbolTable::getSymbolAttrName())
        tesseraDefineOp->setAttr(namedAttr.getName(), namedAttr.getValue());
    }

    // Clone body of function
    if (!funcOp.isExternal()) {
      IRMapping mapper;
      funcOp.getBody().cloneInto(&tesseraDefineOp.getBody(),
                            tesseraDefineOp.getBody().end(),
                            mapper);
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

    auto calleeAttr = callOp.getCalleeAttr();
    Operation *moduleOp = callOp->getParentOfType<ModuleOp>();
    Operation *calleeOp = SymbolTable::lookupSymbolIn(moduleOp, calleeAttr);
    
    // Only convert if the callee is a Tessera DefineOp
    if (isa<tessera::DefineOp>(calleeOp))
      return rewriter.notifyMatchFailure(callOp, "Callee is not a Tessera DefineOp");
    
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
    Operation *parent = returnOp->getParentOp();
    
    // Only convert if the function is a Tessera DefineOp
    if (!isa<tessera::DefineOp>(parent))
      return rewriter.notifyMatchFailure(returnOp, "Parent is not a Tessera DefineOp");
    
    rewriter.replaceOpWithNewOp<tessera::ReturnOp>(returnOp,
        returnOp.getOperands());
    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Pass to convert Func operations into Tessera operations
//===----------------------------------------------------------------------===//

namespace mlir::enzyme::tessera {

struct FuncToTesseraPass
  : public PassWrapper<FuncToTesseraPass, OperationPass<ModuleOp>> {

  void runOnOperation() override {
  MLIRContext *ctx = &getContext();
  RewritePatternSet patterns(ctx);

  patterns.add<FuncOpRewrite, CallOpRewrite, ReturnOpRewrite>(ctx);

  if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                          std::move(patterns))))
    signalPassFailure();
}
};

std::unique_ptr<mlir::Pass> createFuncToTesseraPass() {
  return std::make_unique<FuncToTesseraPass>();
}
}
