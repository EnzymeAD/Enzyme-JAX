//===----------------------------------------------------------------------===//
//
// This file implements patterns to convert operations in the Func dialect to
// operations in the Tessera dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinDialect.h"
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
#define GEN_PASS_DEF_FUNCTOTESSERAPASS
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

// Rewrite 'func.func' -> 'tessera.define'
class FuncOpRewrite final : public OpRewritePattern<func::FuncOp> {
public:
  using OpRewritePattern<func::FuncOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(func::FuncOp funcOp,
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
                                 tesseraDefineOp.getBody().end(), mapper);

      // Now walk through the cloned operations and convert func.return to
      // tessera.return
      tesseraDefineOp.walk([&](func::ReturnOp returnOp) {
        rewriter.setInsertionPoint(returnOp);
        rewriter.replaceOpWithNewOp<tessera::ReturnOp>(returnOp,
                                                       returnOp.getOperands());
      });

      // Convert func.call to tessera.call
      tesseraDefineOp.walk([&](func::CallOp callOp) {
        rewriter.setInsertionPoint(callOp);
        rewriter.replaceOpWithNewOp<tessera::CallOp>(
            callOp, callOp.getResultTypes(), callOp.getOperands(),
            callOp->getAttrs());
      });
    }

    rewriter.eraseOp(funcOp);

    return success();
  }
};

// Rewrite 'func.call' -> 'tessera.call'
class CallOpRewrite final : public OpRewritePattern<func::CallOp> {
public:
  using OpRewritePattern<func::CallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(func::CallOp callOp,
                                PatternRewriter &rewriter) const override {

    rewriter.replaceOpWithNewOp<tessera::CallOp>(
        callOp, callOp.getResultTypes(), callOp.getOperands(),
        callOp->getAttrs());

    return success();
  }
};

// Rewrite 'func.return' -> 'tessera.return'
class ReturnOpRewrite final : public OpRewritePattern<func::ReturnOp> {
public:
  using OpRewritePattern<func::ReturnOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(func::ReturnOp returnOp,
                                PatternRewriter &rewriter) const override {

    rewriter.replaceOpWithNewOp<tessera::ReturnOp>(returnOp,
                                                   returnOp.getOperands());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass to convert Func operations into Tessera operations
//===----------------------------------------------------------------------===//

struct FuncToTesseraPass
    : public enzyme::tessera::impl::FuncToTesseraPassBase<FuncToTesseraPass> {
  using FuncToTesseraPassBase::FuncToTesseraPassBase;

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();

    ConversionTarget target(*ctx);
    target.addLegalDialect<tessera::TesseraDialect>();
    target.addLegalDialect<BuiltinDialect>();
    target.addIllegalDialect<func::FuncDialect>();

    // Define which func operations are legal/illegal
    target.addDynamicallyLegalOp<func::FuncOp>([](func::FuncOp op) {
      // Return true = legal (don't convert)
      // Return false = illegal (must convert)
      return !op->hasAttr("tessera.convert");
    });

    target.addDynamicallyLegalOp<func::CallOp>([](func::CallOp op) {
      auto module = op->getParentOfType<ModuleOp>();
      auto callee = SymbolTable::lookupSymbolIn(module, op.getCalleeAttr());
      return !isa_and_nonnull<tessera::DefineOp>(callee);
    });

    target.addDynamicallyLegalOp<func::ReturnOp>([](func::ReturnOp op) {
      return !isa<tessera::DefineOp>(op->getParentOp());
    });

    RewritePatternSet patterns(ctx);

    patterns.add<FuncOpRewrite, CallOpRewrite, ReturnOpRewrite>(ctx);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      llvm::errs() << "Failed to convert func dialect operations to tessera "
                      "dialect operations\n";
      signalPassFailure();
    }
  }
};
} // namespace
