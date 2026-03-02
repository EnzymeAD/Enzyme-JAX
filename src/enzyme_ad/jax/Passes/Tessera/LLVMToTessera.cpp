//===----------------------------------------------------------------------===//
//
// This file implements patterns to convert operations in the LLVM dialect to
// operations in the Tessera dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
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
#define GEN_PASS_DEF_LLVMTOTESSERAPASS
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

// Rewrite 'llvm.func' -> 'tessera.define'
class FuncOpRewrite final : public OpRewritePattern<LLVM::LLVMFuncOp> {
public:
  using OpRewritePattern<LLVM::LLVMFuncOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(LLVM::LLVMFuncOp funcOp,
                                PatternRewriter &rewriter) const override {

    // Only rewrite if op has tessera.convert attribute
    auto convertAttr =
        funcOp->getAttrOfType<enzyme::tessera::ConvertAttr>("tessera.convert");
    if (!convertAttr)
      return failure();
    auto tesseraName = convertAttr.getValue();
    auto funcName = funcOp.getName();
    auto fnType = funcOp.getFunctionType();
    auto module = funcOp->getParentOfType<ModuleOp>();
    auto *ctx = funcOp->getContext();

    // Replace current function name with tessera name defined in
    // tessera.convert attribute
    if (failed(SymbolTable::replaceAllSymbolUses(
            funcOp.getSymNameAttr(), StringAttr::get(ctx, tesseraName),
            module)))
      return failure();
    auto tesseraDefineOp = tessera::DefineOp::create(rewriter, funcOp.getLoc(),
                                                     tesseraName, fnType);

    // Copy over all attributes other than the function name and type
    // and tessera.convert attribute.
    for (const auto &namedAttr : funcOp->getAttrs()) {
      if (namedAttr.getName() != funcOp.getFunctionTypeAttrName() &&
          namedAttr.getName() != SymbolTable::getSymbolAttrName() &&
          namedAttr.getName() != "tessera.convert")
        tesseraDefineOp->setAttr(namedAttr.getName(), namedAttr.getValue());
    }
    // Store the original function name so we can convert back to it later
    tesseraDefineOp->setAttr("tessera.original_name",
                             rewriter.getStringAttr(funcName));

    // Clone body of function
    if (!funcOp.isExternal()) {
      rewriter.inlineRegionBefore(funcOp.getBody(), tesseraDefineOp.getBody(),
                                  tesseraDefineOp.end());
    }

    rewriter.eraseOp(funcOp);

    return success();
  }
};

// Rewrite 'llvm.call' -> 'tessera.call'
class CallOpRewrite final : public OpRewritePattern<LLVM::CallOp> {
public:
  using OpRewritePattern<LLVM::CallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(LLVM::CallOp callOp,
                                PatternRewriter &rewriter) const override {

    auto callee = SymbolTable::lookupSymbolIn(
        callOp->getParentOfType<ModuleOp>(), callOp.getCalleeAttr());
    // Only rewrite if callee is a tessera.define op
    if (!isa_and_nonnull<tessera::DefineOp>(callee))
      return failure();

    rewriter.replaceOpWithNewOp<tessera::CallOp>(
        callOp, callOp.getResultTypes(), callOp.getOperands(),
        callOp->getAttrs());

    return success();
  }
};

// Rewrite 'llvm.return' -> 'tessera.return'
class ReturnOpRewrite final : public OpRewritePattern<LLVM::ReturnOp> {
public:
  using OpRewritePattern<LLVM::ReturnOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(LLVM::ReturnOp returnOp,
                                PatternRewriter &rewriter) const override {

    // Only rewrite if parent op is a tessera.define op
    if (!isa<tessera::DefineOp>(returnOp->getParentOp()))
      return failure();

    rewriter.replaceOpWithNewOp<tessera::ReturnOp>(returnOp,
                                                   returnOp.getOperands());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass to convert Func operations into Tessera operations
//===----------------------------------------------------------------------===//

struct LLVMToTesseraPass
    : public enzyme::tessera::impl::LLVMToTesseraPassBase<LLVMToTesseraPass> {
  using LLVMToTesseraPassBase::LLVMToTesseraPassBase;

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);

    patterns.add<FuncOpRewrite, CallOpRewrite, ReturnOpRewrite>(ctx);

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      llvm::errs() << "Failed to convert LLVM dialect operations to tessera "
                      "dialect operations\n";
      signalPassFailure();
    }

    // Clean up llvm.global.annotations after conversion if it still exists
    auto module = cast<ModuleOp>(getOperation());
    if (auto annotations = module.lookupSymbol("llvm.global.annotations")) {
      annotations->erase();
    }
  }
};
} // namespace
