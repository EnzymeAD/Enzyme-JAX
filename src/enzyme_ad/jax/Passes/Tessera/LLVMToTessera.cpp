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
    auto module = funcOp->getParentOfType<ModuleOp>();
    auto *ctx = funcOp->getContext();
    auto funcName = funcOp.getName();
    auto llvmFuncType = funcOp.getFunctionType();
    auto params = llvmFuncType.getParams();
    auto retType = llvmFuncType.getReturnType();

    // Check if first argument has sret attribute
    bool hasSret = false;
    auto argAttrs = funcOp.getArgAttrsAttr();
    if (!params.empty() && argAttrs) {
      auto firstArgAttrs = cast<DictionaryAttr>(argAttrs[0]);
      if (auto sretAttr =
              firstArgAttrs.get(LLVM::LLVMDialect::getStructRetAttrName()))
        hasSret = true;
    }

    auto fnType = FunctionType::get(
        ctx, params,
        isa<LLVM::LLVMVoidType>(retType) ? TypeRange{} : TypeRange{retType});

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

    // Add attribute if function uses struct return and store the first arg's
    // attributes for exact reconstruction later
    if (hasSret)
      tesseraDefineOp->setAttr("tessera.sret_attrs", argAttrs[0]);

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

    auto calleeAttr = callOp.getCalleeAttr();
    if (!calleeAttr)
      return failure();

    auto callee = SymbolTable::lookupSymbolIn(
        callOp->getParentOfType<ModuleOp>(), calleeAttr);
    // Only rewrite if callee is a tessera.define op
    if (!isa_and_nonnull<tessera::DefineOp>(callee))
      return failure();

    // Check if first operand has sret attribute. If so, remove it from
    // the operand list and use its pointed-to type as the SSA return type,
    // since tessera.call returns values directly rather than writing through
    // a pointer.
    Value sretPtr;
    Type sretType;
    auto operands = callOp.getOperands();
    auto argAttrs = callOp.getArgAttrsAttr();
    SmallVector<Value> newOperands;
    SmallVector<Attribute> newArgAttrs;
    SmallVector<NamedAttribute> newAttrs;

    if (!operands.empty() && argAttrs) {
      auto firstArgAttrs = cast<DictionaryAttr>(argAttrs[0]);
      if (auto sretAttr =
              firstArgAttrs.get(LLVM::LLVMDialect::getStructRetAttrName())) {
        sretPtr = callOp.getOperand(0);
        sretType = cast<TypeAttr>(sretAttr).getValue();
        // Build operands and arg attributes without first element
        for (int i = 1; i < operands.size(); i++)
          newOperands.push_back(callOp.getOperand(i));
        for (int j = 1; j < argAttrs.size(); j++)
          newArgAttrs.push_back(argAttrs[j]);
        // Filter out arg_attrs from attributes
        for (auto attr : callOp->getAttrs()) {
          if (attr.getName() != callOp.getArgAttrsAttrName())
            newAttrs.push_back(attr);
        }
      }
    }

    // Create tessera.call op with SSA return type
    if (sretPtr) {
      auto newCall = rewriter.create<tessera::CallOp>(
          callOp.getLoc(), TypeRange{sretType}, newOperands, newAttrs);
      rewriter.create<LLVM::StoreOp>(callOp.getLoc(), newCall.getResult(0),
                                     sretPtr);
      newCall->setAttr(newCall.getArgAttrsAttrName(),
                       rewriter.getArrayAttr(newArgAttrs));
      rewriter.eraseOp(callOp);
    } else {
      rewriter.replaceOpWithNewOp<tessera::CallOp>(
          callOp, callOp.getResultTypes(), operands, callOp->getAttrs());
    }

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

    GreedyRewriteConfig config;
    config.setUseTopDownTraversal(true);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns),
                                     config))) {
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
