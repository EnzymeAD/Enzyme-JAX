//===----------------------------------------------------------------------===//
//
// This file implements patterns to convert operations in the Tessera dialect to
// operations in the LLVM dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
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
#define GEN_PASS_DEF_TESSERATOLLVMPASS
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

// Rewrite 'tessera.define' -> 'llvm.func'
class DefineOpRewrite final : public OpRewritePattern<tessera::DefineOp> {
public:
  DefineOpRewrite(LLVMTypeConverter &typeConverter, MLIRContext *ctx)
      : OpRewritePattern(ctx), typeConverter(typeConverter) {}

  LogicalResult matchAndRewrite(tessera::DefineOp defineOp,
                                PatternRewriter &rewriter) const override {
    auto funcNameAttr =
        defineOp->getAttrOfType<StringAttr>("tessera.original_name");
    if (!funcNameAttr)
      return failure();
    auto funcName = funcNameAttr.getValue();
    auto module = defineOp->getParentOfType<ModuleOp>();
    auto *ctx = defineOp->getContext();
    auto fnType = defineOp.getFunctionType();

    // Convert argument types
    SmallVector<Type> argTypes;
    for (auto type : fnType.getInputs())
      argTypes.push_back(typeConverter.convertType(type));

    // Handle return type - void if no results
    Type returnType = fnType.getNumResults() == 0
                          ? LLVM::LLVMVoidType::get(ctx)
                          : typeConverter.convertType(fnType.getResult(0));
    auto llvmFuncType = LLVM::LLVMFunctionType::get(returnType, argTypes);
    if (!llvmFuncType)
      return failure();

    // Replace tessera name with original function name
    if (failed(SymbolTable::replaceAllSymbolUses(
            defineOp.getSymNameAttr(), StringAttr::get(ctx, funcName), module)))
      return failure();

    // Create the `llvm.func` op
    auto funcOp = LLVM::LLVMFuncOp::create(rewriter, defineOp.getLoc(),
                                           funcName, llvmFuncType);

    // Copy over all attributes other than the function name and type and
    // attributes used only for tessera conversion
    for (const auto &namedAttr : defineOp->getAttrs()) {
      if (namedAttr.getName() != defineOp.getFunctionTypeAttrName() &&
          namedAttr.getName() != SymbolTable::getSymbolAttrName() &&
          namedAttr.getName() != "tessera.original_name" &&
          namedAttr.getName() != "tessera.sret_attrs")
        funcOp->setAttr(namedAttr.getName(), namedAttr.getValue());
    }

    // Clone body of function
    if (!defineOp.isExternal()) {
      rewriter.inlineRegionBefore(defineOp.getBody(), funcOp.getBody(),
                                  funcOp.end());
    }

    rewriter.eraseOp(defineOp);

    return success();
  }

private:
  LLVMTypeConverter &typeConverter;
};

// Rewrite 'tessera.call' -> 'llvm.call'
class CallOpRewrite final : public OpRewritePattern<tessera::CallOp> {
public:
  using OpRewritePattern<tessera::CallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tessera::CallOp callOp,
                                PatternRewriter &rewriter) const override {

    auto calleeAttr = callOp.getCalleeAttr();
    if (!calleeAttr)
      return failure();

    auto callee = SymbolTable::lookupSymbolIn(
        callOp->getParentOfType<ModuleOp>(), calleeAttr);

    // Check if callee has sret attribute. If so, allocate new pointer to
    // contain result of tessera.call and insert as first argument in llvm.call.
    auto defineOp = dyn_cast_or_null<tessera::DefineOp>(callee);
    if (!defineOp)
      return failure();

    auto sretAttrs =
        defineOp->getAttrOfType<DictionaryAttr>("tessera.sret_attrs");
    if (sretAttrs) {
      if (callOp.getNumResults() == 0)
        return callOp.emitOpError(
            "tessera.call to sret function must have a result");
      auto sretType = callOp.getResult(0).getType();
      int64_t alignment = 0;
      if (auto alignAttr = sretAttrs.get(LLVM::LLVMDialect::getAlignAttrName()))
        alignment = cast<IntegerAttr>(alignAttr).getInt();
      Value one = rewriter.create<LLVM::ConstantOp>(
          callOp.getLoc(), rewriter.getI32Type(),
          rewriter.getI32IntegerAttr(1));

      // Allocate stack storage for the sret return value
      Value sretPtr = rewriter.create<LLVM::AllocaOp>(
          callOp.getLoc(), LLVM::LLVMPointerType::get(callOp->getContext()),
          sretType, one, alignment);

      // Build new operands with sretPtr as first arg
      SmallVector<Value> newOperands;
      newOperands.push_back(sretPtr);
      for (auto operand : callOp.getOperands())
        newOperands.push_back(operand);

      // Reconstruct arg attributes with sret attr first
      SmallVector<Attribute> newArgAttrs;
      newArgAttrs.push_back(sretAttrs);
      if (auto argAttrs = callOp.getArgAttrsAttr()) {
        for (auto argAttr : argAttrs)
          newArgAttrs.push_back(argAttr);
      }

      // Filter out arg_attrs from attributes
      SmallVector<NamedAttribute> newAttrs;
      for (auto attr : callOp->getAttrs()) {
        if (attr.getName() != callOp.getArgAttrsAttrName())
          newAttrs.push_back(attr);
      }

      auto newCall = rewriter.create<LLVM::CallOp>(callOp.getLoc(), TypeRange{},
                                                   newOperands, newAttrs);
      newCall->setAttr(newCall.getArgAttrsAttrName(),
                       rewriter.getArrayAttr(newArgAttrs));

      // Load result from sret pointer and replace uses
      auto loadedResult =
          rewriter.create<LLVM::LoadOp>(callOp.getLoc(), sretType, sretPtr);
      rewriter.replaceOp(callOp, loadedResult.getResult());
    } else {
      rewriter.replaceOpWithNewOp<LLVM::CallOp>(callOp, callOp.getResultTypes(),
                                                callOp.getOperands(),
                                                callOp->getAttrs());
    }

    return success();
  }
};

// Rewrite 'tessera.return' -> 'llvm.return'
class ReturnOpRewrite final : public OpRewritePattern<tessera::ReturnOp> {
public:
  using OpRewritePattern<tessera::ReturnOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tessera::ReturnOp returnOp,
                                PatternRewriter &rewriter) const override {

    rewriter.replaceOpWithNewOp<LLVM::ReturnOp>(returnOp,
                                                returnOp.getOperands());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass to convert Tessera operations into Func operations
//===----------------------------------------------------------------------===//

struct TesseraToLLVMPass
    : public enzyme::tessera::impl::TesseraToLLVMPassBase<TesseraToLLVMPass> {
  using TesseraToLLVMPassBase::TesseraToLLVMPassBase;

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    LLVMTypeConverter typeConverter(ctx);
    RewritePatternSet patterns(ctx);

    patterns.add<DefineOpRewrite>(typeConverter, ctx);
    patterns.add<CallOpRewrite, ReturnOpRewrite>(ctx);

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
      llvm::errs() << "Failed to convert tessera dialect operations to LLVM "
                      "dialect operations\n";
    }
  }
};
} // namespace
