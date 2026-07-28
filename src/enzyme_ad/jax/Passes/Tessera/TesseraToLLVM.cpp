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

    // Copy over attributes other than the function name and type, byRef args,
    // argSizes, pure, and other attributes used only for tessera conversion
    for (const auto &namedAttr : defineOp->getAttrs()) {
      if (namedAttr.getName() != SymbolTable::getSymbolAttrName() &&
          namedAttr.getName() != defineOp.getFunctionTypeAttrName() &&
          namedAttr.getName() != defineOp.getByRefTypesAttrName() &&
          namedAttr.getName() != defineOp.getPureAttrName() &&
          namedAttr.getName() != defineOp.getResultArgTypesAttrName() &&
          namedAttr.getName() != "tessera.original_name")
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

    // Check if callee's first argument has sret attribute. If so, allocate new
    // pointer to contain result of tessera.call and insert as first argument in
    // llvm.call.
    auto defineOp = dyn_cast_or_null<tessera::DefineOp>(callee);
    if (!defineOp)
      return failure();

    auto buildNewAttrs = [&](ArrayRef<NamedAttribute> baseAttrs,
                             int32_t numOperands,
                             std::optional<ArrayAttr> argAttrsOverride) {
      SmallVector<NamedAttribute> newAttrs;
      for (auto attr : baseAttrs) {
        if (attr.getName() != callOp.getArgAttrsAttrName() &&
            attr.getName() != "tessera.loaded_operands" &&
            attr.getName() != "operandSegmentSizes" &&
            attr.getName() != "op_bundle_sizes")
          newAttrs.push_back(attr);
      }
      if (argAttrsOverride)
        newAttrs.push_back(rewriter.getNamedAttr(callOp.getArgAttrsAttrName(),
                                                 *argAttrsOverride));
      else if (auto argAttrs = callOp.getArgAttrsAttr())
        newAttrs.push_back(
            rewriter.getNamedAttr(callOp.getArgAttrsAttrName(), argAttrs));
      newAttrs.push_back(rewriter.getNamedAttr(
          "operandSegmentSizes",
          rewriter.getDenseI32ArrayAttr({numOperands, 0})));
      newAttrs.push_back(rewriter.getNamedAttr(
          "op_bundle_sizes", rewriter.getDenseI32ArrayAttr({})));
      return newAttrs;
    };

    // For each of callOp's operands marked in "tessera.loaded_operands"
    // (i.e. it was loaded from a byref/byval pointer when converting from
    // LLVM to tessera), allocate fresh stack storage, store the operand's
    // value into it, and substitute that pointer in its place; all other
    // operands pass through unchanged. This is independent of sret --
    // byref-loadedness and sret-ness are separate properties of a callee's
    // arguments, so this reconstruction applies regardless of which branch
    // below is taken.
    SmallVector<int32_t> argsToReplace;
    if (auto loadedOperands = callOp->getAttrOfType<DenseI32ArrayAttr>(
            "tessera.loaded_operands"))
      argsToReplace = llvm::to_vector(loadedOperands.asArrayRef());

    Value one;
    SmallVector<Value> reconstructedOperands;
    for (auto [i, operand] : llvm::enumerate(callOp.getOperands())) {
      if (llvm::is_contained(argsToReplace, (int32_t)i)) {
        if (!one)
          one = LLVM::ConstantOp::create(rewriter, callOp.getLoc(),
                                         rewriter.getI32Type(),
                                         rewriter.getI32IntegerAttr(1));
        int64_t alignment = 0;
        if (auto alignAttr =
                defineOp.getArgAttr(i, LLVM::LLVMDialect::getAlignAttrName()))
          alignment = cast<IntegerAttr>(alignAttr).getInt();
        Value AI = LLVM::AllocaOp::create(
            rewriter, callOp.getLoc(),
            LLVM::LLVMPointerType::get(callOp->getContext()),
            operand.getType(), one, alignment);
        LLVM::StoreOp::create(rewriter, callOp.getLoc(), operand, AI);
        reconstructedOperands.push_back(AI);
      } else {
        reconstructedOperands.push_back(operand);
      }
    }

    if (defineOp.getNumArguments() > 0 && defineOp.getSretAttr()) {
      auto sretArgAttrs = defineOp.getArgAttrDict(0);
      if (callOp.getNumResults() == 0)
        return callOp.emitOpError(
            "tessera.call to sret function must have a result");
      auto sretType = callOp.getResult(0).getType();
      int64_t sret_alignment = 0;
      if (auto sretAlignAttr =
              sretArgAttrs.get(LLVM::LLVMDialect::getAlignAttrName()))
        sret_alignment = cast<IntegerAttr>(sretAlignAttr).getInt();
      if (!one)
        one = LLVM::ConstantOp::create(rewriter, callOp.getLoc(),
                                       rewriter.getI32Type(),
                                       rewriter.getI32IntegerAttr(1));

      // Allocate stack storage for the sret return value
      Value sretPtr = LLVM::AllocaOp::create(
          rewriter, callOp.getLoc(),
          LLVM::LLVMPointerType::get(callOp->getContext()), sretType, one,
          sret_alignment);

      // Build new operands with sretPtr as first arg, followed by the
      // reconstructed (byref-resolved) operands
      SmallVector<Value> newOperands;
      newOperands.push_back(sretPtr);
      newOperands.append(reconstructedOperands.begin(),
                         reconstructedOperands.end());

      // Reconstruct arg attributes with sret attr first
      SmallVector<Attribute> newArgAttrs;
      newArgAttrs.push_back(sretArgAttrs);
      if (auto argAttrs = callOp.getArgAttrsAttr()) {
        for (auto argAttr : argAttrs)
          newArgAttrs.push_back(argAttr);
      }

      auto newAttrs = buildNewAttrs(callOp->getAttrs(), newOperands.size(),
                                    rewriter.getArrayAttr(newArgAttrs));

      LLVM::CallOp::create(rewriter, callOp.getLoc(), TypeRange{}, newOperands,
                           newAttrs);

      // Load result from sret pointer and replace uses
      auto loadedResult =
          LLVM::LoadOp::create(rewriter, callOp.getLoc(), sretType, sretPtr);
      rewriter.replaceOp(callOp, loadedResult.getResult());
    } else if (defineOp.getNumResultArgs() == 0) {
      auto newAttrs = buildNewAttrs(callOp->getAttrs(),
                                    reconstructedOperands.size(), std::nullopt);
      rewriter.replaceOpWithNewOp<LLVM::CallOp>(
          callOp, callOp.getResultTypes(), reconstructedOperands, newAttrs);
    } else {
      // Allocate stack storage for each result argument, and splice those
      // pointers into the operand list in the right positions, so that the
      // LLVM::CallOp can be issued with the callee's real function type
      // and the result-only arguments passed by pointer.
      SmallVector<Value> newOperands;
      SmallVector<Attribute> newArgAttrs;
      SmallVector<Value> resultArgPtrs;
      SmallVector<Type> resultArgTypes;
      unsigned tesseraCallIdx = 0;
      for (unsigned i = 0, e = defineOp.getNumArguments(); i != e; ++i) {
        if (defineOp.isResultOnlyArg(i)) {
          int64_t alignment = 0;
          auto resultArgAttrs = defineOp.getArgAttrDict(i);
          if (resultArgAttrs) {
            if (auto alignAttr =
                  resultArgAttrs.get(LLVM::LLVMDialect::getAlignAttrName()))
            alignment = cast<IntegerAttr>(alignAttr).getInt();
          }
          if (!one)
            one = LLVM::ConstantOp::create(rewriter, callOp.getLoc(),
                                          rewriter.getI32Type(),
                                          rewriter.getI32IntegerAttr(1));

          Type resultArgType = defineOp.getResultArgType(i);
          if (!resultArgType)
            return callOp.emitOpError(
                "tessera.call to function with result-only argument must "
                "have a result type for that argument");
          // Allocate stack storage for the result/output argument
          Value resultArgPtr = LLVM::AllocaOp::create(
              rewriter, callOp.getLoc(),
              LLVM::LLVMPointerType::get(callOp->getContext()), resultArgType, 
              one, alignment);
          newOperands.push_back(resultArgPtr);
          resultArgPtrs.push_back(resultArgPtr);
          resultArgTypes.push_back(resultArgType);
          newArgAttrs.push_back(resultArgAttrs ? resultArgAttrs
                                : rewriter.getDictionaryAttr({}));
        } else {
          if (tesseraCallIdx >= reconstructedOperands.size())
            return callOp.emitOpError(
                "tessera.call has fewer operands than expected by callee");
          newOperands.push_back(reconstructedOperands[tesseraCallIdx]);
          auto callArgAttrs = callOp.getArgAttrsAttr();
          newArgAttrs.push_back(callArgAttrs ? callArgAttrs[tesseraCallIdx]
                                : rewriter.getDictionaryAttr({}));
          ++tesseraCallIdx;
        }
      }

      auto newAttrs = buildNewAttrs(callOp->getAttrs(), newOperands.size(),
                                    rewriter.getArrayAttr(newArgAttrs));

      auto fnType = defineOp.getFunctionType();
      auto returnType = fnType.getNumResults() == 0
                        ? TypeRange{} : TypeRange{fnType.getResult(0)};
      auto newCall = LLVM::CallOp::create(rewriter, callOp.getLoc(), returnType,
                             newOperands, newAttrs);

      // Load each result arg's value back from its alloca, then replace
      // callOp's results: one loaded value per result-only argument (the
      // leading results on the tessera.call), in argument order, followed
      // by the natural result (if any), matching the new LLVM call's
      // direct result.
      SmallVector<Value> replacementValues;
      for (auto [ptr, type] : llvm::zip(resultArgPtrs, resultArgTypes)) {
        auto loaded = LLVM::LoadOp::create(rewriter, callOp.getLoc(), type, ptr);
        replacementValues.push_back(loaded.getResult());
      }
      if (fnType.getNumResults() > 0)
        replacementValues.push_back(newCall.getResult());
      rewriter.replaceOp(callOp, replacementValues);
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

    if (failed(applyPatternsGreedily(
            getOperation(), std::move(patterns),
            GreedyRewriteConfig().setRegionSimplificationLevel(
                GreedySimplifyRegionLevel::Normal)))) {
      llvm::errs() << "Failed to convert tessera dialect operations to LLVM "
                      "dialect operations\n";
      return signalPassFailure();
    }
  }
};
} // namespace
