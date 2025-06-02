#include "Enzyme/MLIR/Dialect/Ops.h"
#include "mhlo/IR/hlo_ops.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "src/enzyme_ad/jax/Dialect/Dialect.h"
#include "src/enzyme_ad/jax/Dialect/Ops.h"
#include "src/enzyme_ad/jax/Passes/Passes.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "llvm/ADT/DynamicAPInt.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/MathExtras.h"
#include <algorithm>
#include <cstdint>

#define DEBUG_TYPE "lower-enzyme-probprog"

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_LOWERENZYMEPROBPROGPASS
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

using namespace mlir;
using namespace mlir::enzyme;

template <typename T> Attribute makeAttr(mlir::Type elemType, T val) {
  if (auto TT = dyn_cast<RankedTensorType>(elemType))
    return SplatElementsAttr::get(
        TT, ArrayRef(makeAttr<T>(TT.getElementType(), val)));
  if (isa<FloatType>(elemType))
    return FloatAttr::get(elemType, val);
  else
    return IntegerAttr::get(elemType, val);
}

struct FuncOpConversion : public OpConversionPattern<func::FuncOp> {
  using OpConversionPattern::OpConversionPattern;

  std::string backend;
  FuncOpConversion(std::string backend, TypeConverter &typeConverter,
                   MLIRContext *context, PatternBenefit benefit = 1)
      : OpConversionPattern(typeConverter, context, benefit), backend(backend) {
  }

  LogicalResult
  matchAndRewrite(func::FuncOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto ctx = op->getContext();

    // For now: only convert functions that return a single enzyme.Trace
    if (op.getFunctionType().getNumResults() != 1 ||
        !isa<enzyme::TraceType>(op.getFunctionType().getResult(0)))
      return failure();

    auto newResultType = RankedTensorType::get(
        {1}, IntegerType::get(ctx, 64, IntegerType::Unsigned));

    SmallVector<Type> newResultTypes = {newResultType};
    auto newFuncType = FunctionType::get(ctx, op.getFunctionType().getInputs(),
                                         newResultTypes);

    rewriter.modifyOpInPlace(op, [&] { op.setType(newFuncType); });

    return success();
  }
};

struct CallOpConversion : public OpConversionPattern<func::CallOp> {
  using OpConversionPattern::OpConversionPattern;

  std::string backend;
  CallOpConversion(std::string backend, TypeConverter &typeConverter,
                   MLIRContext *context, PatternBenefit benefit = 1)
      : OpConversionPattern(typeConverter, context, benefit), backend(backend) {
  }

  LogicalResult
  matchAndRewrite(func::CallOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto ctx = op->getContext();

    // For now: only convert calls that return a single enzyme.Trace
    if (op.getNumResults() != 1 ||
        !isa<enzyme::TraceType>(op.getResult(0).getType()))
      return failure();

    auto newResultType = RankedTensorType::get(
        {1}, IntegerType::get(ctx, 64, IntegerType::Unsigned));

    rewriter.replaceOpWithNewOp<func::CallOp>(op, newResultType, op.getCallee(),
                                              adaptor.getOperands());
    return success();
  }
};

struct ReturnOpConversion : public OpConversionPattern<func::ReturnOp> {
  using OpConversionPattern::OpConversionPattern;

  std::string backend;
  ReturnOpConversion(std::string backend, TypeConverter &typeConverter,
                     MLIRContext *context, PatternBenefit benefit = 1)
      : OpConversionPattern(typeConverter, context, benefit), backend(backend) {
  }

  LogicalResult
  matchAndRewrite(func::ReturnOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // For now: only convert returns of a single enzyme.Trace
    if (op.getNumOperands() != 1 ||
        !isa<enzyme::TraceType>(op.getOperand(0).getType()))
      return failure();

    rewriter.replaceOpWithNewOp<func::ReturnOp>(op, adaptor.getOperands());
    return success();
  }
};

struct InitTraceOpConversion : public OpConversionPattern<enzyme::initTraceOp> {
  using OpConversionPattern::OpConversionPattern;

  std::string backend;
  InitTraceOpConversion(std::string backend, TypeConverter &typeConverter,
                        MLIRContext *context, PatternBenefit benefit = 1)
      : OpConversionPattern(typeConverter, context, benefit), backend(backend) {
  }

  LogicalResult
  matchAndRewrite(enzyme::initTraceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto ctx = op->getContext();

    if (backend == "cpu") {
      auto moduleOp = op->getParentOfType<ModuleOp>();
      static int64_t fnNum = 0;

      auto llvmPtrType = LLVM::LLVMPointerType::get(ctx);
      auto loweredTraceType = RankedTensorType::get(
          {1}, IntegerType::get(ctx, 64, IntegerType::Unsigned));

      std::string initTraceFn = "enzyme_probprog_init_trace";

      // Generate the LLVM function body
      std::string fnName = initTraceFn + "_wrapper_" + std::to_string(fnNum);
      fnNum++;
      {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(moduleOp.getBody());

        auto funcType = LLVM::LLVMFunctionType::get(llvmPtrType, {}, false);

        auto func =
            rewriter.create<LLVM::LLVMFuncOp>(op.getLoc(), fnName, funcType);
        rewriter.setInsertionPointToStart(func.addEntryBlock(rewriter));

        auto callResult = rewriter.create<LLVM::CallOp>(
            op.getLoc(), TypeRange{llvmPtrType},
            SymbolRefAttr::get(ctx, initTraceFn), ValueRange{});

        rewriter.create<LLVM::ReturnOp>(op.getLoc(), callResult.getResults());
      }

      // Insert function declaration if not already present
      if (!moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(initTraceFn)) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(moduleOp.getBody());

        auto funcType = LLVM::LLVMFunctionType::get(llvmPtrType, {}, false);

        rewriter.create<LLVM::LLVMFuncOp>(op.getLoc(), initTraceFn, funcType,
                                          LLVM::Linkage::External);
      }

      // Call the LLVM function with enzymexla.jit_call
      auto jitCall = rewriter.create<enzymexla::JITCallOp>(
          op.getLoc(), TypeRange{loweredTraceType},
          mlir::FlatSymbolRefAttr::get(ctx, fnName), ValueRange{},
          rewriter.getStringAttr(""),
          /*operand_layouts=*/rewriter.getArrayAttr({}),
          /*result_layouts=*/rewriter.getArrayAttr({}),
          /*output_operand_aliases=*/rewriter.getArrayAttr({}),
          /*xla_side_effect_free=*/nullptr);

      // Replace the initTraceOp with the result of the JIT call
      rewriter.replaceOp(op, jitCall.getResults());

      return success();
    } else {
      return rewriter.notifyMatchFailure(op, "Unknown backend " + backend);
    }
  }
};

struct addSampleToTraceOpConversion
    : public OpConversionPattern<enzyme::addSampleToTraceOp> {
  using OpConversionPattern::OpConversionPattern;

  std::string backend;
  addSampleToTraceOpConversion(std::string backend,
                               TypeConverter &typeConverter,
                               MLIRContext *context, PatternBenefit benefit = 1)
      : OpConversionPattern(typeConverter, context, benefit), backend(backend) {
  }

  LogicalResult
  matchAndRewrite(enzyme::addSampleToTraceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto ctx = op->getContext();

    if (backend == "cpu") {
      auto moduleOp = op->getParentOfType<ModuleOp>();
      static int64_t fnNum = 0;

      auto llvmPtrType = LLVM::LLVMPointerType::get(ctx);
      auto llvmVoidType = LLVM::LLVMVoidType::get(ctx);
      auto loweredTraceType = RankedTensorType::get(
          {1}, IntegerType::get(ctx, 64, IntegerType::Unsigned));

      std::string addSampleToTraceFn = "enzyme_probprog_add_sample_to_trace";

      // Generate the LLVM function body
      std::string fnName =
          addSampleToTraceFn + "_wrapper_" + std::to_string(fnNum);
      fnNum++;
      {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(moduleOp.getBody());

        auto funcType = LLVM::LLVMFunctionType::get(
            llvmVoidType, {llvmPtrType, llvmPtrType, llvmPtrType}, false);

        auto func =
            rewriter.create<LLVM::LLVMFuncOp>(op.getLoc(), fnName, funcType);
        rewriter.setInsertionPointToStart(func.addEntryBlock(rewriter));

        auto callResult = rewriter.create<LLVM::CallOp>(
            op.getLoc(), TypeRange{},
            SymbolRefAttr::get(ctx, addSampleToTraceFn),
            ValueRange{
                func.getArgument(0),
                func.getArgument(1),
                func.getArgument(2),
            });

        rewriter.create<LLVM::ReturnOp>(op.getLoc(), ValueRange{});
      }

      // Insert function declaration if not already present
      if (!moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(addSampleToTraceFn)) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(moduleOp.getBody());

        auto funcType = LLVM::LLVMFunctionType::get(
            llvmVoidType, {llvmPtrType, llvmPtrType, llvmPtrType}, false);

        rewriter.create<LLVM::LLVMFuncOp>(op.getLoc(), addSampleToTraceFn,
                                          funcType, LLVM::Linkage::External);
      }

      // Call the LLVM function with enzymexla.jit_call
      auto jitCall = rewriter.create<enzymexla::JITCallOp>(
          op.getLoc(), TypeRange{}, mlir::FlatSymbolRefAttr::get(ctx, fnName),
          ValueRange{}, rewriter.getStringAttr(""),
          /*operand_layouts=*/rewriter.getArrayAttr({}),
          /*result_layouts=*/rewriter.getArrayAttr({}),
          /*output_operand_aliases=*/rewriter.getArrayAttr({}),
          /*xla_side_effect_free=*/nullptr);

      // Replace the addSampleToTraceOp with the result of the JIT call
      rewriter.replaceOp(op, jitCall.getResults());

      return success();
    } else {
      return rewriter.notifyMatchFailure(op, "Unknown backend " + backend);
    }
  }
};

struct UnrealizedConversionCastOpConversion
    : public OpConversionPattern<UnrealizedConversionCastOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(UnrealizedConversionCastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.getNumResults() == 1 && op.getNumOperands() == 1) {
      auto sourceType = adaptor.getOperands()[0].getType();
      auto targetType =
          getTypeConverter()->convertType(op.getResult(0).getType());

      if (sourceType == targetType) {
        rewriter.replaceOp(op, adaptor.getOperands()[0]);
        return success();
      }
    }

    return failure();
  }
};

struct LowerEnzymeProbProgPass
    : public enzyme::impl::LowerEnzymeProbProgPassBase<
          LowerEnzymeProbProgPass> {
  using LowerEnzymeProbProgPassBase::LowerEnzymeProbProgPassBase;

  void runOnOperation() override {
    auto context = getOperation()->getContext();

    TypeConverter typeConverter;
    typeConverter.addConversion([](Type t) { return t; });
    typeConverter.addConversion([&](enzyme::TraceType t) {
      return RankedTensorType::get(
          {1},
          IntegerType::get(context, /*bitwidth=*/64, IntegerType::Unsigned));
    });

    ConversionTarget target(*context);

    target.addLegalDialect<LLVM::LLVMDialect>();
    target.addLegalDialect<func::FuncDialect>();
    target.addLegalDialect<enzymexla::EnzymeXLADialect>();
    target.addIllegalOp<enzyme::initTraceOp>();
    target.addIllegalOp<enzyme::addSampleToTraceOp>();
    target.addDynamicallyLegalOp<UnrealizedConversionCastOp>(
        [&](UnrealizedConversionCastOp op) {
          return typeConverter.isLegal(op.getOperation());
        });
    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp f) {
      return typeConverter.isSignatureLegal(f.getFunctionType());
    });
    target.addDynamicallyLegalOp<func::CallOp>([&](func::CallOp c) {
      return typeConverter.isSignatureLegal(c.getCalleeType());
    });
    target.addDynamicallyLegalOp<func::ReturnOp>(
        [&](func::ReturnOp r) { return typeConverter.isLegal(r); });

    RewritePatternSet patterns(context);
    patterns.add<InitTraceOpConversion>(backend, typeConverter, context);
    patterns.add<FuncOpConversion>(backend, typeConverter, context);
    patterns.add<CallOpConversion>(backend, typeConverter, context);
    patterns.add<ReturnOpConversion>(backend, typeConverter, context);
    patterns.add<addSampleToTraceOpConversion>(backend, typeConverter, context);
    patterns.add<UnrealizedConversionCastOpConversion>(typeConverter, context);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      signalPassFailure();
    }

    getOperation()->dump();
  }
};
