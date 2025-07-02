#include "Enzyme/MLIR/Dialect/Ops.h"
#include "mhlo/IR/hlo_ops.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"
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

struct InitTraceOpConversion : public OpConversionPattern<enzyme::InitTraceOp> {
  using OpConversionPattern::OpConversionPattern;

  std::string backend;
  InitTraceOpConversion(std::string backend, TypeConverter &typeConverter,
                        MLIRContext *context, PatternBenefit benefit = 1)
      : OpConversionPattern(typeConverter, context, benefit), backend(backend) {
  }

  LogicalResult
  matchAndRewrite(enzyme::InitTraceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto ctx = op->getContext();

    if (backend == "cpu") {
      auto moduleOp = op->getParentOfType<ModuleOp>();
      static int64_t fnNum = 0;

      auto llvmVoidType = LLVM::LLVMVoidType::get(ctx);
      auto llvmPtrType = LLVM::LLVMPointerType::get(ctx);
      auto loweredTraceType = RankedTensorType::get(
          {}, IntegerType::get(ctx, 64, IntegerType::Unsigned));
      auto tracePtr = rewriter.create<stablehlo::ConstantOp>(
          op.getLoc(), loweredTraceType,
          cast<ElementsAttr>(makeAttr(loweredTraceType, 0)));

      std::string initTraceFn = "enzyme_probprog_init_trace";
      std::string wrapperFn =
          initTraceFn + "_wrapper_" + std::to_string(fnNum++);

      {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(moduleOp.getBody());

        auto funcType =
            LLVM::LLVMFunctionType::get(llvmVoidType, {llvmPtrType}, false);
        auto func =
            rewriter.create<LLVM::LLVMFuncOp>(op.getLoc(), wrapperFn, funcType);
        rewriter.setInsertionPointToStart(func.addEntryBlock(rewriter));

        rewriter.create<LLVM::CallOp>(op.getLoc(), TypeRange{},
                                      SymbolRefAttr::get(ctx, initTraceFn),
                                      ValueRange{func.getArgument(0)});

        rewriter.create<LLVM::ReturnOp>(op.getLoc(), ValueRange{});
      }

      if (!moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(initTraceFn)) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(moduleOp.getBody());

        auto funcType =
            LLVM::LLVMFunctionType::get(llvmVoidType, {llvmPtrType}, false);

        rewriter.create<LLVM::LLVMFuncOp>(op.getLoc(), initTraceFn, funcType,
                                          LLVM::Linkage::External);
      }

      SmallVector<Attribute> aliases;
      aliases.push_back(stablehlo::OutputOperandAliasAttr::get(
          ctx, std::vector<int64_t>{}, 0, std::vector<int64_t>{}));

      auto jitCall = rewriter.create<enzymexla::JITCallOp>(
          op.getLoc(), TypeRange{loweredTraceType},
          mlir::FlatSymbolRefAttr::get(ctx, wrapperFn), ValueRange{tracePtr},
          rewriter.getStringAttr(""),
          /*operand_layouts=*/
          rewriter.getArrayAttr({rewriter.getIndexTensorAttr({})}),
          /*result_layouts=*/
          rewriter.getArrayAttr({rewriter.getIndexTensorAttr({})}),
          /*output_operand_aliases=*/rewriter.getArrayAttr(aliases),
          /*xla_side_effect_free=*/rewriter.getUnitAttr());

      rewriter.replaceOp(op, jitCall.getResults());

      return success();
    } else {
      return rewriter.notifyMatchFailure(op, "Unknown backend " + backend);
    }
  }
};

struct AddSampleToTraceOpConversion
    : public OpConversionPattern<enzyme::AddSampleToTraceOp> {
  using OpConversionPattern::OpConversionPattern;

  std::string backend;
  AddSampleToTraceOpConversion(std::string backend,
                               TypeConverter &typeConverter,
                               MLIRContext *context, PatternBenefit benefit = 1)
      : OpConversionPattern(typeConverter, context, benefit), backend(backend) {
  }

  LogicalResult
  matchAndRewrite(enzyme::AddSampleToTraceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto ctx = op->getContext();

    Value trace = adaptor.getTrace();
    auto sample = adaptor.getSample();

    auto symbolWrappedAttr = op.getSymbolAttr();
    if (!symbolWrappedAttr) {
      return rewriter.notifyMatchFailure(op, "Missing symbol attribute");
    }

    uint64_t symbolValue = symbolWrappedAttr.getPtr();

    size_t numSamples = sample.size();
    if (numSamples == 0)
      return rewriter.notifyMatchFailure(op, "No samples provided");

    if (backend == "cpu") {
      auto moduleOp = op->getParentOfType<ModuleOp>();
      static int64_t fnNum = 0;

      auto llvmPtrType = LLVM::LLVMPointerType::get(ctx);
      auto llvmVoidType = LLVM::LLVMVoidType::get(ctx);
      auto llvmI64Type = IntegerType::get(ctx, 64);

      std::string addSampleToTraceFn = "enzyme_probprog_add_sample_to_trace";

      auto i64TensorType = RankedTensorType::get({}, llvmI64Type);

      auto symbolConst = rewriter.create<stablehlo::ConstantOp>(
          op.getLoc(), i64TensorType,
          cast<ElementsAttr>(makeAttr(i64TensorType, symbolValue)));

      SmallVector<Type> llvmArgTypes; // (trace, symbol, n_sample_pointers...)
      llvmArgTypes.push_back(llvmPtrType);
      llvmArgTypes.push_back(llvmPtrType);
      llvmArgTypes.append(numSamples, llvmPtrType);

      auto funcType = LLVM::LLVMFunctionType::get(llvmVoidType, llvmArgTypes,
                                                  /*isVarArg=*/false);

      std::string wrapperFn =
          addSampleToTraceFn + "_wrapper_" + std::to_string(fnNum++);

      {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(moduleOp.getBody());
        auto func =
            rewriter.create<LLVM::LLVMFuncOp>(op.getLoc(), wrapperFn, funcType);

        rewriter.setInsertionPointToStart(func.addEntryBlock(rewriter));

        auto oneConst = rewriter.create<LLVM::ConstantOp>(
            op.getLoc(), llvmI64Type, rewriter.getIntegerAttr(llvmI64Type, 1));
        auto numSamplesConst = rewriter.create<LLVM::ConstantOp>(
            op.getLoc(), llvmI64Type,
            rewriter.getIntegerAttr(llvmI64Type, numSamples));
        auto numSamplesAlloca = rewriter.create<LLVM::AllocaOp>(
            op.getLoc(), llvmPtrType, llvmI64Type, oneConst);
        rewriter.create<LLVM::StoreOp>(op.getLoc(), numSamplesConst,
                                       numSamplesAlloca);

        // Metadata pointer arrays of size `numSamples`
        auto samplePtrArrayAlloca = rewriter.create<LLVM::AllocaOp>(
            op.getLoc(), llvmPtrType, llvmPtrType, numSamplesConst);
        auto numDimsArrayAlloca = rewriter.create<LLVM::AllocaOp>(
            op.getLoc(), llvmPtrType, llvmI64Type, numSamplesConst);
        auto shapePtrArrayAlloca = rewriter.create<LLVM::AllocaOp>(
            op.getLoc(), llvmPtrType, llvmPtrType, numSamplesConst);
        auto dtypeWidthArrayAlloca = rewriter.create<LLVM::AllocaOp>(
            op.getLoc(), llvmPtrType, llvmI64Type, numSamplesConst);

        for (size_t i = 0; i < numSamples; ++i) {
          auto sampleType = cast<RankedTensorType>(sample[i].getType());
          auto sampleShape = sampleType.getShape();
          size_t sampleNumDims = sampleShape.size();
          size_t sampleWidth =
              sampleType.getElementType().getIntOrFloatBitWidth();

          // 1. Store `sample` pointer in `samplePtrArrayAlloca` for each
          // sampled value.
          auto samplePtrGEP = rewriter.create<LLVM::GEPOp>(
              op.getLoc(), llvmPtrType, llvmI64Type, samplePtrArrayAlloca,
              ValueRange{rewriter.create<LLVM::ConstantOp>(
                  op.getLoc(), llvmI64Type,
                  rewriter.getIntegerAttr(llvmI64Type, i))});
          rewriter.create<LLVM::StoreOp>(op.getLoc(), func.getArgument(2 + i),
                                         samplePtrGEP);

          // 2. Store `numDims` in `numDimsArrayAlloca` for each sampled value.
          auto numDimsConst = rewriter.create<LLVM::ConstantOp>(
              op.getLoc(), llvmI64Type,
              rewriter.getIntegerAttr(llvmI64Type, sampleNumDims));
          auto numDimsGEP = rewriter.create<LLVM::GEPOp>(
              op.getLoc(), llvmPtrType, llvmI64Type, numDimsArrayAlloca,
              ValueRange{rewriter.create<LLVM::ConstantOp>(
                  op.getLoc(), llvmI64Type,
                  rewriter.getIntegerAttr(llvmI64Type, i))});
          rewriter.create<LLVM::StoreOp>(op.getLoc(), numDimsConst, numDimsGEP);

          // 3. Store `dtypeWidth` in `dtypeWidthArrayAlloca` for each sampled
          // value.
          auto widthConst = rewriter.create<LLVM::ConstantOp>(
              op.getLoc(), llvmI64Type,
              rewriter.getIntegerAttr(llvmI64Type, sampleWidth));
          auto widthGEP = rewriter.create<LLVM::GEPOp>(
              op.getLoc(), llvmPtrType, llvmI64Type, dtypeWidthArrayAlloca,
              ValueRange{rewriter.create<LLVM::ConstantOp>(
                  op.getLoc(), llvmI64Type,
                  rewriter.getIntegerAttr(llvmI64Type, i))});
          rewriter.create<LLVM::StoreOp>(op.getLoc(), widthConst, widthGEP);

          // 4a. Allocate and fill shape array for this sample
          auto shapeSizeConst = rewriter.create<LLVM::ConstantOp>(
              op.getLoc(), llvmI64Type,
              rewriter.getIntegerAttr(llvmI64Type, sampleNumDims));
          auto shapeArrAlloca = rewriter.create<LLVM::AllocaOp>(
              op.getLoc(), llvmPtrType, llvmI64Type, shapeSizeConst);

          for (size_t j = 0; j < sampleNumDims; ++j) {
            auto dimConst = rewriter.create<LLVM::ConstantOp>(
                op.getLoc(), llvmI64Type,
                rewriter.getIntegerAttr(llvmI64Type, sampleShape[j]));
            auto dimGEP = rewriter.create<LLVM::GEPOp>(
                op.getLoc(), llvmPtrType, llvmI64Type, shapeArrAlloca,
                ValueRange{rewriter.create<LLVM::ConstantOp>(
                    op.getLoc(), llvmI64Type,
                    rewriter.getIntegerAttr(llvmI64Type, j))});
            rewriter.create<LLVM::StoreOp>(op.getLoc(), dimConst, dimGEP);
          }

          // 4b. Store `shapeArrAlloca` in `shapePtrArrayAlloca` for each
          // sampled value.
          auto shapePtrGEP = rewriter.create<LLVM::GEPOp>(
              op.getLoc(), llvmPtrType, llvmI64Type, shapePtrArrayAlloca,
              ValueRange{rewriter.create<LLVM::ConstantOp>(
                  op.getLoc(), llvmI64Type,
                  rewriter.getIntegerAttr(llvmI64Type, i))});
          rewriter.create<LLVM::StoreOp>(op.getLoc(), shapeArrAlloca,
                                         shapePtrGEP);
        }

        rewriter.create<LLVM::CallOp>(
            op.getLoc(), TypeRange{},
            SymbolRefAttr::get(ctx, addSampleToTraceFn),
            ValueRange{func.getArgument(0), func.getArgument(1),
                       samplePtrArrayAlloca, numSamplesAlloca,
                       numDimsArrayAlloca, shapePtrArrayAlloca,
                       dtypeWidthArrayAlloca});

        rewriter.create<LLVM::ReturnOp>(op.getLoc(), ValueRange{});
      }

      if (!moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(addSampleToTraceFn)) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(moduleOp.getBody());

        auto funcType = LLVM::LLVMFunctionType::get(
            llvmVoidType,
            {llvmPtrType, llvmPtrType, llvmPtrType, llvmPtrType, llvmPtrType,
             llvmPtrType, llvmPtrType},
            false);
        rewriter.create<LLVM::LLVMFuncOp>(op.getLoc(), addSampleToTraceFn,
                                          funcType, LLVM::Linkage::External);
      }

      SmallVector<Value> jitOperands;
      jitOperands.push_back(trace);
      jitOperands.push_back(symbolConst);
      jitOperands.append(sample.begin(), sample.end());

      rewriter.create<enzymexla::JITCallOp>(
          op.getLoc(), TypeRange{},
          mlir::FlatSymbolRefAttr::get(ctx, wrapperFn), jitOperands,
          rewriter.getStringAttr(""),
          /*operand_layouts=*/nullptr,
          /*result_layouts=*/nullptr,
          /*output_operand_aliases=*/rewriter.getArrayAttr({}),
          /*xla_side_effect_free=*/nullptr);

      rewriter.eraseOp(op);

      return success();
    } else {
      return rewriter.notifyMatchFailure(op, "Unknown backend " + backend);
    }
  }
};

struct AddSubtraceOpConversion
    : public OpConversionPattern<enzyme::AddSubtraceOp> {
  using OpConversionPattern::OpConversionPattern;

  std::string backend;
  AddSubtraceOpConversion(std::string backend, TypeConverter &typeConverter,
                          MLIRContext *context, PatternBenefit benefit = 1)
      : OpConversionPattern(typeConverter, context, benefit), backend(backend) {
  }

  LogicalResult
  matchAndRewrite(enzyme::AddSubtraceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto ctx = op->getContext();

    Value subtrace = adaptor.getSubtrace();
    Value trace = adaptor.getTrace();

    auto symbolWrappedAttr = op.getSymbolAttr();
    if (!symbolWrappedAttr)
      return rewriter.notifyMatchFailure(op, "Missing symbol attribute");

    uint64_t symbolPtr = symbolWrappedAttr.getPtr();

    if (backend == "cpu") {
      auto moduleOp = op->getParentOfType<ModuleOp>();
      static int64_t fnNum = 0;

      auto llvmPtrType = LLVM::LLVMPointerType::get(ctx);
      auto llvmVoidType = LLVM::LLVMVoidType::get(ctx);
      auto i64Type = IntegerType::get(ctx, 64);
      auto i64TensorType = RankedTensorType::get({}, i64Type);

      std::string addSubtraceFn = "enzyme_probprog_add_subtrace";
      std::string wrapperFn =
          addSubtraceFn + "_wrapper_" + std::to_string(fnNum++);

      auto symbolConst = rewriter.create<stablehlo::ConstantOp>(
          op.getLoc(), i64TensorType,
          cast<ElementsAttr>(makeAttr(i64TensorType, symbolPtr)));

      {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(moduleOp.getBody());

        auto funcType = LLVM::LLVMFunctionType::get(
            llvmVoidType, {llvmPtrType, llvmPtrType, llvmPtrType},
            /*isVarArg=*/false);
        auto func =
            rewriter.create<LLVM::LLVMFuncOp>(op.getLoc(), wrapperFn, funcType);
        rewriter.setInsertionPointToStart(func.addEntryBlock(rewriter));

        rewriter.create<LLVM::CallOp>(
            op.getLoc(), TypeRange{}, SymbolRefAttr::get(ctx, addSubtraceFn),
            ValueRange{func.getArgument(0), func.getArgument(1),
                       func.getArgument(2)});

        rewriter.create<LLVM::ReturnOp>(op.getLoc(), ValueRange{});
      }

      if (!moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(addSubtraceFn)) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(moduleOp.getBody());
        auto funcType = LLVM::LLVMFunctionType::get(
            llvmVoidType, {llvmPtrType, llvmPtrType, llvmPtrType},
            /*isVarArg=*/false);
        rewriter.create<LLVM::LLVMFuncOp>(op.getLoc(), addSubtraceFn, funcType,
                                          LLVM::Linkage::External);
      }

      rewriter.create<enzymexla::JITCallOp>(
          op.getLoc(), TypeRange{},
          mlir::FlatSymbolRefAttr::get(ctx, wrapperFn),
          ValueRange{trace, symbolConst, subtrace}, rewriter.getStringAttr(""),
          /*operand_layouts=*/nullptr,
          /*result_layouts=*/nullptr,
          /*output_operand_aliases=*/rewriter.getArrayAttr({}),
          /*xla_side_effect_free=*/nullptr);

      rewriter.eraseOp(op);
      return success();
    } else {
      return rewriter.notifyMatchFailure(op, "Unknown backend " + backend);
    }
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
          {},
          IntegerType::get(context, /*bitwidth=*/64, IntegerType::Unsigned));
    });

    ConversionTarget target(*context);

    target.addLegalDialect<LLVM::LLVMDialect>();
    target.addLegalDialect<func::FuncDialect>();
    target.addLegalDialect<enzymexla::EnzymeXLADialect>();
    target.addLegalDialect<stablehlo::StablehloDialect>();

    target.addIllegalOp<enzyme::InitTraceOp>();
    target.addIllegalOp<enzyme::AddSampleToTraceOp>();
    target.addIllegalOp<enzyme::AddSubtraceOp>();

    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp f) {
      return typeConverter.isSignatureLegal(f.getFunctionType());
    });
    target.addDynamicallyLegalOp<func::CallOp>([&](func::CallOp c) {
      return typeConverter.isSignatureLegal(c.getCalleeType());
    });
    target.addDynamicallyLegalOp<func::ReturnOp>(
        [&](func::ReturnOp r) { return typeConverter.isLegal(r); });

    RewritePatternSet patterns(context);

    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(
        patterns, typeConverter);
    populateCallOpTypeConversionPattern(patterns, typeConverter);
    populateReturnOpTypeConversionPattern(patterns, typeConverter);

    patterns.add<InitTraceOpConversion, AddSampleToTraceOpConversion,
                 AddSubtraceOpConversion>(backend, typeConverter, context);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      signalPassFailure();
    }

    getOperation()->dump();
  }
};
