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
#include "src/enzyme_ad/jax/Utils.h"
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

static std::string getTensorSignature(Type tensorType) {
  if (auto rankedType = dyn_cast<RankedTensorType>(tensorType)) {
    std::string sig;

    for (auto dim : rankedType.getShape()) {
      sig += std::to_string(dim) + "x";
    }

    auto elemType = rankedType.getElementType();
    if (elemType.isF32())
      sig += "f32";
    else if (elemType.isF64())
      sig += "f64";
    else
      llvm_unreachable("Unsupported tensor element type");

    return sig;
  }
  return "ptr";
}

static std::string getOrCreateWrapper(const std::string &baseFnName,
                                      ArrayRef<Type> originalTypes = {}) {
  std::string signature = baseFnName;
  for (Type t : originalTypes) {
    signature += "_" + getTensorSignature(t);
  }

  static llvm::StringMap<std::string> signatureToWrapper;
  auto it = signatureToWrapper.find(signature);
  if (it != signatureToWrapper.end()) {
    return it->second;
  }

  static llvm::StringMap<int64_t> fnCounters;
  int64_t fnNum = fnCounters[baseFnName]++;
  std::string wrapperName = baseFnName + "_wrapper_" + std::to_string(fnNum);
  signatureToWrapper[signature] = wrapperName;
  return wrapperName;
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

      auto llvmVoidType = LLVM::LLVMVoidType::get(ctx);
      auto llvmPtrType = LLVM::LLVMPointerType::get(ctx);
      auto loweredTraceType = RankedTensorType::get(
          {}, IntegerType::get(ctx, 64, IntegerType::Unsigned));
      auto tracePtr = rewriter.create<stablehlo::ConstantOp>(
          op.getLoc(), loweredTraceType,
          cast<ElementsAttr>(makeAttr(loweredTraceType, 0)));

      std::string initTraceFn = "enzyme_probprog_init_trace";
      auto funcType =
          LLVM::LLVMFunctionType::get(llvmVoidType, {llvmPtrType}, false);
      std::string wrapperFn = getOrCreateWrapper(initTraceFn);

      if (!moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(wrapperFn)) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(moduleOp.getBody());

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
          /*arg_attrs=*/nullptr,
          /*res_attrs=*/nullptr,
          /*output_operand_aliases=*/rewriter.getArrayAttr(aliases),
          /*xla_side_effect_free=*/nullptr);

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

      auto llvmPtrType = LLVM::LLVMPointerType::get(ctx);
      auto llvmVoidType = LLVM::LLVMVoidType::get(ctx);
      auto llvmI64Type = IntegerType::get(ctx, 64);
      auto loweredTraceType = RankedTensorType::get(
          {}, IntegerType::get(ctx, 64, IntegerType::Unsigned));

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

      SmallVector<Type> originalTypes;
      for (auto s : sample) {
        originalTypes.push_back(s.getType());
      }

      std::string wrapperFn =
          getOrCreateWrapper(addSampleToTraceFn, originalTypes);

      if (!moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(wrapperFn)) {
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

          // 2. Store `numDims` in `numDimsArrayAlloca` for each sampled
          // value.
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

      SmallVector<Attribute> aliases;
      aliases.push_back(stablehlo::OutputOperandAliasAttr::get(
          ctx, std::vector<int64_t>{}, 0, std::vector<int64_t>{}));

      auto jitCall = rewriter.create<enzymexla::JITCallOp>(
          op.getLoc(), TypeRange{loweredTraceType},
          mlir::FlatSymbolRefAttr::get(ctx, wrapperFn), jitOperands,
          rewriter.getStringAttr(""),
          /*operand_layouts=*/nullptr,
          /*result_layouts=*/nullptr,
          /*arg_attrs=*/nullptr,
          /*res_attrs=*/nullptr,
          /*output_operand_aliases=*/rewriter.getArrayAttr(aliases),
          /*xla_side_effect_free=*/nullptr);

      rewriter.replaceOp(op, jitCall.getResults());

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

      auto llvmPtrType = LLVM::LLVMPointerType::get(ctx);
      auto llvmVoidType = LLVM::LLVMVoidType::get(ctx);
      auto i64Type = IntegerType::get(ctx, 64);
      auto i64TensorType = RankedTensorType::get({}, i64Type);
      auto loweredTraceType = RankedTensorType::get(
          {}, IntegerType::get(ctx, 64, IntegerType::Unsigned));

      std::string addSubtraceFn = "enzyme_probprog_add_subtrace";
      std::string wrapperFn = getOrCreateWrapper(addSubtraceFn);

      auto symbolConst = rewriter.create<stablehlo::ConstantOp>(
          op.getLoc(), i64TensorType,
          cast<ElementsAttr>(makeAttr(i64TensorType, symbolPtr)));

      if (!moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(wrapperFn)) {
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

      SmallVector<Attribute> aliases;
      aliases.push_back(stablehlo::OutputOperandAliasAttr::get(
          ctx, std::vector<int64_t>{}, 0, std::vector<int64_t>{}));

      auto jitCall = rewriter.create<enzymexla::JITCallOp>(
          op.getLoc(), TypeRange{loweredTraceType},
          mlir::FlatSymbolRefAttr::get(ctx, wrapperFn),
          ValueRange{trace, symbolConst, subtrace}, rewriter.getStringAttr(""),
          /*operand_layouts=*/nullptr,
          /*result_layouts=*/nullptr,
          /*arg_attrs=*/nullptr,
          /*res_attrs=*/nullptr,
          /*output_operand_aliases=*/rewriter.getArrayAttr(aliases),
          /*xla_side_effect_free=*/nullptr);

      rewriter.replaceOp(op, jitCall.getResults());

      return success();
    } else {
      return rewriter.notifyMatchFailure(op, "Unknown backend " + backend);
    }
  }
};

struct AddWeightToTraceOpConversion
    : public OpConversionPattern<enzyme::AddWeightToTraceOp> {
  using OpConversionPattern::OpConversionPattern;

  std::string backend;
  AddWeightToTraceOpConversion(std::string backend,
                               TypeConverter &typeConverter,
                               MLIRContext *context, PatternBenefit benefit = 1)
      : OpConversionPattern(typeConverter, context, benefit), backend(backend) {
  }

  LogicalResult
  matchAndRewrite(enzyme::AddWeightToTraceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto ctx = op->getContext();

    Value trace = adaptor.getTrace();
    Value weight = adaptor.getWeight();

    if (backend == "cpu") {
      auto moduleOp = op->getParentOfType<ModuleOp>();

      auto llvmPtrType = LLVM::LLVMPointerType::get(ctx);
      auto llvmVoidType = LLVM::LLVMVoidType::get(ctx);
      auto loweredTraceType = RankedTensorType::get(
          {}, IntegerType::get(ctx, 64, IntegerType::Unsigned));

      std::string addWeightFn = "enzyme_probprog_add_weight_to_trace";
      SmallVector<Type> originalTypes = {weight.getType()};
      std::string wrapperFn = getOrCreateWrapper(addWeightFn, originalTypes);

      if (!moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(wrapperFn)) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(moduleOp.getBody());

        auto funcType = LLVM::LLVMFunctionType::get(
            llvmVoidType, {llvmPtrType, llvmPtrType}, /*isVarArg=*/false);
        auto func =
            rewriter.create<LLVM::LLVMFuncOp>(op.getLoc(), wrapperFn, funcType);

        rewriter.setInsertionPointToStart(func.addEntryBlock(rewriter));
        rewriter.create<LLVM::CallOp>(op.getLoc(), TypeRange{},
                                      SymbolRefAttr::get(ctx, addWeightFn),
                                      func.getArguments());
        rewriter.create<LLVM::ReturnOp>(op.getLoc(), ValueRange{});
      }

      if (!moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(addWeightFn)) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(moduleOp.getBody());
        auto funcType = LLVM::LLVMFunctionType::get(
            llvmVoidType, {llvmPtrType, llvmPtrType}, /*isVarArg=*/false);
        rewriter.create<LLVM::LLVMFuncOp>(op.getLoc(), addWeightFn, funcType,
                                          LLVM::Linkage::External);
      }

      SmallVector<Attribute> aliases;
      aliases.push_back(stablehlo::OutputOperandAliasAttr::get(
          ctx, std::vector<int64_t>{}, 0, std::vector<int64_t>{}));

      auto jitCall = rewriter.create<enzymexla::JITCallOp>(
          op.getLoc(), TypeRange{loweredTraceType},
          mlir::FlatSymbolRefAttr::get(ctx, wrapperFn),
          ValueRange{trace, weight}, rewriter.getStringAttr(""),
          /*operand_layouts=*/nullptr, /*result_layouts=*/nullptr,
          /*arg_attrs=*/nullptr, /*res_attrs=*/nullptr,
          /*output_operand_aliases=*/rewriter.getArrayAttr(aliases),
          /*xla_side_effect_free=*/nullptr);

      rewriter.replaceOp(op, jitCall.getResults());

      return success();
    }

    return rewriter.notifyMatchFailure(op, "Unknown backend " + backend);
  }
};

struct AddRetvalToTraceOpConversion
    : public OpConversionPattern<enzyme::AddRetvalToTraceOp> {
  using OpConversionPattern::OpConversionPattern;

  std::string backend;
  AddRetvalToTraceOpConversion(std::string backend,
                               TypeConverter &typeConverter,
                               MLIRContext *context, PatternBenefit benefit = 1)
      : OpConversionPattern(typeConverter, context, benefit), backend(backend) {
  }

  LogicalResult
  matchAndRewrite(enzyme::AddRetvalToTraceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto ctx = op->getContext();

    Value trace = adaptor.getTrace();
    auto retvalVals = adaptor.getRetval();

    size_t numResults = retvalVals.size();
    if (numResults == 0)
      return rewriter.notifyMatchFailure(op, "No retval values provided");

    if (backend == "cpu") {
      auto moduleOp = op->getParentOfType<ModuleOp>();

      auto llvmPtrType = LLVM::LLVMPointerType::get(ctx);
      auto llvmVoidType = LLVM::LLVMVoidType::get(ctx);
      auto llvmI64Type = IntegerType::get(ctx, 64);
      auto loweredTraceType = RankedTensorType::get(
          {}, IntegerType::get(ctx, 64, IntegerType::Unsigned));

      std::string addRetvalFn = "enzyme_probprog_add_retval_to_trace";

      SmallVector<Type> llvmArgTypes;
      llvmArgTypes.push_back(llvmPtrType);
      llvmArgTypes.append(numResults, llvmPtrType);

      auto funcType =
          LLVM::LLVMFunctionType::get(llvmVoidType, llvmArgTypes, false);

      SmallVector<Type> originalTypes;
      for (auto rv : retvalVals) {
        originalTypes.push_back(rv.getType());
      }

      std::string wrapperFn = getOrCreateWrapper(addRetvalFn, originalTypes);

      if (!moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(wrapperFn)) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(moduleOp.getBody());
        auto func =
            rewriter.create<LLVM::LLVMFuncOp>(op.getLoc(), wrapperFn, funcType);

        rewriter.setInsertionPointToStart(func.addEntryBlock(rewriter));

        auto oneConst = rewriter.create<LLVM::ConstantOp>(
            op.getLoc(), llvmI64Type, rewriter.getIntegerAttr(llvmI64Type, 1));
        auto numResultsConst = rewriter.create<LLVM::ConstantOp>(
            op.getLoc(), llvmI64Type,
            rewriter.getIntegerAttr(llvmI64Type, numResults));

        auto numResultsAlloca = rewriter.create<LLVM::AllocaOp>(
            op.getLoc(), llvmPtrType, llvmI64Type, oneConst);
        rewriter.create<LLVM::StoreOp>(op.getLoc(), numResultsConst,
                                       numResultsAlloca);

        auto retvalPtrArrayAlloca = rewriter.create<LLVM::AllocaOp>(
            op.getLoc(), llvmPtrType, llvmPtrType, numResultsConst);
        auto numDimsArrayAlloca = rewriter.create<LLVM::AllocaOp>(
            op.getLoc(), llvmPtrType, llvmI64Type, numResultsConst);
        auto shapePtrArrayAlloca = rewriter.create<LLVM::AllocaOp>(
            op.getLoc(), llvmPtrType, llvmPtrType, numResultsConst);
        auto dtypeWidthArrayAlloca = rewriter.create<LLVM::AllocaOp>(
            op.getLoc(), llvmPtrType, llvmI64Type, numResultsConst);

        for (size_t i = 0; i < numResults; ++i) {
          auto resType = cast<RankedTensorType>(retvalVals[i].getType());
          auto resShape = resType.getShape();
          size_t resNumDims = resShape.size();
          size_t resWidth = resType.getElementType().getIntOrFloatBitWidth();

          auto ptrGEP = rewriter.create<LLVM::GEPOp>(
              op.getLoc(), llvmPtrType, llvmI64Type, retvalPtrArrayAlloca,
              ValueRange{rewriter.create<LLVM::ConstantOp>(
                  op.getLoc(), llvmI64Type,
                  rewriter.getIntegerAttr(llvmI64Type, i))});
          rewriter.create<LLVM::StoreOp>(op.getLoc(), func.getArgument(1 + i),
                                         ptrGEP);

          auto numDimsConst = rewriter.create<LLVM::ConstantOp>(
              op.getLoc(), llvmI64Type,
              rewriter.getIntegerAttr(llvmI64Type, resNumDims));
          auto numDimsGEP = rewriter.create<LLVM::GEPOp>(
              op.getLoc(), llvmPtrType, llvmI64Type, numDimsArrayAlloca,
              ValueRange{rewriter.create<LLVM::ConstantOp>(
                  op.getLoc(), llvmI64Type,
                  rewriter.getIntegerAttr(llvmI64Type, i))});
          rewriter.create<LLVM::StoreOp>(op.getLoc(), numDimsConst, numDimsGEP);

          auto widthConst = rewriter.create<LLVM::ConstantOp>(
              op.getLoc(), llvmI64Type,
              rewriter.getIntegerAttr(llvmI64Type, resWidth));
          auto widthGEP = rewriter.create<LLVM::GEPOp>(
              op.getLoc(), llvmPtrType, llvmI64Type, dtypeWidthArrayAlloca,
              ValueRange{rewriter.create<LLVM::ConstantOp>(
                  op.getLoc(), llvmI64Type,
                  rewriter.getIntegerAttr(llvmI64Type, i))});
          rewriter.create<LLVM::StoreOp>(op.getLoc(), widthConst, widthGEP);

          auto shapeSizeConst = rewriter.create<LLVM::ConstantOp>(
              op.getLoc(), llvmI64Type,
              rewriter.getIntegerAttr(llvmI64Type, resNumDims));
          auto shapeArrAlloca = rewriter.create<LLVM::AllocaOp>(
              op.getLoc(), llvmPtrType, llvmI64Type, shapeSizeConst);

          for (size_t j = 0; j < resNumDims; ++j) {
            auto dimConst = rewriter.create<LLVM::ConstantOp>(
                op.getLoc(), llvmI64Type,
                rewriter.getIntegerAttr(llvmI64Type, resShape[j]));
            auto dimGEP = rewriter.create<LLVM::GEPOp>(
                op.getLoc(), llvmPtrType, llvmI64Type, shapeArrAlloca,
                ValueRange{rewriter.create<LLVM::ConstantOp>(
                    op.getLoc(), llvmI64Type,
                    rewriter.getIntegerAttr(llvmI64Type, j))});
            rewriter.create<LLVM::StoreOp>(op.getLoc(), dimConst, dimGEP);
          }

          auto shapePtrGEP = rewriter.create<LLVM::GEPOp>(
              op.getLoc(), llvmPtrType, llvmI64Type, shapePtrArrayAlloca,
              ValueRange{rewriter.create<LLVM::ConstantOp>(
                  op.getLoc(), llvmI64Type,
                  rewriter.getIntegerAttr(llvmI64Type, i))});
          rewriter.create<LLVM::StoreOp>(op.getLoc(), shapeArrAlloca,
                                         shapePtrGEP);
        }

        rewriter.create<LLVM::CallOp>(
            op.getLoc(), TypeRange{}, SymbolRefAttr::get(ctx, addRetvalFn),
            ValueRange{func.getArgument(0), retvalPtrArrayAlloca,
                       numResultsAlloca, numDimsArrayAlloca,
                       shapePtrArrayAlloca, dtypeWidthArrayAlloca});

        rewriter.create<LLVM::ReturnOp>(op.getLoc(), ValueRange{});
      }

      if (!moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(addRetvalFn)) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(moduleOp.getBody());
        auto funcTypeExt =
            LLVM::LLVMFunctionType::get(llvmVoidType,
                                        {llvmPtrType, llvmPtrType, llvmPtrType,
                                         llvmPtrType, llvmPtrType, llvmPtrType},
                                        false);
        rewriter.create<LLVM::LLVMFuncOp>(op.getLoc(), addRetvalFn, funcTypeExt,
                                          LLVM::Linkage::External);
      }

      SmallVector<Value> jitOperands;
      jitOperands.push_back(trace);
      jitOperands.append(retvalVals.begin(), retvalVals.end());

      SmallVector<Attribute> aliases;
      aliases.push_back(stablehlo::OutputOperandAliasAttr::get(
          ctx, std::vector<int64_t>{}, 0, std::vector<int64_t>{}));

      auto jitCall = rewriter.create<enzymexla::JITCallOp>(
          op.getLoc(), TypeRange{loweredTraceType},
          mlir::FlatSymbolRefAttr::get(ctx, wrapperFn), jitOperands,
          rewriter.getStringAttr(""), /*operand_layouts=*/nullptr,
          /*result_layouts=*/nullptr,
          /*arg_attrs=*/nullptr, /*res_attrs=*/nullptr,
          /*output_operand_aliases=*/rewriter.getArrayAttr(aliases),
          /*xla_side_effect_free=*/nullptr);

      rewriter.replaceOp(op, jitCall.getResults());

      return success();
    }

    return rewriter.notifyMatchFailure(op, "Unknown backend " + backend);
  }
};

struct UnrealizedConversionCastOpConversion
    : public OpConversionPattern<UnrealizedConversionCastOp> {
  using OpConversionPattern::OpConversionPattern;

  std::string backend;
  UnrealizedConversionCastOpConversion(std::string backend,
                                       TypeConverter &typeConverter,
                                       MLIRContext *context,
                                       PatternBenefit benefit = 1)
      : OpConversionPattern(typeConverter, context, benefit), backend(backend) {
  }

  LogicalResult
  matchAndRewrite(UnrealizedConversionCastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.getOperandTypes().size() == 1 && op.getResultTypes().size() == 1) {
      auto operandType = op.getOperandTypes().front();
      auto resultType = op.getResultTypes().front();

      if (isa<enzyme::TraceType>(operandType) ||
          isa<enzyme::TraceType>(resultType) ||
          isa<enzyme::ConstraintType>(operandType) ||
          isa<enzyme::ConstraintType>(resultType)) {
        rewriter.replaceOp(op, adaptor.getOperands());
        return success();
      }
    }

    return failure();
  }
};

struct GetSampleFromConstraintOpConversion
    : public OpConversionPattern<enzyme::GetSampleFromConstraintOp> {
  using OpConversionPattern::OpConversionPattern;

  std::string backend;
  GetSampleFromConstraintOpConversion(std::string backend,
                                      TypeConverter &typeConverter,
                                      MLIRContext *context,
                                      PatternBenefit benefit = 1)
      : OpConversionPattern(typeConverter, context, benefit), backend(backend) {
  }

  LogicalResult
  matchAndRewrite(enzyme::GetSampleFromConstraintOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto ctx = op->getContext();

    Value constraint = adaptor.getConstraint();
    auto outputs = op.getOutputs();

    auto symbolWrappedAttr = op.getSymbolAttr();
    if (!symbolWrappedAttr) {
      return rewriter.notifyMatchFailure(op, "Missing symbol attribute");
    }

    uint64_t symbolValue = symbolWrappedAttr.getPtr();

    size_t numOutputs = outputs.size();
    if (numOutputs == 0)
      return rewriter.notifyMatchFailure(
          op, "GetSampleFromConstraintOp has no outputs");

    if (backend == "cpu") {
      auto moduleOp = op->getParentOfType<ModuleOp>();

      auto llvmPtrType = LLVM::LLVMPointerType::get(ctx);
      auto llvmVoidType = LLVM::LLVMVoidType::get(ctx);
      auto llvmI64Type = IntegerType::get(ctx, 64);

      std::string getSampleFn = "enzyme_probprog_get_sample_from_constraint";

      auto i64TensorType = RankedTensorType::get({}, llvmI64Type);
      auto symbolConst = rewriter.create<stablehlo::ConstantOp>(
          op.getLoc(), i64TensorType,
          cast<ElementsAttr>(makeAttr(i64TensorType, symbolValue)));

      SmallVector<Type> llvmArgTypes; // (constraint, symbol, out_ptrs...)
      llvmArgTypes.push_back(llvmPtrType);
      llvmArgTypes.push_back(llvmPtrType);
      llvmArgTypes.append(numOutputs, llvmPtrType); // one per output tensor

      auto funcType = LLVM::LLVMFunctionType::get(llvmVoidType, llvmArgTypes,
                                                  /*isVarArg=*/false);

      SmallVector<Type> originalTypes;
      for (auto output : outputs) {
        originalTypes.push_back(output.getType());
      }

      std::string wrapperFn = getOrCreateWrapper(getSampleFn, originalTypes);

      if (!moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(wrapperFn)) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(moduleOp.getBody());

        auto func =
            rewriter.create<LLVM::LLVMFuncOp>(op.getLoc(), wrapperFn, funcType);

        rewriter.setInsertionPointToStart(func.addEntryBlock(rewriter));

        auto oneConst = rewriter.create<LLVM::ConstantOp>(
            op.getLoc(), llvmI64Type, rewriter.getIntegerAttr(llvmI64Type, 1));

        auto numOutputsConst = rewriter.create<LLVM::ConstantOp>(
            op.getLoc(), llvmI64Type,
            rewriter.getIntegerAttr(llvmI64Type, numOutputs));

        auto numOutputsAlloca = rewriter.create<LLVM::AllocaOp>(
            op.getLoc(), llvmPtrType, llvmI64Type, oneConst);
        rewriter.create<LLVM::StoreOp>(op.getLoc(), numOutputsConst,
                                       numOutputsAlloca);

        auto samplePtrArrayAlloca = rewriter.create<LLVM::AllocaOp>(
            op.getLoc(), llvmPtrType, llvmPtrType, numOutputsConst);
        auto numDimsArrayAlloca = rewriter.create<LLVM::AllocaOp>(
            op.getLoc(), llvmPtrType, llvmI64Type, numOutputsConst);
        auto shapePtrArrayAlloca = rewriter.create<LLVM::AllocaOp>(
            op.getLoc(), llvmPtrType, llvmPtrType, numOutputsConst);
        auto dtypeWidthArrayAlloca = rewriter.create<LLVM::AllocaOp>(
            op.getLoc(), llvmPtrType, llvmI64Type, numOutputsConst);

        for (size_t i = 0; i < numOutputs; ++i) {
          auto outType = cast<RankedTensorType>(outputs[i].getType());
          auto outShape = outType.getShape();
          size_t outNumDims = outShape.size();
          size_t outWidth = outType.getElementType().getIntOrFloatBitWidth();

          auto ptrGEP = rewriter.create<LLVM::GEPOp>(
              op.getLoc(), llvmPtrType, llvmI64Type, samplePtrArrayAlloca,
              ValueRange{rewriter.create<LLVM::ConstantOp>(
                  op.getLoc(), llvmI64Type,
                  rewriter.getIntegerAttr(llvmI64Type, i))});
          rewriter.create<LLVM::StoreOp>(op.getLoc(), func.getArgument(2 + i),
                                         ptrGEP);

          auto numDimsConst = rewriter.create<LLVM::ConstantOp>(
              op.getLoc(), llvmI64Type,
              rewriter.getIntegerAttr(llvmI64Type, outNumDims));
          auto numDimsGEP = rewriter.create<LLVM::GEPOp>(
              op.getLoc(), llvmPtrType, llvmI64Type, numDimsArrayAlloca,
              ValueRange{rewriter.create<LLVM::ConstantOp>(
                  op.getLoc(), llvmI64Type,
                  rewriter.getIntegerAttr(llvmI64Type, i))});
          rewriter.create<LLVM::StoreOp>(op.getLoc(), numDimsConst, numDimsGEP);

          auto widthConst = rewriter.create<LLVM::ConstantOp>(
              op.getLoc(), llvmI64Type,
              rewriter.getIntegerAttr(llvmI64Type, outWidth));
          auto widthGEP = rewriter.create<LLVM::GEPOp>(
              op.getLoc(), llvmPtrType, llvmI64Type, dtypeWidthArrayAlloca,
              ValueRange{rewriter.create<LLVM::ConstantOp>(
                  op.getLoc(), llvmI64Type,
                  rewriter.getIntegerAttr(llvmI64Type, i))});
          rewriter.create<LLVM::StoreOp>(op.getLoc(), widthConst, widthGEP);

          auto shapeSizeConst = rewriter.create<LLVM::ConstantOp>(
              op.getLoc(), llvmI64Type,
              rewriter.getIntegerAttr(llvmI64Type, outNumDims));
          auto shapeArrAlloca = rewriter.create<LLVM::AllocaOp>(
              op.getLoc(), llvmPtrType, llvmI64Type, shapeSizeConst);

          for (size_t j = 0; j < outNumDims; ++j) {
            auto dimConst = rewriter.create<LLVM::ConstantOp>(
                op.getLoc(), llvmI64Type,
                rewriter.getIntegerAttr(llvmI64Type, outShape[j]));
            auto dimGEP = rewriter.create<LLVM::GEPOp>(
                op.getLoc(), llvmPtrType, llvmI64Type, shapeArrAlloca,
                ValueRange{rewriter.create<LLVM::ConstantOp>(
                    op.getLoc(), llvmI64Type,
                    rewriter.getIntegerAttr(llvmI64Type, j))});
            rewriter.create<LLVM::StoreOp>(op.getLoc(), dimConst, dimGEP);
          }

          auto shapePtrGEP = rewriter.create<LLVM::GEPOp>(
              op.getLoc(), llvmPtrType, llvmI64Type, shapePtrArrayAlloca,
              ValueRange{rewriter.create<LLVM::ConstantOp>(
                  op.getLoc(), llvmI64Type,
                  rewriter.getIntegerAttr(llvmI64Type, i))});
          rewriter.create<LLVM::StoreOp>(op.getLoc(), shapeArrAlloca,
                                         shapePtrGEP);
        }

        rewriter.create<LLVM::CallOp>(
            op.getLoc(), TypeRange{}, SymbolRefAttr::get(ctx, getSampleFn),
            ValueRange{func.getArgument(0), func.getArgument(1),
                       samplePtrArrayAlloca, numOutputsAlloca,
                       numDimsArrayAlloca, shapePtrArrayAlloca,
                       dtypeWidthArrayAlloca});

        rewriter.create<LLVM::ReturnOp>(op.getLoc(), ValueRange{});
      }

      if (!moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(getSampleFn)) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(moduleOp.getBody());
        auto funcTypeExt = LLVM::LLVMFunctionType::get(
            llvmVoidType,
            {llvmPtrType, llvmPtrType, llvmPtrType, llvmPtrType, llvmPtrType,
             llvmPtrType, llvmPtrType},
            /*isVarArg=*/false);
        rewriter.create<LLVM::LLVMFuncOp>(op.getLoc(), getSampleFn, funcTypeExt,
                                          LLVM::Linkage::External);
      }

      SmallVector<Value> jitOperands;
      jitOperands.push_back(constraint);
      jitOperands.push_back(symbolConst);

      for (size_t i = 0; i < numOutputs; ++i) {
        auto outType = outputs[i].getType();
        auto bufConst = rewriter.create<stablehlo::ConstantOp>(
            op.getLoc(), outType, cast<ElementsAttr>(makeAttr(outType, 0)));
        jitOperands.push_back(bufConst);
      }

      SmallVector<Attribute> aliases;
      for (size_t i = 0; i < numOutputs; ++i) {
        aliases.push_back(stablehlo::OutputOperandAliasAttr::get(
            ctx, std::vector<int64_t>{}, /*operand_index=*/2 + i,
            std::vector<int64_t>{}));
      }

      auto jitCall = rewriter.create<enzymexla::JITCallOp>(
          op.getLoc(), op->getResultTypes(),
          mlir::FlatSymbolRefAttr::get(ctx, wrapperFn), jitOperands,
          rewriter.getStringAttr(""),
          /*operand_layouts=*/nullptr, /*result_layouts=*/nullptr,
          /*arg_attrs=*/nullptr, /*res_attrs=*/nullptr,
          /*output_operand_aliases=*/rewriter.getArrayAttr(aliases),
          /*xla_side_effect_free=*/nullptr);

      rewriter.replaceOp(op, jitCall.getResults());

      return success();
    }

    return rewriter.notifyMatchFailure(op, "Unknown backend" + backend);
  }
};

struct GetSubconstraintOpConversion
    : public OpConversionPattern<enzyme::GetSubconstraintOp> {
  using OpConversionPattern::OpConversionPattern;

  std::string backend;
  GetSubconstraintOpConversion(std::string backend,
                               TypeConverter &typeConverter,
                               MLIRContext *context, PatternBenefit benefit = 1)
      : OpConversionPattern(typeConverter, context, benefit), backend(backend) {
  }

  LogicalResult
  matchAndRewrite(enzyme::GetSubconstraintOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto ctx = op->getContext();

    Value constraint = adaptor.getConstraint();

    auto symbolAttr = op.getSymbolAttr();
    if (!symbolAttr) {
      return rewriter.notifyMatchFailure(op, "Missing symbol attribute");
    }

    uint64_t symbolValue = symbolAttr.getPtr();

    if (backend == "cpu") {
      auto moduleOp = op->getParentOfType<ModuleOp>();

      auto llvmPtrType = LLVM::LLVMPointerType::get(ctx);
      auto llvmVoidType = LLVM::LLVMVoidType::get(ctx);
      auto llvmI64Type = IntegerType::get(ctx, 64);
      auto loweredConstraintType = RankedTensorType::get(
          {}, IntegerType::get(ctx, 64, IntegerType::Unsigned));

      std::string getSubconstraintFn = "enzyme_probprog_get_subconstraint";

      auto i64TensorType = RankedTensorType::get({}, llvmI64Type);
      auto symbolConst = rewriter.create<stablehlo::ConstantOp>(
          op.getLoc(), i64TensorType,
          cast<ElementsAttr>(makeAttr(i64TensorType, symbolValue)));

      auto subconstraintPtr = rewriter.create<stablehlo::ConstantOp>(
          op.getLoc(), loweredConstraintType,
          cast<ElementsAttr>(makeAttr(loweredConstraintType, 0)));

      std::string wrapperFn = getOrCreateWrapper(getSubconstraintFn);

      if (!moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(wrapperFn)) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(moduleOp.getBody());

        auto funcType = LLVM::LLVMFunctionType::get(
            llvmVoidType, {llvmPtrType, llvmPtrType, llvmPtrType},
            /*isVarArg=*/false);
        auto func =
            rewriter.create<LLVM::LLVMFuncOp>(op.getLoc(), wrapperFn, funcType);

        rewriter.setInsertionPointToStart(func.addEntryBlock(rewriter));
        rewriter.create<LLVM::CallOp>(
            op.getLoc(), TypeRange{},
            SymbolRefAttr::get(ctx, getSubconstraintFn),
            ValueRange{func.getArgument(0), func.getArgument(1),
                       func.getArgument(2)});
        rewriter.create<LLVM::ReturnOp>(op.getLoc(), ValueRange{});
      }

      if (!moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(getSubconstraintFn)) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(moduleOp.getBody());
        auto funcType = LLVM::LLVMFunctionType::get(
            llvmVoidType, {llvmPtrType, llvmPtrType, llvmPtrType},
            /*isVarArg=*/false);
        rewriter.create<LLVM::LLVMFuncOp>(op.getLoc(), getSubconstraintFn,
                                          funcType, LLVM::Linkage::External);
      }

      SmallVector<Attribute> aliases;
      aliases.push_back(stablehlo::OutputOperandAliasAttr::get(
          ctx, std::vector<int64_t>{}, 2, std::vector<int64_t>{}));

      auto jitCall = rewriter.create<enzymexla::JITCallOp>(
          op.getLoc(), TypeRange{loweredConstraintType},
          mlir::FlatSymbolRefAttr::get(ctx, wrapperFn),
          ValueRange{constraint, symbolConst, subconstraintPtr},
          rewriter.getStringAttr(""),
          /*operand_layouts=*/nullptr,
          /*result_layouts=*/nullptr,
          /*arg_attrs=*/nullptr,
          /*res_attrs=*/nullptr,
          /*output_operand_aliases=*/rewriter.getArrayAttr(aliases),
          /*xla_side_effect_free=*/nullptr);

      rewriter.replaceOp(op, jitCall.getResults());

      return success();
    }

    return rewriter.notifyMatchFailure(op, "Unknown backend " + backend);
  }
};

// TODO: add passes

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
    typeConverter.addConversion([&](enzyme::ConstraintType t) {
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
    target.addIllegalOp<enzyme::AddWeightToTraceOp>();
    target.addIllegalOp<enzyme::AddRetvalToTraceOp>();
    target.addIllegalOp<enzyme::GetSampleFromConstraintOp>();
    target.addIllegalOp<enzyme::GetSubconstraintOp>();

    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp f) {
      return typeConverter.isSignatureLegal(f.getFunctionType());
    });
    target.addDynamicallyLegalOp<func::CallOp>([&](func::CallOp c) {
      return typeConverter.isSignatureLegal(c.getCalleeType());
    });
    target.addDynamicallyLegalOp<func::ReturnOp>(
        [&](func::ReturnOp r) { return typeConverter.isLegal(r); });
    target.addDynamicallyLegalOp<UnrealizedConversionCastOp>(
        [&](UnrealizedConversionCastOp c) {
          return typeConverter.isLegal(c.getOperandTypes()) &&
                 typeConverter.isLegal(c.getResultTypes());
        });

    RewritePatternSet patterns(context);

    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(
        patterns, typeConverter);
    populateCallOpTypeConversionPattern(patterns, typeConverter);
    populateReturnOpTypeConversionPattern(patterns, typeConverter);

    patterns.add<
        InitTraceOpConversion, AddSampleToTraceOpConversion,
        AddSubtraceOpConversion, AddWeightToTraceOpConversion,
        AddRetvalToTraceOpConversion, GetSampleFromConstraintOpConversion,
        GetSubconstraintOpConversion, UnrealizedConversionCastOpConversion>(
        backend, typeConverter, context);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
