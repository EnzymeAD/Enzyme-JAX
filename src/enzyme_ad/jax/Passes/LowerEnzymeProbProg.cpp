#include "Enzyme/MLIR/Dialect/Dialect.h"
#include "Enzyme/MLIR/Dialect/Ops.h"
#include "mhlo/IR/hlo_ops.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"
#include "src/enzyme_ad/jax/Dialect/Dialect.h"
#include "src/enzyme_ad/jax/Dialect/Ops.h"
#include "src/enzyme_ad/jax/Passes/Passes.h"
#include "src/enzyme_ad/jax/Utils.h"
#include "stablehlo/dialect/ChloOps.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ErrorHandling.h"
#include <cmath>
#include <cstdint>

#define DEBUG_TYPE "lower-enzyme-probprog"

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_LOWERPROBPROGTOSTABLEHLOPASS
#define GEN_PASS_DEF_LOWERPROBPROGTRACEOPSPASS
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

struct SelectTraceOpConversion
    : public OpConversionPattern<enzyme::SelectTraceOp> {
  using OpConversionPattern::OpConversionPattern;

  std::string backend;
  SelectTraceOpConversion(std::string backend, TypeConverter &typeConverter,
                          MLIRContext *context, PatternBenefit benefit = 1)
      : OpConversionPattern(typeConverter, context, benefit), backend(backend) {
  }

  LogicalResult
  matchAndRewrite(enzyme::SelectTraceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto newOp = rewriter.create<stablehlo::SelectOp>(
        op.getLoc(), adaptor.getTrueValue().getType(), adaptor.getCondition(),
        adaptor.getTrueValue(), adaptor.getFalseValue());

    rewriter.replaceOp(op, newOp.getResult());

    return success();
  }
};

struct DumpOpConversion : public OpConversionPattern<enzyme::DumpOp> {
  using OpConversionPattern::OpConversionPattern;

  std::string backend;
  DumpOpConversion(std::string backend, TypeConverter &typeConverter,
                   MLIRContext *context, PatternBenefit benefit = 1)
      : OpConversionPattern(typeConverter, context, benefit), backend(backend) {
  }

  LogicalResult
  matchAndRewrite(enzyme::DumpOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto ctx = op->getContext();
    Value value = adaptor.getValue();
    auto valueType = cast<RankedTensorType>(value.getType());

    if (backend == "cpu") {
      auto moduleOp = op->getParentOfType<ModuleOp>();

      auto llvmPtrType = LLVM::LLVMPointerType::get(ctx);
      auto llvmVoidType = LLVM::LLVMVoidType::get(ctx);
      auto llvmI64Type = IntegerType::get(ctx, 64);

      std::string dumpFn = "enzyme_probprog_dump";
      SmallVector<Type> originalTypes = {valueType};
      std::string wrapperFn = getOrCreateWrapper(dumpFn, originalTypes);

      auto shape = valueType.getShape();
      size_t ndims = shape.size();
      size_t width = valueType.getElementType().getIntOrFloatBitWidth();

      if (!moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(wrapperFn)) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(moduleOp.getBody());

        auto funcType = LLVM::LLVMFunctionType::get(
            llvmVoidType,
            {llvmPtrType, llvmPtrType, llvmPtrType, llvmPtrType, llvmPtrType},
            /*isVarArg=*/false);
        auto func =
            rewriter.create<LLVM::LLVMFuncOp>(op.getLoc(), wrapperFn, funcType);

        rewriter.setInsertionPointToStart(func.addEntryBlock(rewriter));

        auto oneConst = rewriter.create<LLVM::ConstantOp>(
            op.getLoc(), llvmI64Type, rewriter.getIntegerAttr(llvmI64Type, 1));

        auto ndimConst = rewriter.create<LLVM::ConstantOp>(
            op.getLoc(), llvmI64Type,
            rewriter.getIntegerAttr(llvmI64Type, ndims));
        auto ndimsAlloca = rewriter.create<LLVM::AllocaOp>(
            op.getLoc(), llvmPtrType, llvmI64Type, oneConst);
        rewriter.create<LLVM::StoreOp>(op.getLoc(), ndimConst, ndimsAlloca);

        auto widthConst = rewriter.create<LLVM::ConstantOp>(
            op.getLoc(), llvmI64Type,
            rewriter.getIntegerAttr(llvmI64Type, width));
        auto widthAlloca = rewriter.create<LLVM::AllocaOp>(
            op.getLoc(), llvmPtrType, llvmI64Type, oneConst);
        rewriter.create<LLVM::StoreOp>(op.getLoc(), widthConst, widthAlloca);

        Value shapeArrAlloca;
        if (ndims > 0) {
          auto shapeSizeConst = rewriter.create<LLVM::ConstantOp>(
              op.getLoc(), llvmI64Type,
              rewriter.getIntegerAttr(llvmI64Type, ndims));
          shapeArrAlloca = rewriter.create<LLVM::AllocaOp>(
              op.getLoc(), llvmPtrType, llvmI64Type, shapeSizeConst);

          for (size_t i = 0; i < ndims; ++i) {
            auto dimConst = rewriter.create<LLVM::ConstantOp>(
                op.getLoc(), llvmI64Type,
                rewriter.getIntegerAttr(llvmI64Type, shape[i]));
            auto dimGEP = rewriter.create<LLVM::GEPOp>(
                op.getLoc(), llvmPtrType, llvmI64Type, shapeArrAlloca,
                ValueRange{rewriter.create<LLVM::ConstantOp>(
                    op.getLoc(), llvmI64Type,
                    rewriter.getIntegerAttr(llvmI64Type, i))});
            rewriter.create<LLVM::StoreOp>(op.getLoc(), dimConst, dimGEP);
          }
        } else {
          shapeArrAlloca =
              rewriter.create<LLVM::ZeroOp>(op.getLoc(), llvmPtrType);
        }

        rewriter.create<LLVM::CallOp>(
            op.getLoc(), TypeRange{}, SymbolRefAttr::get(ctx, dumpFn),
            ValueRange{func.getArgument(0), func.getArgument(1), ndimsAlloca,
                       shapeArrAlloca, widthAlloca});

        rewriter.create<LLVM::ReturnOp>(op.getLoc(), ValueRange{});
      }

      if (!moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(dumpFn)) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(moduleOp.getBody());
        auto funcType = LLVM::LLVMFunctionType::get(
            llvmVoidType,
            {llvmPtrType, llvmPtrType, llvmPtrType, llvmPtrType, llvmPtrType},
            /*isVarArg=*/false);
        rewriter.create<LLVM::LLVMFuncOp>(op.getLoc(), dumpFn, funcType,
                                          LLVM::Linkage::External);
      }

      auto labelStr = op.getLabel().str();
      auto i8Type = IntegerType::get(ctx, 8);
      SmallVector<APInt> labelChars;
      for (char c : labelStr) {
        labelChars.push_back(APInt(8, static_cast<uint8_t>(c)));
      }
      labelChars.push_back(APInt(8, 0));

      auto labelArrayType = RankedTensorType::get(
          {static_cast<int64_t>(labelChars.size())}, i8Type);
      auto labelConst = rewriter.create<stablehlo::ConstantOp>(
          op.getLoc(), labelArrayType,
          DenseIntElementsAttr::get(labelArrayType, labelChars));

      auto i64TensorType = RankedTensorType::get({}, llvmI64Type);
      auto ndimsConst = rewriter.create<stablehlo::ConstantOp>(
          op.getLoc(), i64TensorType,
          cast<ElementsAttr>(
              makeAttr(i64TensorType, static_cast<int64_t>(ndims))));
      auto widthConst = rewriter.create<stablehlo::ConstantOp>(
          op.getLoc(), i64TensorType,
          cast<ElementsAttr>(
              makeAttr(i64TensorType, static_cast<int64_t>(width))));

      Value shapeConst;
      if (ndims > 0) {
        auto shapeArrayType =
            RankedTensorType::get({static_cast<int64_t>(ndims)}, llvmI64Type);
        SmallVector<APInt> shapeAPInt;
        for (auto dim : shape) {
          shapeAPInt.push_back(APInt(64, dim));
        }
        shapeConst = rewriter.create<stablehlo::ConstantOp>(
            op.getLoc(), shapeArrayType,
            DenseIntElementsAttr::get(shapeArrayType, shapeAPInt));
      } else {
        auto shapeArrayType = RankedTensorType::get({0}, llvmI64Type);
        shapeConst = rewriter.create<stablehlo::ConstantOp>(
            op.getLoc(), shapeArrayType,
            DenseIntElementsAttr::get(shapeArrayType, ArrayRef<APInt>{}));
      }

      SmallVector<Attribute> aliases;
      aliases.push_back(stablehlo::OutputOperandAliasAttr::get(
          ctx, std::vector<int64_t>{}, 0, std::vector<int64_t>{}));

      auto jitCall = rewriter.create<enzymexla::JITCallOp>(
          op.getLoc(), TypeRange{valueType},
          mlir::FlatSymbolRefAttr::get(ctx, wrapperFn),
          ValueRange{value, labelConst, ndimsConst, shapeConst, widthConst},
          rewriter.getStringAttr(""),
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

struct GetSampleFromTraceOpConversion
    : public OpConversionPattern<enzyme::GetSampleFromTraceOp> {
  using OpConversionPattern::OpConversionPattern;

  std::string backend;
  GetSampleFromTraceOpConversion(std::string backend,
                                 TypeConverter &typeConverter,
                                 MLIRContext *context,
                                 PatternBenefit benefit = 1)
      : OpConversionPattern(typeConverter, context, benefit), backend(backend) {
  }

  LogicalResult
  matchAndRewrite(enzyme::GetSampleFromTraceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto ctx = op->getContext();

    Value trace = adaptor.getTrace();
    auto outputs = op.getSample();

    auto symbolWrappedAttr = op.getSymbolAttr();
    if (!symbolWrappedAttr) {
      return rewriter.notifyMatchFailure(op, "Missing symbol attribute");
    }

    uint64_t symbolValue = symbolWrappedAttr.getPtr();

    size_t numOutputs = outputs.size();
    if (numOutputs == 0)
      return rewriter.notifyMatchFailure(op,
                                         "GetSampleFromTraceOp has no outputs");

    if (backend == "cpu") {
      auto moduleOp = op->getParentOfType<ModuleOp>();

      auto llvmPtrType = LLVM::LLVMPointerType::get(ctx);
      auto llvmVoidType = LLVM::LLVMVoidType::get(ctx);
      auto llvmI64Type = IntegerType::get(ctx, 64);

      std::string getSampleFn = "enzyme_probprog_get_sample_from_trace";

      auto i64TensorType = RankedTensorType::get({}, llvmI64Type);
      auto symbolConst = rewriter.create<stablehlo::ConstantOp>(
          op.getLoc(), i64TensorType,
          cast<ElementsAttr>(makeAttr(i64TensorType, symbolValue)));

      SmallVector<Type> llvmArgTypes; // (trace, symbol, out_ptrs...)
      llvmArgTypes.push_back(llvmPtrType);
      llvmArgTypes.push_back(llvmPtrType);
      llvmArgTypes.append(numOutputs, llvmPtrType);

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
      jitOperands.push_back(trace);
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

    return rewriter.notifyMatchFailure(op, "Unknown backend " + backend);
  }
};

struct CholeskySolveOpConversion
    : public OpConversionPattern<enzyme::CholeskySolveOp> {
  using OpConversionPattern::OpConversionPattern;

  std::string backend;
  CholeskySolveOpConversion(std::string backend, TypeConverter &typeConverter,
                            MLIRContext *context, PatternBenefit benefit = 1)
      : OpConversionPattern(typeConverter, context, benefit), backend(backend) {
  }

  LogicalResult
  matchAndRewrite(enzyme::CholeskySolveOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto lhs = adaptor.getLhs();
    auto rhs = adaptor.getRhs();
    auto resultType = cast<RankedTensorType>(op.getResult().getType());
    auto lhsType = cast<RankedTensorType>(lhs.getType());
    auto rhsType = cast<RankedTensorType>(rhs.getType());

    // StableHLO triangular_solve requires both operands to have the same rank
    Value rhsReshaped = rhs;
    Type intermediateResultType = resultType;

    if (rhsType.getRank() == 1) {
      auto shape = rhsType.getShape();
      auto reshapedType =
          RankedTensorType::get({shape[0], 1}, rhsType.getElementType());
      rhsReshaped =
          rewriter.create<stablehlo::ReshapeOp>(op.getLoc(), reshapedType, rhs);
      intermediateResultType = reshapedType;
    }

    // Cholesky decomposition: A = LL^T, A is symmetric positive definite
    // Then solve: Ly = b, L^T x = y, where L is lower triangular
    auto choleskyOp = rewriter.create<stablehlo::CholeskyOp>(
        op.getLoc(), lhsType, lhs,
        /*lower=*/rewriter.getBoolAttr(true));
    Value L = choleskyOp.getResult();

    // Forward substitution: solve Ly = b, where L is lower triangular
    auto forwardSolve = rewriter.create<stablehlo::TriangularSolveOp>(
        op.getLoc(), intermediateResultType,
        /*a=*/L,
        /*b=*/rhsReshaped,
        /*left_side=*/true,
        /*lower=*/true,
        /*unit_diagonal=*/false,
        /*transpose_a=*/stablehlo::Transpose::NO_TRANSPOSE);

    // Backward substitution: solve L^T x = y, where L^T is upper triangular
    auto backwardSolve = rewriter.create<stablehlo::TriangularSolveOp>(
        op.getLoc(), intermediateResultType,
        /*a=*/L,
        /*b=*/forwardSolve.getResult(),
        /*left_side=*/true,
        /*lower=*/true,
        /*unit_diagonal=*/false,
        /*transpose_a=*/stablehlo::Transpose::TRANSPOSE);

    // If we reshaped to column matrix, reshape back to vector
    Value result = backwardSolve.getResult();
    if (rhsType.getRank() == 1) {
      result = rewriter.create<stablehlo::ReshapeOp>(op.getLoc(), resultType,
                                                     result);
    }

    rewriter.replaceOp(op, result);
    return success();
  }
};

struct DotOpConversion : public OpConversionPattern<enzyme::DotOp> {
  using OpConversionPattern::OpConversionPattern;

  std::string backend;
  DotOpConversion(std::string backend, TypeConverter &typeConverter,
                  MLIRContext *context, PatternBenefit benefit = 1)
      : OpConversionPattern(typeConverter, context, benefit), backend(backend) {
  }

  LogicalResult
  matchAndRewrite(enzyme::DotOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto lhs = adaptor.getLhs();
    auto rhs = adaptor.getRhs();
    auto resultType = cast<RankedTensorType>(op.getResult().getType());
    auto lhsType = cast<RankedTensorType>(lhs.getType());

    auto dotDimensionNumbers = stablehlo::DotDimensionNumbersAttr::get(
        rewriter.getContext(),
        /*lhs_batching_dimensions=*/{},
        /*rhs_batching_dimensions=*/{},
        /*lhs_contracting_dimensions=*/{0},
        /*rhs_contracting_dimensions=*/{0});

    auto dotOp = rewriter.create<stablehlo::DotGeneralOp>(
        op.getLoc(), resultType, lhs, rhs, dotDimensionNumbers,
        /*precision_config=*/ArrayAttr(),
        /*algorithm=*/stablehlo::DotAlgorithmAttr());

    rewriter.replaceOp(op, dotOp.getResult());
    return success();
  }
};

struct RandomOpConversion : public OpConversionPattern<enzyme::RandomOp> {
  using OpConversionPattern::OpConversionPattern;

  std::string backend;
  RandomOpConversion(std::string backend, TypeConverter &typeConverter,
                     MLIRContext *context, PatternBenefit benefit = 1)
      : OpConversionPattern(typeConverter, context, benefit), backend(backend) {
  }

  LogicalResult
  matchAndRewrite(enzyme::RandomOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto distribution = op.getRngDistribution();
    auto resultType = op.getResult().getType();
    auto rankedType = dyn_cast<RankedTensorType>(resultType);
    if (!rankedType) {
      return rewriter.notifyMatchFailure(op, "Result must be a ranked tensor");
    }

    auto elemType = rankedType.getElementType();
    assert(isa<FloatType>(elemType));
    auto rngStateType = adaptor.getRngState().getType();
    auto rngStateTensorType = dyn_cast<RankedTensorType>(rngStateType);
    if (!rngStateTensorType) {
      return rewriter.notifyMatchFailure(op, "RNG state must be a tensor");
    }

    unsigned nbits = elemType.getIntOrFloatBitWidth();
    Type uintType =
        IntegerType::get(rewriter.getContext(), nbits, IntegerType::Unsigned);
    if (!uintType)
      return rewriter.notifyMatchFailure(
          op, "Failed to create unsigned integer type");

    auto uintResultType =
        RankedTensorType::get(rankedType.getShape(), uintType);
    auto rngAlgorithm = mlir::stablehlo::RngAlgorithmAttr::get(
        rewriter.getContext(), mlir::stablehlo::RngAlgorithm::DEFAULT);
    auto rngBitGenOp = rewriter.create<stablehlo::RngBitGeneratorOp>(
        op.getLoc(),
        /*output_state=*/rngStateTensorType,
        /*output=*/uintResultType,
        /*rng_algorithm=*/rngAlgorithm,
        /*initial_state=*/adaptor.getRngState());

    Value outputState = rngBitGenOp.getOutputState();
    Value randomBits = rngBitGenOp.getOutput();
    Value result;

    if (distribution == enzyme::RngDistribution::UNIFORM) {
      unsigned mantissaBits;
      if (nbits == 16)
        mantissaBits = 10; // TODO bfloat16
      else if (nbits == 32)
        mantissaBits = 23;
      else if (nbits == 64)
        mantissaBits = 52;
      else
        return rewriter.notifyMatchFailure(op, "Unsupported float type");

      auto shiftAmount = rewriter.create<stablehlo::ConstantOp>(
          op.getLoc(), uintResultType,
          DenseElementsAttr::get(
              uintResultType,
              rewriter.getIntegerAttr(uintType, nbits - mantissaBits)));
      auto shiftedBits = rewriter.create<stablehlo::ShiftRightLogicalOp>(
          op.getLoc(), uintResultType, randomBits, shiftAmount);

      uint64_t onePattern;
      if (nbits == 16)
        onePattern = 0x3C00; // TODO bfloat16
      else if (nbits == 32)
        onePattern = 0x3F800000;
      else if (nbits == 64)
        onePattern = 0x3FF0000000000000ULL;
      else
        return rewriter.notifyMatchFailure(op,
                                           "Unsupported float type: $(nbits)");

      auto onePatternConst = rewriter.create<stablehlo::ConstantOp>(
          op.getLoc(), uintResultType,
          DenseElementsAttr::get(
              uintResultType, rewriter.getIntegerAttr(uintType, onePattern)));
      auto floatBits = rewriter.create<stablehlo::OrOp>(
          op.getLoc(), uintResultType, shiftedBits, onePatternConst);
      auto floatValue = rewriter.create<stablehlo::BitcastConvertOp>(
          op.getLoc(), rankedType, floatBits);
      auto oneConst = rewriter.create<stablehlo::ConstantOp>(
          op.getLoc(), rankedType,
          DenseElementsAttr::get(rankedType,
                                 rewriter.getFloatAttr(elemType, 1.0)));
      result = rewriter.create<stablehlo::SubtractOp>(op.getLoc(), rankedType,
                                                      floatValue, oneConst);
    } else if (distribution == enzyme::RngDistribution::NORMAL) {
      unsigned mantissaBits;
      if (nbits == 16)
        mantissaBits = 10; // TODO bfloat16
      else if (nbits == 32)
        mantissaBits = 23;
      else if (nbits == 64)
        mantissaBits = 52;
      else
        return rewriter.notifyMatchFailure(op,
                                           "Unsupported float type: $(nbits)");

      auto shiftAmount = rewriter.create<stablehlo::ConstantOp>(
          op.getLoc(), uintResultType,
          DenseElementsAttr::get(
              uintResultType,
              rewriter.getIntegerAttr(uintType, nbits - mantissaBits)));
      auto shiftedBits = rewriter.create<stablehlo::ShiftRightLogicalOp>(
          op.getLoc(), uintResultType, randomBits, shiftAmount);

      uint64_t onePattern;
      if (nbits == 16)
        onePattern = 0x3C00;
      else if (nbits == 32)
        onePattern = 0x3F800000;
      else if (nbits == 64)
        onePattern = 0x3FF0000000000000ULL;
      else
        return rewriter.notifyMatchFailure(op,
                                           "Unsupported float type: $(nbits)");

      auto onePatternConst = rewriter.create<stablehlo::ConstantOp>(
          op.getLoc(), uintResultType,
          DenseElementsAttr::get(
              uintResultType, rewriter.getIntegerAttr(uintType, onePattern)));
      auto floatBits = rewriter.create<stablehlo::OrOp>(
          op.getLoc(), uintResultType, shiftedBits, onePatternConst);

      Value randUniform = rewriter
                              .create<stablehlo::BitcastConvertOp>(
                                  op.getLoc(), rankedType, floatBits)
                              .getResult();
      auto oneConst = rewriter.create<stablehlo::ConstantOp>(
          op.getLoc(), rankedType,
          DenseElementsAttr::get(rankedType,
                                 rewriter.getFloatAttr(elemType, 1.0)));
      randUniform = rewriter
                        .create<stablehlo::SubtractOp>(op.getLoc(), rankedType,
                                                       randUniform, oneConst)
                        .getResult();
      auto twoConst = rewriter.create<stablehlo::ConstantOp>(
          op.getLoc(), rankedType,
          DenseElementsAttr::get(rankedType,
                                 rewriter.getFloatAttr(elemType, 2.0)));
      Value scaledUniform =
          rewriter
              .create<stablehlo::MulOp>(op.getLoc(), rankedType, randUniform,
                                        twoConst)
              .getResult();
      scaledUniform = rewriter
                          .create<stablehlo::SubtractOp>(
                              op.getLoc(), rankedType, scaledUniform, oneConst)
                          .getResult();
      auto probit = rewriter.create<chlo::ErfInvOp>(op.getLoc(), rankedType,
                                                    scaledUniform);
      double sqrt2 = std::sqrt(2.0);
      auto sqrt2Const = rewriter.create<stablehlo::ConstantOp>(
          op.getLoc(), rankedType,
          DenseElementsAttr::get(rankedType,
                                 rewriter.getFloatAttr(elemType, sqrt2)));
      result = rewriter
                   .create<stablehlo::MulOp>(op.getLoc(), rankedType, probit,
                                             sqrt2Const)
                   .getResult();
    } else {
      return rewriter.notifyMatchFailure(op, "Unknown RNG distribution");
    }

    rewriter.replaceOp(op, {outputState, result});
    return success();
  }
};

struct GetSubtraceOpConversion
    : public OpConversionPattern<enzyme::GetSubtraceOp> {
  using OpConversionPattern::OpConversionPattern;

  std::string backend;
  GetSubtraceOpConversion(std::string backend, TypeConverter &typeConverter,
                          MLIRContext *context, PatternBenefit benefit = 1)
      : OpConversionPattern(typeConverter, context, benefit), backend(backend) {
  }

  LogicalResult
  matchAndRewrite(enzyme::GetSubtraceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto ctx = op->getContext();

    Value trace = adaptor.getTrace();

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
      auto loweredTraceType = RankedTensorType::get(
          {}, IntegerType::get(ctx, 64, IntegerType::Unsigned));

      std::string getSubtraceFn = "enzyme_probprog_get_subtrace";

      auto i64TensorType = RankedTensorType::get({}, llvmI64Type);
      auto symbolConst = rewriter.create<stablehlo::ConstantOp>(
          op.getLoc(), i64TensorType,
          cast<ElementsAttr>(makeAttr(i64TensorType, symbolValue)));

      auto subtracePtr = rewriter.create<stablehlo::ConstantOp>(
          op.getLoc(), loweredTraceType,
          cast<ElementsAttr>(makeAttr(loweredTraceType, 0)));

      std::string wrapperFn = getOrCreateWrapper(getSubtraceFn);

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
            op.getLoc(), TypeRange{}, SymbolRefAttr::get(ctx, getSubtraceFn),
            ValueRange{func.getArgument(0), func.getArgument(1),
                       func.getArgument(2)});
        rewriter.create<LLVM::ReturnOp>(op.getLoc(), ValueRange{});
      }

      if (!moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(getSubtraceFn)) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(moduleOp.getBody());
        auto funcType = LLVM::LLVMFunctionType::get(
            llvmVoidType, {llvmPtrType, llvmPtrType, llvmPtrType},
            /*isVarArg=*/false);
        rewriter.create<LLVM::LLVMFuncOp>(op.getLoc(), getSubtraceFn, funcType,
                                          LLVM::Linkage::External);
      }

      SmallVector<Attribute> aliases;
      aliases.push_back(stablehlo::OutputOperandAliasAttr::get(
          ctx, std::vector<int64_t>{}, 2, std::vector<int64_t>{}));

      auto jitCall = rewriter.create<enzymexla::JITCallOp>(
          op.getLoc(), TypeRange{loweredTraceType},
          mlir::FlatSymbolRefAttr::get(ctx, wrapperFn),
          ValueRange{trace, symbolConst, subtracePtr},
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

struct GetWeightFromTraceOpConversion
    : public OpConversionPattern<enzyme::GetWeightFromTraceOp> {
  using OpConversionPattern::OpConversionPattern;

  std::string backend;
  GetWeightFromTraceOpConversion(std::string backend,
                                 TypeConverter &typeConverter,
                                 MLIRContext *context,
                                 PatternBenefit benefit = 1)
      : OpConversionPattern(typeConverter, context, benefit), backend(backend) {
  }

  LogicalResult
  matchAndRewrite(enzyme::GetWeightFromTraceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto ctx = op->getContext();

    Value trace = adaptor.getTrace();
    auto weightType = op.getWeight().getType();

    if (backend == "cpu") {
      auto moduleOp = op->getParentOfType<ModuleOp>();

      auto llvmPtrType = LLVM::LLVMPointerType::get(ctx);
      auto llvmVoidType = LLVM::LLVMVoidType::get(ctx);

      std::string getWeightFn = "enzyme_probprog_get_weight_from_trace";
      SmallVector<Type> originalTypes = {weightType};
      std::string wrapperFn = getOrCreateWrapper(getWeightFn, originalTypes);
      auto weightConst = rewriter.create<stablehlo::ConstantOp>(
          op.getLoc(), weightType, cast<ElementsAttr>(makeAttr(weightType, 0)));

      if (!moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(wrapperFn)) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(moduleOp.getBody());

        auto funcType = LLVM::LLVMFunctionType::get(
            llvmVoidType, {llvmPtrType, llvmPtrType}, /*isVarArg=*/false);
        auto func =
            rewriter.create<LLVM::LLVMFuncOp>(op.getLoc(), wrapperFn, funcType);

        rewriter.setInsertionPointToStart(func.addEntryBlock(rewriter));
        rewriter.create<LLVM::CallOp>(op.getLoc(), TypeRange{},
                                      SymbolRefAttr::get(ctx, getWeightFn),
                                      func.getArguments());
        rewriter.create<LLVM::ReturnOp>(op.getLoc(), ValueRange{});
      }

      if (!moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(getWeightFn)) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(moduleOp.getBody());
        auto funcType = LLVM::LLVMFunctionType::get(
            llvmVoidType, {llvmPtrType, llvmPtrType}, /*isVarArg=*/false);
        rewriter.create<LLVM::LLVMFuncOp>(op.getLoc(), getWeightFn, funcType,
                                          LLVM::Linkage::External);
      }

      SmallVector<Attribute> aliases;
      aliases.push_back(stablehlo::OutputOperandAliasAttr::get(
          ctx, std::vector<int64_t>{}, 1, std::vector<int64_t>{}));

      auto jitCall = rewriter.create<enzymexla::JITCallOp>(
          op.getLoc(), TypeRange{weightType},
          mlir::FlatSymbolRefAttr::get(ctx, wrapperFn),
          ValueRange{trace, weightConst}, rewriter.getStringAttr(""),
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
    target.addIllegalOp<enzyme::GetSampleFromTraceOp>();
    target.addIllegalOp<enzyme::GetSubtraceOp>();
    target.addIllegalOp<enzyme::GetWeightFromTraceOp>();
    target.addIllegalOp<enzyme::RandomOp>();

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
    target.addDynamicallyLegalOp<arith::SelectOp>(
        [&](arith::SelectOp s) { return typeConverter.isLegal(s.getType()); });
    target.addDynamicallyLegalOp<tensor::ExtractOp>(
        [&](tensor::ExtractOp extract) {
          if (!extract->hasOneUse())
            return true;
          auto selectOp = dyn_cast<arith::SelectOp>(*extract->user_begin());
          if (!selectOp)
            return true;
          return typeConverter.isLegal(selectOp.getType());
        });

    RewritePatternSet patterns(context);

    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(
        patterns, typeConverter);
    populateCallOpTypeConversionPattern(patterns, typeConverter);
    populateReturnOpTypeConversionPattern(patterns, typeConverter);

    patterns
        .add<InitTraceOpConversion, AddSampleToTraceOpConversion,
             AddSubtraceOpConversion, AddWeightToTraceOpConversion,
             AddRetvalToTraceOpConversion, GetSampleFromConstraintOpConversion,
             GetSubconstraintOpConversion, GetSampleFromTraceOpConversion,
             GetSubtraceOpConversion, GetWeightFromTraceOpConversion,
             RandomOpConversion, ArithSelectOpConversion,
             TensorExtractOpElimination, UnrealizedConversionCastOpConversion>(
            backend, typeConverter, context);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
