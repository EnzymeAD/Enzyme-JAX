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
    else if (auto intType = dyn_cast<IntegerType>(elemType)) {
      if (intType.isUnsigned())
        sig += "u";
      else if (intType.isSigned())
        sig += "s";
      sig += "i" + std::to_string(intType.getWidth());
    } else if (elemType.isInteger(1))
      sig += "i1";
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

static Value conditionalDump(OpBuilder &builder, Location loc, Value value,
                             StringRef label, bool debugDump) {
  if (debugDump) {
    return enzyme::DumpOp::create(builder, loc, value.getType(), value,
                                  builder.getStringAttr(label))
        .getOutput();
  }
  return value;
}

// Reference (_make_rotate_left):
// https://github.com/jax-ml/jax/blob/3aa8a6b0d4de5e554f45db638b0f3056e4c520f1/jax/_src/prng.py#L832
static Value createRotateLeft(OpBuilder &builder, Location loc, Value x,
                              Value distance) {
  auto xType = cast<RankedTensorType>(x.getType());
  auto elemType = cast<IntegerType>(xType.getElementType());
  unsigned nbits = elemType.getWidth();

  auto nbitsConst = stablehlo::ConstantOp::create(
      builder, loc, xType,
      DenseElementsAttr::get(xType, builder.getIntegerAttr(elemType, nbits)));
  auto nbitsMinusD =
      stablehlo::SubtractOp::create(builder, loc, xType, nbitsConst, distance);
  auto shiftedLeft =
      stablehlo::ShiftLeftOp::create(builder, loc, xType, x, distance);
  auto shiftedRight = stablehlo::ShiftRightLogicalOp::create(
      builder, loc, xType, x, nbitsMinusD);
  auto result =
      stablehlo::OrOp::create(builder, loc, xType, shiftedLeft, shiftedRight);

  return result;
}

// Reference (_apply_round):
// https://github.com/jax-ml/jax/blob/3aa8a6b0d4de5e554f45db638b0f3056e4c520f1/jax/_src/prng.py#L863
static std::pair<Value, Value> applyRound(OpBuilder &builder, Location loc,
                                          Value v0, Value v1,
                                          uint32_t rotation) {
  auto vType = cast<RankedTensorType>(v0.getType());
  auto elemType = vType.getElementType();

  auto newV0 = stablehlo::AddOp::create(builder, loc, vType, v0, v1);
  auto rotConst = stablehlo::ConstantOp::create(
      builder, loc, vType,
      DenseElementsAttr::get(vType,
                             builder.getIntegerAttr(elemType, rotation)));
  auto rotated = createRotateLeft(builder, loc, v1, rotConst);
  auto newV1 = stablehlo::XorOp::create(builder, loc, vType, newV0, rotated);

  return {newV0, newV1};
}

// Reference (_threefry2x32):
// https://github.com/jax-ml/jax/blob/3aa8a6b0d4de5e554f45db638b0f3056e4c520f1/jax/_src/prng.py#L883
static std::pair<Value, Value> threefry2x32Hash(OpBuilder &builder,
                                                Location loc, Value key1,
                                                Value key2, Value x1,
                                                Value x2) {
  auto xType = cast<RankedTensorType>(x1.getType());
  auto elemType = xType.getElementType();

  auto parityConst = stablehlo::ConstantOp::create(
      builder, loc, xType,
      DenseElementsAttr::get(xType,
                             builder.getIntegerAttr(elemType, 0x1BD11BDA)));

  // [key1, key2, key1 ^ key2 ^ 0x1BD11BDA]
  auto ks2 = stablehlo::XorOp::create(
      builder, loc, xType,
      stablehlo::XorOp::create(builder, loc, xType, key1, key2), parityConst);

  const uint32_t rotations[2][4] = {{13, 15, 26, 6}, {17, 29, 16, 24}};

  Value v0 = stablehlo::AddOp::create(builder, loc, xType, x1, key1);
  Value v1 = stablehlo::AddOp::create(builder, loc, xType, x2, key2);

  // 1st iteration in rotations[0], then v0 += ks[1], v1 += ks[2] + 1
  for (uint32_t rot : rotations[0]) {
    std::tie(v0, v1) = applyRound(builder, loc, v0, v1, rot);
  }
  v0 = stablehlo::AddOp::create(builder, loc, xType, v0, key2);
  auto oneConst = stablehlo::ConstantOp::create(
      builder, loc, xType,
      DenseElementsAttr::get(xType, builder.getIntegerAttr(elemType, 1)));
  auto v1PlusKs2 = stablehlo::AddOp::create(builder, loc, xType, v1, ks2);
  v1 = stablehlo::AddOp::create(builder, loc, xType, v1PlusKs2, oneConst);

  // 2nd iteration in rotations[1], then v0 += ks[2], v1 += ks[0] + 2
  for (uint32_t rot : rotations[1]) {
    std::tie(v0, v1) = applyRound(builder, loc, v0, v1, rot);
  }
  v0 = stablehlo::AddOp::create(builder, loc, xType, v0, ks2);
  auto twoConst = stablehlo::ConstantOp::create(
      builder, loc, xType,
      DenseElementsAttr::get(xType, builder.getIntegerAttr(elemType, 2)));
  auto v1PlusKey1 = stablehlo::AddOp::create(builder, loc, xType, v1, key1);
  v1 = stablehlo::AddOp::create(builder, loc, xType, v1PlusKey1, twoConst);

  // 3rd iteration in rotations[0], then v0 += ks[0], v1 += ks[1] + 3
  for (uint32_t rot : rotations[0]) {
    std::tie(v0, v1) = applyRound(builder, loc, v0, v1, rot);
  }
  v0 = stablehlo::AddOp::create(builder, loc, xType, v0, key1);
  auto threeConst = stablehlo::ConstantOp::create(
      builder, loc, xType,
      DenseElementsAttr::get(xType, builder.getIntegerAttr(elemType, 3)));
  auto v1PlusKey2 = stablehlo::AddOp::create(builder, loc, xType, v1, key2);
  v1 = stablehlo::AddOp::create(builder, loc, xType, v1PlusKey2, threeConst);

  // 4th iteration in rotations[1], then v0 += ks[1], v1 += ks[2] + 4
  for (uint32_t rot : rotations[1]) {
    std::tie(v0, v1) = applyRound(builder, loc, v0, v1, rot);
  }
  v0 = stablehlo::AddOp::create(builder, loc, xType, v0, key2);
  auto fourConst = stablehlo::ConstantOp::create(
      builder, loc, xType,
      DenseElementsAttr::get(xType, builder.getIntegerAttr(elemType, 4)));
  auto v1PlusKs2_2 = stablehlo::AddOp::create(builder, loc, xType, v1, ks2);
  v1 = stablehlo::AddOp::create(builder, loc, xType, v1PlusKs2_2, fourConst);

  // 5th iteration in rotations[0], then v0 += ks[2], v1 += ks[0] + 5
  for (uint32_t rot : rotations[0]) {
    std::tie(v0, v1) = applyRound(builder, loc, v0, v1, rot);
  }
  v0 = stablehlo::AddOp::create(builder, loc, xType, v0, ks2);
  auto fiveConst = stablehlo::ConstantOp::create(
      builder, loc, xType,
      DenseElementsAttr::get(xType, builder.getIntegerAttr(elemType, 5)));
  auto v1PlusKey1_2 = stablehlo::AddOp::create(builder, loc, xType, v1, key1);
  v1 = stablehlo::AddOp::create(builder, loc, xType, v1PlusKey1_2, fiveConst);

  return {v0, v1};
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
      auto tracePtr = stablehlo::ConstantOp::create(
          rewriter, op.getLoc(), loweredTraceType,
          cast<ElementsAttr>(makeAttr(loweredTraceType, 0)));

      std::string initTraceFn = "enzyme_probprog_init_trace";
      auto funcType =
          LLVM::LLVMFunctionType::get(llvmVoidType, {llvmPtrType}, false);
      std::string wrapperFn = getOrCreateWrapper(initTraceFn);

      if (!moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(wrapperFn)) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(moduleOp.getBody());

        auto func = LLVM::LLVMFuncOp::create(rewriter, op.getLoc(), wrapperFn,
                                             funcType);
        rewriter.setInsertionPointToStart(func.addEntryBlock(rewriter));

        LLVM::CallOp::create(rewriter, op.getLoc(), TypeRange{},
                             SymbolRefAttr::get(ctx, initTraceFn),
                             ValueRange{func.getArgument(0)});

        LLVM::ReturnOp::create(rewriter, op.getLoc(), ValueRange{});
      }

      if (!moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(initTraceFn)) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(moduleOp.getBody());

        auto funcType =
            LLVM::LLVMFunctionType::get(llvmVoidType, {llvmPtrType}, false);

        LLVM::LLVMFuncOp::create(rewriter, op.getLoc(), initTraceFn, funcType,
                                 LLVM::Linkage::External);
      }

      SmallVector<Attribute> aliases;
      aliases.push_back(stablehlo::OutputOperandAliasAttr::get(
          ctx, std::vector<int64_t>{}, 0, std::vector<int64_t>{}));

      auto jitCall = enzymexla::JITCallOp::create(
          rewriter, op.getLoc(), TypeRange{loweredTraceType},
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

      auto symbolConst = stablehlo::ConstantOp::create(
          rewriter, op.getLoc(), i64TensorType,
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
        auto func = LLVM::LLVMFuncOp::create(rewriter, op.getLoc(), wrapperFn,
                                             funcType);

        rewriter.setInsertionPointToStart(func.addEntryBlock(rewriter));

        auto oneConst =
            LLVM::ConstantOp::create(rewriter, op.getLoc(), llvmI64Type,
                                     rewriter.getIntegerAttr(llvmI64Type, 1));
        auto numSamplesConst = LLVM::ConstantOp::create(
            rewriter, op.getLoc(), llvmI64Type,
            rewriter.getIntegerAttr(llvmI64Type, numSamples));
        auto numSamplesAlloca = LLVM::AllocaOp::create(
            rewriter, op.getLoc(), llvmPtrType, llvmI64Type, oneConst);
        LLVM::StoreOp::create(rewriter, op.getLoc(), numSamplesConst,
                              numSamplesAlloca);

        // Metadata pointer arrays of size `numSamples`
        auto samplePtrArrayAlloca = LLVM::AllocaOp::create(
            rewriter, op.getLoc(), llvmPtrType, llvmPtrType, numSamplesConst);
        auto numDimsArrayAlloca = LLVM::AllocaOp::create(
            rewriter, op.getLoc(), llvmPtrType, llvmI64Type, numSamplesConst);
        auto shapePtrArrayAlloca = LLVM::AllocaOp::create(
            rewriter, op.getLoc(), llvmPtrType, llvmPtrType, numSamplesConst);
        auto dtypeWidthArrayAlloca = LLVM::AllocaOp::create(
            rewriter, op.getLoc(), llvmPtrType, llvmI64Type, numSamplesConst);

        for (size_t i = 0; i < numSamples; ++i) {
          auto sampleType = cast<RankedTensorType>(sample[i].getType());
          auto sampleShape = sampleType.getShape();
          size_t sampleNumDims = sampleShape.size();
          size_t sampleWidth =
              sampleType.getElementType().getIntOrFloatBitWidth();

          // 1. Store `sample` pointer in `samplePtrArrayAlloca` for each
          // sampled value.
          auto samplePtrGEP = LLVM::GEPOp::create(
              rewriter, op.getLoc(), llvmPtrType, llvmI64Type,
              samplePtrArrayAlloca,
              ValueRange{LLVM::ConstantOp::create(
                  rewriter, op.getLoc(), llvmI64Type,
                  rewriter.getIntegerAttr(llvmI64Type, i))});
          LLVM::StoreOp::create(rewriter, op.getLoc(), func.getArgument(2 + i),
                                samplePtrGEP);

          // 2. Store `numDims` in `numDimsArrayAlloca` for each sampled
          // value.
          auto numDimsConst = LLVM::ConstantOp::create(
              rewriter, op.getLoc(), llvmI64Type,
              rewriter.getIntegerAttr(llvmI64Type, sampleNumDims));
          auto numDimsGEP = LLVM::GEPOp::create(
              rewriter, op.getLoc(), llvmPtrType, llvmI64Type,
              numDimsArrayAlloca,
              ValueRange{LLVM::ConstantOp::create(
                  rewriter, op.getLoc(), llvmI64Type,
                  rewriter.getIntegerAttr(llvmI64Type, i))});
          LLVM::StoreOp::create(rewriter, op.getLoc(), numDimsConst,
                                numDimsGEP);

          // 3. Store `dtypeWidth` in `dtypeWidthArrayAlloca` for each sampled
          // value.
          auto widthConst = LLVM::ConstantOp::create(
              rewriter, op.getLoc(), llvmI64Type,
              rewriter.getIntegerAttr(llvmI64Type, sampleWidth));
          auto widthGEP = LLVM::GEPOp::create(
              rewriter, op.getLoc(), llvmPtrType, llvmI64Type,
              dtypeWidthArrayAlloca,
              ValueRange{LLVM::ConstantOp::create(
                  rewriter, op.getLoc(), llvmI64Type,
                  rewriter.getIntegerAttr(llvmI64Type, i))});
          LLVM::StoreOp::create(rewriter, op.getLoc(), widthConst, widthGEP);

          // 4a. Allocate and fill shape array for this sample
          auto shapeSizeConst = LLVM::ConstantOp::create(
              rewriter, op.getLoc(), llvmI64Type,
              rewriter.getIntegerAttr(llvmI64Type, sampleNumDims));
          auto shapeArrAlloca = LLVM::AllocaOp::create(
              rewriter, op.getLoc(), llvmPtrType, llvmI64Type, shapeSizeConst);

          for (size_t j = 0; j < sampleNumDims; ++j) {
            auto dimConst = LLVM::ConstantOp::create(
                rewriter, op.getLoc(), llvmI64Type,
                rewriter.getIntegerAttr(llvmI64Type, sampleShape[j]));
            auto dimGEP = LLVM::GEPOp::create(
                rewriter, op.getLoc(), llvmPtrType, llvmI64Type, shapeArrAlloca,
                ValueRange{LLVM::ConstantOp::create(
                    rewriter, op.getLoc(), llvmI64Type,
                    rewriter.getIntegerAttr(llvmI64Type, j))});
            LLVM::StoreOp::create(rewriter, op.getLoc(), dimConst, dimGEP);
          }

          // 4b. Store `shapeArrAlloca` in `shapePtrArrayAlloca` for each
          // sampled value.
          auto shapePtrGEP = LLVM::GEPOp::create(
              rewriter, op.getLoc(), llvmPtrType, llvmI64Type,
              shapePtrArrayAlloca,
              ValueRange{LLVM::ConstantOp::create(
                  rewriter, op.getLoc(), llvmI64Type,
                  rewriter.getIntegerAttr(llvmI64Type, i))});
          LLVM::StoreOp::create(rewriter, op.getLoc(), shapeArrAlloca,
                                shapePtrGEP);
        }

        LLVM::CallOp::create(rewriter, op.getLoc(), TypeRange{},
                             SymbolRefAttr::get(ctx, addSampleToTraceFn),
                             ValueRange{func.getArgument(0),
                                        func.getArgument(1),
                                        samplePtrArrayAlloca, numSamplesAlloca,
                                        numDimsArrayAlloca, shapePtrArrayAlloca,
                                        dtypeWidthArrayAlloca});

        LLVM::ReturnOp::create(rewriter, op.getLoc(), ValueRange{});
      }

      if (!moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(addSampleToTraceFn)) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(moduleOp.getBody());

        auto funcType = LLVM::LLVMFunctionType::get(
            llvmVoidType,
            {llvmPtrType, llvmPtrType, llvmPtrType, llvmPtrType, llvmPtrType,
             llvmPtrType, llvmPtrType},
            false);
        LLVM::LLVMFuncOp::create(rewriter, op.getLoc(), addSampleToTraceFn,
                                 funcType, LLVM::Linkage::External);
      }

      SmallVector<Value> jitOperands;
      jitOperands.push_back(trace);
      jitOperands.push_back(symbolConst);
      jitOperands.append(sample.begin(), sample.end());

      SmallVector<Attribute> aliases;
      aliases.push_back(stablehlo::OutputOperandAliasAttr::get(
          ctx, std::vector<int64_t>{}, 0, std::vector<int64_t>{}));

      auto jitCall = enzymexla::JITCallOp::create(
          rewriter, op.getLoc(), TypeRange{loweredTraceType},
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

      auto symbolConst = stablehlo::ConstantOp::create(
          rewriter, op.getLoc(), i64TensorType,
          cast<ElementsAttr>(makeAttr(i64TensorType, symbolPtr)));

      if (!moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(wrapperFn)) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(moduleOp.getBody());

        auto funcType = LLVM::LLVMFunctionType::get(
            llvmVoidType, {llvmPtrType, llvmPtrType, llvmPtrType},
            /*isVarArg=*/false);
        auto func = LLVM::LLVMFuncOp::create(rewriter, op.getLoc(), wrapperFn,
                                             funcType);
        rewriter.setInsertionPointToStart(func.addEntryBlock(rewriter));

        LLVM::CallOp::create(rewriter, op.getLoc(), TypeRange{},
                             SymbolRefAttr::get(ctx, addSubtraceFn),
                             ValueRange{func.getArgument(0),
                                        func.getArgument(1),
                                        func.getArgument(2)});

        LLVM::ReturnOp::create(rewriter, op.getLoc(), ValueRange{});
      }

      if (!moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(addSubtraceFn)) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(moduleOp.getBody());
        auto funcType = LLVM::LLVMFunctionType::get(
            llvmVoidType, {llvmPtrType, llvmPtrType, llvmPtrType},
            /*isVarArg=*/false);
        LLVM::LLVMFuncOp::create(rewriter, op.getLoc(), addSubtraceFn, funcType,
                                 LLVM::Linkage::External);
      }

      SmallVector<Attribute> aliases;
      aliases.push_back(stablehlo::OutputOperandAliasAttr::get(
          ctx, std::vector<int64_t>{}, 0, std::vector<int64_t>{}));

      auto jitCall = enzymexla::JITCallOp::create(
          rewriter, op.getLoc(), TypeRange{loweredTraceType},
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
        auto func = LLVM::LLVMFuncOp::create(rewriter, op.getLoc(), wrapperFn,
                                             funcType);

        rewriter.setInsertionPointToStart(func.addEntryBlock(rewriter));
        LLVM::CallOp::create(rewriter, op.getLoc(), TypeRange{},
                             SymbolRefAttr::get(ctx, addWeightFn),
                             func.getArguments());
        LLVM::ReturnOp::create(rewriter, op.getLoc(), ValueRange{});
      }

      if (!moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(addWeightFn)) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(moduleOp.getBody());
        auto funcType = LLVM::LLVMFunctionType::get(
            llvmVoidType, {llvmPtrType, llvmPtrType}, /*isVarArg=*/false);
        LLVM::LLVMFuncOp::create(rewriter, op.getLoc(), addWeightFn, funcType,
                                 LLVM::Linkage::External);
      }

      SmallVector<Attribute> aliases;
      aliases.push_back(stablehlo::OutputOperandAliasAttr::get(
          ctx, std::vector<int64_t>{}, 0, std::vector<int64_t>{}));

      auto jitCall = enzymexla::JITCallOp::create(
          rewriter, op.getLoc(), TypeRange{loweredTraceType},
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
        auto func = LLVM::LLVMFuncOp::create(rewriter, op.getLoc(), wrapperFn,
                                             funcType);

        rewriter.setInsertionPointToStart(func.addEntryBlock(rewriter));

        auto oneConst =
            LLVM::ConstantOp::create(rewriter, op.getLoc(), llvmI64Type,
                                     rewriter.getIntegerAttr(llvmI64Type, 1));
        auto numResultsConst = LLVM::ConstantOp::create(
            rewriter, op.getLoc(), llvmI64Type,
            rewriter.getIntegerAttr(llvmI64Type, numResults));

        auto numResultsAlloca = LLVM::AllocaOp::create(
            rewriter, op.getLoc(), llvmPtrType, llvmI64Type, oneConst);
        LLVM::StoreOp::create(rewriter, op.getLoc(), numResultsConst,
                              numResultsAlloca);

        auto retvalPtrArrayAlloca = LLVM::AllocaOp::create(
            rewriter, op.getLoc(), llvmPtrType, llvmPtrType, numResultsConst);
        auto numDimsArrayAlloca = LLVM::AllocaOp::create(
            rewriter, op.getLoc(), llvmPtrType, llvmI64Type, numResultsConst);
        auto shapePtrArrayAlloca = LLVM::AllocaOp::create(
            rewriter, op.getLoc(), llvmPtrType, llvmPtrType, numResultsConst);
        auto dtypeWidthArrayAlloca = LLVM::AllocaOp::create(
            rewriter, op.getLoc(), llvmPtrType, llvmI64Type, numResultsConst);

        for (size_t i = 0; i < numResults; ++i) {
          auto resType = cast<RankedTensorType>(retvalVals[i].getType());
          auto resShape = resType.getShape();
          size_t resNumDims = resShape.size();
          size_t resWidth = resType.getElementType().getIntOrFloatBitWidth();

          auto ptrGEP = LLVM::GEPOp::create(
              rewriter, op.getLoc(), llvmPtrType, llvmI64Type,
              retvalPtrArrayAlloca,
              ValueRange{LLVM::ConstantOp::create(
                  rewriter, op.getLoc(), llvmI64Type,
                  rewriter.getIntegerAttr(llvmI64Type, i))});
          LLVM::StoreOp::create(rewriter, op.getLoc(), func.getArgument(1 + i),
                                ptrGEP);

          auto numDimsConst = LLVM::ConstantOp::create(
              rewriter, op.getLoc(), llvmI64Type,
              rewriter.getIntegerAttr(llvmI64Type, resNumDims));
          auto numDimsGEP = LLVM::GEPOp::create(
              rewriter, op.getLoc(), llvmPtrType, llvmI64Type,
              numDimsArrayAlloca,
              ValueRange{LLVM::ConstantOp::create(
                  rewriter, op.getLoc(), llvmI64Type,
                  rewriter.getIntegerAttr(llvmI64Type, i))});
          LLVM::StoreOp::create(rewriter, op.getLoc(), numDimsConst,
                                numDimsGEP);

          auto widthConst = LLVM::ConstantOp::create(
              rewriter, op.getLoc(), llvmI64Type,
              rewriter.getIntegerAttr(llvmI64Type, resWidth));
          auto widthGEP = LLVM::GEPOp::create(
              rewriter, op.getLoc(), llvmPtrType, llvmI64Type,
              dtypeWidthArrayAlloca,
              ValueRange{LLVM::ConstantOp::create(
                  rewriter, op.getLoc(), llvmI64Type,
                  rewriter.getIntegerAttr(llvmI64Type, i))});
          LLVM::StoreOp::create(rewriter, op.getLoc(), widthConst, widthGEP);

          auto shapeSizeConst = LLVM::ConstantOp::create(
              rewriter, op.getLoc(), llvmI64Type,
              rewriter.getIntegerAttr(llvmI64Type, resNumDims));
          auto shapeArrAlloca = LLVM::AllocaOp::create(
              rewriter, op.getLoc(), llvmPtrType, llvmI64Type, shapeSizeConst);

          for (size_t j = 0; j < resNumDims; ++j) {
            auto dimConst = LLVM::ConstantOp::create(
                rewriter, op.getLoc(), llvmI64Type,
                rewriter.getIntegerAttr(llvmI64Type, resShape[j]));
            auto dimGEP = LLVM::GEPOp::create(
                rewriter, op.getLoc(), llvmPtrType, llvmI64Type, shapeArrAlloca,
                ValueRange{LLVM::ConstantOp::create(
                    rewriter, op.getLoc(), llvmI64Type,
                    rewriter.getIntegerAttr(llvmI64Type, j))});
            LLVM::StoreOp::create(rewriter, op.getLoc(), dimConst, dimGEP);
          }

          auto shapePtrGEP = LLVM::GEPOp::create(
              rewriter, op.getLoc(), llvmPtrType, llvmI64Type,
              shapePtrArrayAlloca,
              ValueRange{LLVM::ConstantOp::create(
                  rewriter, op.getLoc(), llvmI64Type,
                  rewriter.getIntegerAttr(llvmI64Type, i))});
          LLVM::StoreOp::create(rewriter, op.getLoc(), shapeArrAlloca,
                                shapePtrGEP);
        }

        LLVM::CallOp::create(rewriter, op.getLoc(), TypeRange{},
                             SymbolRefAttr::get(ctx, addRetvalFn),
                             ValueRange{func.getArgument(0),
                                        retvalPtrArrayAlloca, numResultsAlloca,
                                        numDimsArrayAlloca, shapePtrArrayAlloca,
                                        dtypeWidthArrayAlloca});

        LLVM::ReturnOp::create(rewriter, op.getLoc(), ValueRange{});
      }

      if (!moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(addRetvalFn)) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(moduleOp.getBody());
        auto funcTypeExt =
            LLVM::LLVMFunctionType::get(llvmVoidType,
                                        {llvmPtrType, llvmPtrType, llvmPtrType,
                                         llvmPtrType, llvmPtrType, llvmPtrType},
                                        false);
        LLVM::LLVMFuncOp::create(rewriter, op.getLoc(), addRetvalFn,
                                 funcTypeExt, LLVM::Linkage::External);
      }

      SmallVector<Value> jitOperands;
      jitOperands.push_back(trace);
      jitOperands.append(retvalVals.begin(), retvalVals.end());

      SmallVector<Attribute> aliases;
      aliases.push_back(stablehlo::OutputOperandAliasAttr::get(
          ctx, std::vector<int64_t>{}, 0, std::vector<int64_t>{}));

      auto jitCall = enzymexla::JITCallOp::create(
          rewriter, op.getLoc(), TypeRange{loweredTraceType},
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
      auto symbolConst = stablehlo::ConstantOp::create(
          rewriter, op.getLoc(), i64TensorType,
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

        auto func = LLVM::LLVMFuncOp::create(rewriter, op.getLoc(), wrapperFn,
                                             funcType);

        rewriter.setInsertionPointToStart(func.addEntryBlock(rewriter));

        auto oneConst =
            LLVM::ConstantOp::create(rewriter, op.getLoc(), llvmI64Type,
                                     rewriter.getIntegerAttr(llvmI64Type, 1));

        auto numOutputsConst = LLVM::ConstantOp::create(
            rewriter, op.getLoc(), llvmI64Type,
            rewriter.getIntegerAttr(llvmI64Type, numOutputs));

        auto numOutputsAlloca = LLVM::AllocaOp::create(
            rewriter, op.getLoc(), llvmPtrType, llvmI64Type, oneConst);
        LLVM::StoreOp::create(rewriter, op.getLoc(), numOutputsConst,
                              numOutputsAlloca);

        auto samplePtrArrayAlloca = LLVM::AllocaOp::create(
            rewriter, op.getLoc(), llvmPtrType, llvmPtrType, numOutputsConst);
        auto numDimsArrayAlloca = LLVM::AllocaOp::create(
            rewriter, op.getLoc(), llvmPtrType, llvmI64Type, numOutputsConst);
        auto shapePtrArrayAlloca = LLVM::AllocaOp::create(
            rewriter, op.getLoc(), llvmPtrType, llvmPtrType, numOutputsConst);
        auto dtypeWidthArrayAlloca = LLVM::AllocaOp::create(
            rewriter, op.getLoc(), llvmPtrType, llvmI64Type, numOutputsConst);

        for (size_t i = 0; i < numOutputs; ++i) {
          auto outType = cast<RankedTensorType>(outputs[i].getType());
          auto outShape = outType.getShape();
          size_t outNumDims = outShape.size();
          size_t outWidth = outType.getElementType().getIntOrFloatBitWidth();

          auto ptrGEP = LLVM::GEPOp::create(
              rewriter, op.getLoc(), llvmPtrType, llvmI64Type,
              samplePtrArrayAlloca,
              ValueRange{LLVM::ConstantOp::create(
                  rewriter, op.getLoc(), llvmI64Type,
                  rewriter.getIntegerAttr(llvmI64Type, i))});
          LLVM::StoreOp::create(rewriter, op.getLoc(), func.getArgument(2 + i),
                                ptrGEP);

          auto numDimsConst = LLVM::ConstantOp::create(
              rewriter, op.getLoc(), llvmI64Type,
              rewriter.getIntegerAttr(llvmI64Type, outNumDims));
          auto numDimsGEP = LLVM::GEPOp::create(
              rewriter, op.getLoc(), llvmPtrType, llvmI64Type,
              numDimsArrayAlloca,
              ValueRange{LLVM::ConstantOp::create(
                  rewriter, op.getLoc(), llvmI64Type,
                  rewriter.getIntegerAttr(llvmI64Type, i))});
          LLVM::StoreOp::create(rewriter, op.getLoc(), numDimsConst,
                                numDimsGEP);

          auto widthConst = LLVM::ConstantOp::create(
              rewriter, op.getLoc(), llvmI64Type,
              rewriter.getIntegerAttr(llvmI64Type, outWidth));
          auto widthGEP = LLVM::GEPOp::create(
              rewriter, op.getLoc(), llvmPtrType, llvmI64Type,
              dtypeWidthArrayAlloca,
              ValueRange{LLVM::ConstantOp::create(
                  rewriter, op.getLoc(), llvmI64Type,
                  rewriter.getIntegerAttr(llvmI64Type, i))});
          LLVM::StoreOp::create(rewriter, op.getLoc(), widthConst, widthGEP);

          auto shapeSizeConst = LLVM::ConstantOp::create(
              rewriter, op.getLoc(), llvmI64Type,
              rewriter.getIntegerAttr(llvmI64Type, outNumDims));
          auto shapeArrAlloca = LLVM::AllocaOp::create(
              rewriter, op.getLoc(), llvmPtrType, llvmI64Type, shapeSizeConst);

          for (size_t j = 0; j < outNumDims; ++j) {
            auto dimConst = LLVM::ConstantOp::create(
                rewriter, op.getLoc(), llvmI64Type,
                rewriter.getIntegerAttr(llvmI64Type, outShape[j]));
            auto dimGEP = LLVM::GEPOp::create(
                rewriter, op.getLoc(), llvmPtrType, llvmI64Type, shapeArrAlloca,
                ValueRange{LLVM::ConstantOp::create(
                    rewriter, op.getLoc(), llvmI64Type,
                    rewriter.getIntegerAttr(llvmI64Type, j))});
            LLVM::StoreOp::create(rewriter, op.getLoc(), dimConst, dimGEP);
          }

          auto shapePtrGEP = LLVM::GEPOp::create(
              rewriter, op.getLoc(), llvmPtrType, llvmI64Type,
              shapePtrArrayAlloca,
              ValueRange{LLVM::ConstantOp::create(
                  rewriter, op.getLoc(), llvmI64Type,
                  rewriter.getIntegerAttr(llvmI64Type, i))});
          LLVM::StoreOp::create(rewriter, op.getLoc(), shapeArrAlloca,
                                shapePtrGEP);
        }

        LLVM::CallOp::create(rewriter, op.getLoc(), TypeRange{},
                             SymbolRefAttr::get(ctx, getSampleFn),
                             ValueRange{func.getArgument(0),
                                        func.getArgument(1),
                                        samplePtrArrayAlloca, numOutputsAlloca,
                                        numDimsArrayAlloca, shapePtrArrayAlloca,
                                        dtypeWidthArrayAlloca});

        LLVM::ReturnOp::create(rewriter, op.getLoc(), ValueRange{});
      }

      if (!moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(getSampleFn)) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(moduleOp.getBody());
        auto funcTypeExt = LLVM::LLVMFunctionType::get(
            llvmVoidType,
            {llvmPtrType, llvmPtrType, llvmPtrType, llvmPtrType, llvmPtrType,
             llvmPtrType, llvmPtrType},
            /*isVarArg=*/false);
        LLVM::LLVMFuncOp::create(rewriter, op.getLoc(), getSampleFn,
                                 funcTypeExt, LLVM::Linkage::External);
      }

      SmallVector<Value> jitOperands;
      jitOperands.push_back(constraint);
      jitOperands.push_back(symbolConst);

      for (size_t i = 0; i < numOutputs; ++i) {
        auto outType = outputs[i].getType();
        auto bufConst = stablehlo::ConstantOp::create(
            rewriter, op.getLoc(), outType,
            cast<ElementsAttr>(makeAttr(outType, 0)));
        jitOperands.push_back(bufConst);
      }

      SmallVector<Attribute> aliases;
      for (size_t i = 0; i < numOutputs; ++i) {
        aliases.push_back(stablehlo::OutputOperandAliasAttr::get(
            ctx, std::vector<int64_t>{}, /*operand_index=*/2 + i,
            std::vector<int64_t>{}));
      }

      auto jitCall = enzymexla::JITCallOp::create(
          rewriter, op.getLoc(), op->getResultTypes(),
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
      auto symbolConst = stablehlo::ConstantOp::create(
          rewriter, op.getLoc(), i64TensorType,
          cast<ElementsAttr>(makeAttr(i64TensorType, symbolValue)));

      auto subconstraintPtr = stablehlo::ConstantOp::create(
          rewriter, op.getLoc(), loweredConstraintType,
          cast<ElementsAttr>(makeAttr(loweredConstraintType, 0)));

      std::string wrapperFn = getOrCreateWrapper(getSubconstraintFn);

      if (!moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(wrapperFn)) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(moduleOp.getBody());

        auto funcType = LLVM::LLVMFunctionType::get(
            llvmVoidType, {llvmPtrType, llvmPtrType, llvmPtrType},
            /*isVarArg=*/false);
        auto func = LLVM::LLVMFuncOp::create(rewriter, op.getLoc(), wrapperFn,
                                             funcType);

        rewriter.setInsertionPointToStart(func.addEntryBlock(rewriter));
        LLVM::CallOp::create(rewriter, op.getLoc(), TypeRange{},
                             SymbolRefAttr::get(ctx, getSubconstraintFn),
                             ValueRange{func.getArgument(0),
                                        func.getArgument(1),
                                        func.getArgument(2)});
        LLVM::ReturnOp::create(rewriter, op.getLoc(), ValueRange{});
      }

      if (!moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(getSubconstraintFn)) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(moduleOp.getBody());
        auto funcType = LLVM::LLVMFunctionType::get(
            llvmVoidType, {llvmPtrType, llvmPtrType, llvmPtrType},
            /*isVarArg=*/false);
        LLVM::LLVMFuncOp::create(rewriter, op.getLoc(), getSubconstraintFn,
                                 funcType, LLVM::Linkage::External);
      }

      SmallVector<Attribute> aliases;
      aliases.push_back(stablehlo::OutputOperandAliasAttr::get(
          ctx, std::vector<int64_t>{}, 2, std::vector<int64_t>{}));

      auto jitCall = enzymexla::JITCallOp::create(
          rewriter, op.getLoc(), TypeRange{loweredConstraintType},
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

struct SelectOpConversion : public OpConversionPattern<enzyme::SelectOp> {
  using OpConversionPattern::OpConversionPattern;

  std::string backend;
  SelectOpConversion(std::string backend, TypeConverter &typeConverter,
                     MLIRContext *context, PatternBenefit benefit = 1)
      : OpConversionPattern(typeConverter, context, benefit), backend(backend) {
  }

  LogicalResult
  matchAndRewrite(enzyme::SelectOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto newOp = stablehlo::SelectOp::create(
        rewriter, op.getLoc(), adaptor.getTrueValue().getType(),
        adaptor.getCondition(), adaptor.getTrueValue(),
        adaptor.getFalseValue());

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
      auto elemType = valueType.getElementType();
      size_t width = elemType.getIntOrFloatBitWidth();

      int64_t typeKind = 0;
      if (isa<FloatType>(elemType)) {
        typeKind = 0;
      } else if (auto intType = dyn_cast<IntegerType>(elemType)) {
        typeKind = intType.isUnsigned() ? 2 : 1;
      }

      if (!moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(wrapperFn)) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(moduleOp.getBody());

        auto funcType =
            LLVM::LLVMFunctionType::get(llvmVoidType,
                                        {llvmPtrType, llvmPtrType, llvmPtrType,
                                         llvmPtrType, llvmPtrType, llvmPtrType},
                                        /*isVarArg=*/false);
        auto func = LLVM::LLVMFuncOp::create(rewriter, op.getLoc(), wrapperFn,
                                             funcType);

        rewriter.setInsertionPointToStart(func.addEntryBlock(rewriter));

        auto oneConst =
            LLVM::ConstantOp::create(rewriter, op.getLoc(), llvmI64Type,
                                     rewriter.getIntegerAttr(llvmI64Type, 1));

        auto ndimConst = LLVM::ConstantOp::create(
            rewriter, op.getLoc(), llvmI64Type,
            rewriter.getIntegerAttr(llvmI64Type, ndims));
        auto ndimsAlloca = LLVM::AllocaOp::create(
            rewriter, op.getLoc(), llvmPtrType, llvmI64Type, oneConst);
        LLVM::StoreOp::create(rewriter, op.getLoc(), ndimConst, ndimsAlloca);

        auto widthConst = LLVM::ConstantOp::create(
            rewriter, op.getLoc(), llvmI64Type,
            rewriter.getIntegerAttr(llvmI64Type, width));
        auto widthAlloca = LLVM::AllocaOp::create(
            rewriter, op.getLoc(), llvmPtrType, llvmI64Type, oneConst);
        LLVM::StoreOp::create(rewriter, op.getLoc(), widthConst, widthAlloca);

        auto typeKindConstInner = LLVM::ConstantOp::create(
            rewriter, op.getLoc(), llvmI64Type,
            rewriter.getIntegerAttr(llvmI64Type, typeKind));
        auto typeKindAlloca = LLVM::AllocaOp::create(
            rewriter, op.getLoc(), llvmPtrType, llvmI64Type, oneConst);
        LLVM::StoreOp::create(rewriter, op.getLoc(), typeKindConstInner,
                              typeKindAlloca);

        Value shapeArrAlloca;
        if (ndims > 0) {
          auto shapeSizeConst = LLVM::ConstantOp::create(
              rewriter, op.getLoc(), llvmI64Type,
              rewriter.getIntegerAttr(llvmI64Type, ndims));
          shapeArrAlloca = LLVM::AllocaOp::create(
              rewriter, op.getLoc(), llvmPtrType, llvmI64Type, shapeSizeConst);

          for (size_t i = 0; i < ndims; ++i) {
            auto dimConst = LLVM::ConstantOp::create(
                rewriter, op.getLoc(), llvmI64Type,
                rewriter.getIntegerAttr(llvmI64Type, shape[i]));
            auto dimGEP = LLVM::GEPOp::create(
                rewriter, op.getLoc(), llvmPtrType, llvmI64Type, shapeArrAlloca,
                ValueRange{LLVM::ConstantOp::create(
                    rewriter, op.getLoc(), llvmI64Type,
                    rewriter.getIntegerAttr(llvmI64Type, i))});
            LLVM::StoreOp::create(rewriter, op.getLoc(), dimConst, dimGEP);
          }
        } else {
          shapeArrAlloca =
              LLVM::ZeroOp::create(rewriter, op.getLoc(), llvmPtrType);
        }

        LLVM::CallOp::create(
            rewriter, op.getLoc(), TypeRange{}, SymbolRefAttr::get(ctx, dumpFn),
            ValueRange{func.getArgument(0), func.getArgument(1), ndimsAlloca,
                       shapeArrAlloca, widthAlloca, func.getArgument(5)});

        LLVM::ReturnOp::create(rewriter, op.getLoc(), ValueRange{});
      }

      if (!moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(dumpFn)) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(moduleOp.getBody());
        auto funcType =
            LLVM::LLVMFunctionType::get(llvmVoidType,
                                        {llvmPtrType, llvmPtrType, llvmPtrType,
                                         llvmPtrType, llvmPtrType, llvmPtrType},
                                        /*isVarArg=*/false);
        LLVM::LLVMFuncOp::create(rewriter, op.getLoc(), dumpFn, funcType,
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
      auto labelConst = stablehlo::ConstantOp::create(
          rewriter, op.getLoc(), labelArrayType,
          DenseIntElementsAttr::get(labelArrayType, labelChars));

      auto i64TensorType = RankedTensorType::get({}, llvmI64Type);
      auto ndimsConst = stablehlo::ConstantOp::create(
          rewriter, op.getLoc(), i64TensorType,
          cast<ElementsAttr>(
              makeAttr(i64TensorType, static_cast<int64_t>(ndims))));
      auto widthConst = stablehlo::ConstantOp::create(
          rewriter, op.getLoc(), i64TensorType,
          cast<ElementsAttr>(
              makeAttr(i64TensorType, static_cast<int64_t>(width))));
      auto typeKindConst = stablehlo::ConstantOp::create(
          rewriter, op.getLoc(), i64TensorType,
          cast<ElementsAttr>(makeAttr(i64TensorType, typeKind)));

      Value shapeConst;
      if (ndims > 0) {
        auto shapeArrayType =
            RankedTensorType::get({static_cast<int64_t>(ndims)}, llvmI64Type);
        SmallVector<APInt> shapeAPInt;
        for (auto dim : shape) {
          shapeAPInt.push_back(APInt(64, dim));
        }
        shapeConst = stablehlo::ConstantOp::create(
            rewriter, op.getLoc(), shapeArrayType,
            DenseIntElementsAttr::get(shapeArrayType, shapeAPInt));
      } else {
        auto shapeArrayType = RankedTensorType::get({0}, llvmI64Type);
        shapeConst = stablehlo::ConstantOp::create(
            rewriter, op.getLoc(), shapeArrayType,
            DenseIntElementsAttr::get(shapeArrayType, ArrayRef<APInt>{}));
      }

      SmallVector<Attribute> aliases;
      aliases.push_back(stablehlo::OutputOperandAliasAttr::get(
          ctx, std::vector<int64_t>{}, 0, std::vector<int64_t>{}));

      auto jitCall = enzymexla::JITCallOp::create(
          rewriter, op.getLoc(), TypeRange{valueType},
          mlir::FlatSymbolRefAttr::get(ctx, wrapperFn),
          ValueRange{value, labelConst, ndimsConst, shapeConst, widthConst,
                     typeKindConst},
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
      auto symbolConst = stablehlo::ConstantOp::create(
          rewriter, op.getLoc(), i64TensorType,
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

        auto func = LLVM::LLVMFuncOp::create(rewriter, op.getLoc(), wrapperFn,
                                             funcType);

        rewriter.setInsertionPointToStart(func.addEntryBlock(rewriter));

        auto oneConst =
            LLVM::ConstantOp::create(rewriter, op.getLoc(), llvmI64Type,
                                     rewriter.getIntegerAttr(llvmI64Type, 1));

        auto numOutputsConst = LLVM::ConstantOp::create(
            rewriter, op.getLoc(), llvmI64Type,
            rewriter.getIntegerAttr(llvmI64Type, numOutputs));

        auto numOutputsAlloca = LLVM::AllocaOp::create(
            rewriter, op.getLoc(), llvmPtrType, llvmI64Type, oneConst);
        LLVM::StoreOp::create(rewriter, op.getLoc(), numOutputsConst,
                              numOutputsAlloca);

        auto samplePtrArrayAlloca = LLVM::AllocaOp::create(
            rewriter, op.getLoc(), llvmPtrType, llvmPtrType, numOutputsConst);
        auto numDimsArrayAlloca = LLVM::AllocaOp::create(
            rewriter, op.getLoc(), llvmPtrType, llvmI64Type, numOutputsConst);
        auto shapePtrArrayAlloca = LLVM::AllocaOp::create(
            rewriter, op.getLoc(), llvmPtrType, llvmPtrType, numOutputsConst);
        auto dtypeWidthArrayAlloca = LLVM::AllocaOp::create(
            rewriter, op.getLoc(), llvmPtrType, llvmI64Type, numOutputsConst);

        for (size_t i = 0; i < numOutputs; ++i) {
          auto outType = cast<RankedTensorType>(outputs[i].getType());
          auto outShape = outType.getShape();
          size_t outNumDims = outShape.size();
          size_t outWidth = outType.getElementType().getIntOrFloatBitWidth();

          auto ptrGEP = LLVM::GEPOp::create(
              rewriter, op.getLoc(), llvmPtrType, llvmI64Type,
              samplePtrArrayAlloca,
              ValueRange{LLVM::ConstantOp::create(
                  rewriter, op.getLoc(), llvmI64Type,
                  rewriter.getIntegerAttr(llvmI64Type, i))});
          LLVM::StoreOp::create(rewriter, op.getLoc(), func.getArgument(2 + i),
                                ptrGEP);

          auto numDimsConst = LLVM::ConstantOp::create(
              rewriter, op.getLoc(), llvmI64Type,
              rewriter.getIntegerAttr(llvmI64Type, outNumDims));
          auto numDimsGEP = LLVM::GEPOp::create(
              rewriter, op.getLoc(), llvmPtrType, llvmI64Type,
              numDimsArrayAlloca,
              ValueRange{LLVM::ConstantOp::create(
                  rewriter, op.getLoc(), llvmI64Type,
                  rewriter.getIntegerAttr(llvmI64Type, i))});
          LLVM::StoreOp::create(rewriter, op.getLoc(), numDimsConst,
                                numDimsGEP);

          auto widthConst = LLVM::ConstantOp::create(
              rewriter, op.getLoc(), llvmI64Type,
              rewriter.getIntegerAttr(llvmI64Type, outWidth));
          auto widthGEP = LLVM::GEPOp::create(
              rewriter, op.getLoc(), llvmPtrType, llvmI64Type,
              dtypeWidthArrayAlloca,
              ValueRange{LLVM::ConstantOp::create(
                  rewriter, op.getLoc(), llvmI64Type,
                  rewriter.getIntegerAttr(llvmI64Type, i))});
          LLVM::StoreOp::create(rewriter, op.getLoc(), widthConst, widthGEP);

          auto shapeSizeConst = LLVM::ConstantOp::create(
              rewriter, op.getLoc(), llvmI64Type,
              rewriter.getIntegerAttr(llvmI64Type, outNumDims));
          auto shapeArrAlloca = LLVM::AllocaOp::create(
              rewriter, op.getLoc(), llvmPtrType, llvmI64Type, shapeSizeConst);

          for (size_t j = 0; j < outNumDims; ++j) {
            auto dimConst = LLVM::ConstantOp::create(
                rewriter, op.getLoc(), llvmI64Type,
                rewriter.getIntegerAttr(llvmI64Type, outShape[j]));
            auto dimGEP = LLVM::GEPOp::create(
                rewriter, op.getLoc(), llvmPtrType, llvmI64Type, shapeArrAlloca,
                ValueRange{LLVM::ConstantOp::create(
                    rewriter, op.getLoc(), llvmI64Type,
                    rewriter.getIntegerAttr(llvmI64Type, j))});
            LLVM::StoreOp::create(rewriter, op.getLoc(), dimConst, dimGEP);
          }

          auto shapePtrGEP = LLVM::GEPOp::create(
              rewriter, op.getLoc(), llvmPtrType, llvmI64Type,
              shapePtrArrayAlloca,
              ValueRange{LLVM::ConstantOp::create(
                  rewriter, op.getLoc(), llvmI64Type,
                  rewriter.getIntegerAttr(llvmI64Type, i))});
          LLVM::StoreOp::create(rewriter, op.getLoc(), shapeArrAlloca,
                                shapePtrGEP);
        }

        LLVM::CallOp::create(rewriter, op.getLoc(), TypeRange{},
                             SymbolRefAttr::get(ctx, getSampleFn),
                             ValueRange{func.getArgument(0),
                                        func.getArgument(1),
                                        samplePtrArrayAlloca, numOutputsAlloca,
                                        numDimsArrayAlloca, shapePtrArrayAlloca,
                                        dtypeWidthArrayAlloca});

        LLVM::ReturnOp::create(rewriter, op.getLoc(), ValueRange{});
      }

      if (!moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(getSampleFn)) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(moduleOp.getBody());
        auto funcTypeExt = LLVM::LLVMFunctionType::get(
            llvmVoidType,
            {llvmPtrType, llvmPtrType, llvmPtrType, llvmPtrType, llvmPtrType,
             llvmPtrType, llvmPtrType},
            /*isVarArg=*/false);
        LLVM::LLVMFuncOp::create(rewriter, op.getLoc(), getSampleFn,
                                 funcTypeExt, LLVM::Linkage::External);
      }

      SmallVector<Value> jitOperands;
      jitOperands.push_back(trace);
      jitOperands.push_back(symbolConst);

      for (size_t i = 0; i < numOutputs; ++i) {
        auto outType = outputs[i].getType();
        auto bufConst = stablehlo::ConstantOp::create(
            rewriter, op.getLoc(), outType,
            cast<ElementsAttr>(makeAttr(outType, 0)));
        jitOperands.push_back(bufConst);
      }

      SmallVector<Attribute> aliases;
      for (size_t i = 0; i < numOutputs; ++i) {
        aliases.push_back(stablehlo::OutputOperandAliasAttr::get(
            ctx, std::vector<int64_t>{}, /*operand_index=*/2 + i,
            std::vector<int64_t>{}));
      }

      auto jitCall = enzymexla::JITCallOp::create(
          rewriter, op.getLoc(), op->getResultTypes(),
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

struct CholeskyOpConversion : public OpConversionPattern<enzyme::CholeskyOp> {
  using OpConversionPattern::OpConversionPattern;

  std::string backend;
  CholeskyOpConversion(std::string backend, TypeConverter &typeConverter,
                       MLIRContext *context, PatternBenefit benefit = 1)
      : OpConversionPattern(typeConverter, context, benefit), backend(backend) {
  }

  LogicalResult
  matchAndRewrite(enzyme::CholeskyOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto input = adaptor.getInput();
    auto resultType = cast<RankedTensorType>(op.getResult().getType());
    bool lower = op.getLower();

    auto choleskyOp = stablehlo::CholeskyOp::create(
        rewriter, op.getLoc(), resultType, input, rewriter.getBoolAttr(lower));

    rewriter.replaceOp(op, choleskyOp.getResult());
    return success();
  }
};

struct TriangularSolveOpConversion
    : public OpConversionPattern<enzyme::TriangularSolveOp> {
  using OpConversionPattern::OpConversionPattern;

  std::string backend;
  TriangularSolveOpConversion(std::string backend, TypeConverter &typeConverter,
                              MLIRContext *context, PatternBenefit benefit = 1)
      : OpConversionPattern(typeConverter, context, benefit), backend(backend) {
  }

  LogicalResult
  matchAndRewrite(enzyme::TriangularSolveOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto a = adaptor.getA();
    auto b = adaptor.getB();
    auto resultType = cast<RankedTensorType>(op.getResult().getType());
    auto bType = cast<RankedTensorType>(b.getType());

    bool leftSide = op.getLeftSide();
    bool lower = op.getLower();
    bool unitDiagonal = op.getUnitDiagonal();
    auto transposeA = op.getTransposeA();

    stablehlo::Transpose stablehloTranspose;
    switch (transposeA) {
    case enzyme::Transpose::NO_TRANSPOSE:
      stablehloTranspose = stablehlo::Transpose::NO_TRANSPOSE;
      break;
    case enzyme::Transpose::TRANSPOSE:
      stablehloTranspose = stablehlo::Transpose::TRANSPOSE;
      break;
    case enzyme::Transpose::ADJOINT:
      stablehloTranspose = stablehlo::Transpose::ADJOINT;
      break;
    }

    // StableHLO triangular_solve requires both operands to have the same rank.
    // If b is 1D (vector), reshape to 2D column matrix, solve, then reshape
    // back.
    Value bReshaped = b;
    Type intermediateResultType = resultType;

    if (bType.getRank() == 1) {
      auto shape = bType.getShape();
      auto reshapedType =
          RankedTensorType::get({shape[0], 1}, bType.getElementType());
      bReshaped =
          stablehlo::ReshapeOp::create(rewriter, op.getLoc(), reshapedType, b);
      intermediateResultType = reshapedType;
    }

    auto triangularSolveOp = stablehlo::TriangularSolveOp::create(
        rewriter, op.getLoc(), intermediateResultType, a, bReshaped, leftSide,
        lower, unitDiagonal, stablehloTranspose);

    Value result = triangularSolveOp.getResult();
    if (bType.getRank() == 1) {
      result = stablehlo::ReshapeOp::create(rewriter, op.getLoc(), resultType,
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

    auto lhsBatching = op.getLhsBatchingDimensions();
    auto rhsBatching = op.getRhsBatchingDimensions();
    auto lhsContracting = op.getLhsContractingDimensions();
    auto rhsContracting = op.getRhsContractingDimensions();

    auto dotDimensionNumbers = stablehlo::DotDimensionNumbersAttr::get(
        rewriter.getContext(),
        SmallVector<int64_t>(lhsBatching.begin(), lhsBatching.end()),
        SmallVector<int64_t>(rhsBatching.begin(), rhsBatching.end()),
        SmallVector<int64_t>(lhsContracting.begin(), lhsContracting.end()),
        SmallVector<int64_t>(rhsContracting.begin(), rhsContracting.end()));

    auto dotOp = stablehlo::DotGeneralOp::create(
        rewriter, op.getLoc(), resultType, lhs, rhs, dotDimensionNumbers,
        /*precision_config=*/ArrayAttr(),
        /*algorithm=*/stablehlo::DotAlgorithmAttr());

    rewriter.replaceOp(op, dotOp.getResult());
    return success();
  }
};

// Reference:
// https://github.com/jax-ml/jax/blob/e9b487238f0cfe932200bae842d26826f19ba2bc/jax/_src/lax/other.py#L262
struct LogAddExpOpConversion : public OpConversionPattern<enzyme::LogAddExpOp> {
  using OpConversionPattern::OpConversionPattern;

  std::string backend;
  LogAddExpOpConversion(std::string backend, TypeConverter &typeConverter,
                        MLIRContext *context, PatternBenefit benefit = 1)
      : OpConversionPattern(typeConverter, context, benefit), backend(backend) {
  }

  LogicalResult
  matchAndRewrite(enzyme::LogAddExpOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto lhs = adaptor.getLhs();
    auto rhs = adaptor.getRhs();
    auto resultType = cast<RankedTensorType>(op.getResult().getType());

    auto amax =
        stablehlo::MaxOp::create(rewriter, op.getLoc(), resultType, lhs, rhs);
    auto delta = stablehlo::SubtractOp::create(rewriter, op.getLoc(),
                                               resultType, lhs, rhs);
    auto isNaN =
        stablehlo::CompareOp::create(rewriter, op.getLoc(), delta, delta,
                                     stablehlo::ComparisonDirection::NE);
    auto nanResult =
        stablehlo::AddOp::create(rewriter, op.getLoc(), resultType, lhs, rhs);
    auto absDelta =
        stablehlo::AbsOp::create(rewriter, op.getLoc(), resultType, delta);
    auto negAbsDelta =
        stablehlo::NegOp::create(rewriter, op.getLoc(), resultType, absDelta);
    auto expNegAbsDelta = stablehlo::ExpOp::create(rewriter, op.getLoc(),
                                                   resultType, negAbsDelta);
    auto log1pResult = stablehlo::Log1pOp::create(rewriter, op.getLoc(),
                                                  resultType, expNegAbsDelta);
    auto normalResult = stablehlo::AddOp::create(rewriter, op.getLoc(),
                                                 resultType, amax, log1pResult);
    auto result = stablehlo::SelectOp::create(rewriter, op.getLoc(), resultType,
                                              isNaN, nanResult, normalResult);

    rewriter.replaceOp(op, result);
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
    auto rngBitGenOp = stablehlo::RngBitGeneratorOp::create(
        rewriter, op.getLoc(),
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

      auto shiftAmount = stablehlo::ConstantOp::create(
          rewriter, op.getLoc(), uintResultType,
          DenseElementsAttr::get(
              uintResultType,
              rewriter.getIntegerAttr(uintType, nbits - mantissaBits)));
      auto shiftedBits = stablehlo::ShiftRightLogicalOp::create(
          rewriter, op.getLoc(), uintResultType, randomBits, shiftAmount);

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

      auto onePatternConst = stablehlo::ConstantOp::create(
          rewriter, op.getLoc(), uintResultType,
          DenseElementsAttr::get(
              uintResultType, rewriter.getIntegerAttr(uintType, onePattern)));
      auto floatBits = stablehlo::OrOp::create(
          rewriter, op.getLoc(), uintResultType, shiftedBits, onePatternConst);
      auto floatValue = stablehlo::BitcastConvertOp::create(
          rewriter, op.getLoc(), rankedType, floatBits);
      auto oneConst = stablehlo::ConstantOp::create(
          rewriter, op.getLoc(), rankedType,
          DenseElementsAttr::get(rankedType,
                                 rewriter.getFloatAttr(elemType, 1.0)));
      auto uniform01 = stablehlo::SubtractOp::create(
          rewriter, op.getLoc(), rankedType, floatValue, oneConst);

      auto a = adaptor.getA();
      auto b = adaptor.getB();
      auto aBroadcast = stablehlo::BroadcastInDimOp::create(
          rewriter, op.getLoc(), rankedType, a,
          rewriter.getDenseI64ArrayAttr({}));
      auto bBroadcast = stablehlo::BroadcastInDimOp::create(
          rewriter, op.getLoc(), rankedType, b,
          rewriter.getDenseI64ArrayAttr({}));
      auto range = stablehlo::SubtractOp::create(
          rewriter, op.getLoc(), rankedType, bBroadcast, aBroadcast);
      auto scaled = stablehlo::MulOp::create(rewriter, op.getLoc(), rankedType,
                                             range, uniform01);
      result = stablehlo::AddOp::create(rewriter, op.getLoc(), rankedType,
                                        aBroadcast, scaled);
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

      auto shiftAmount = stablehlo::ConstantOp::create(
          rewriter, op.getLoc(), uintResultType,
          DenseElementsAttr::get(
              uintResultType,
              rewriter.getIntegerAttr(uintType, nbits - mantissaBits)));
      auto shiftedBits = stablehlo::ShiftRightLogicalOp::create(
          rewriter, op.getLoc(), uintResultType, randomBits, shiftAmount);

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

      auto onePatternConst = stablehlo::ConstantOp::create(
          rewriter, op.getLoc(), uintResultType,
          DenseElementsAttr::get(
              uintResultType, rewriter.getIntegerAttr(uintType, onePattern)));
      auto floatBits = stablehlo::OrOp::create(
          rewriter, op.getLoc(), uintResultType, shiftedBits, onePatternConst);

      Value randUniform = stablehlo::BitcastConvertOp::create(
                              rewriter, op.getLoc(), rankedType, floatBits)
                              .getResult();
      auto oneConst = stablehlo::ConstantOp::create(
          rewriter, op.getLoc(), rankedType,
          DenseElementsAttr::get(rankedType,
                                 rewriter.getFloatAttr(elemType, 1.0)));
      randUniform =
          stablehlo::SubtractOp::create(rewriter, op.getLoc(), rankedType,
                                        randUniform, oneConst)
              .getResult();
      auto twoConst = stablehlo::ConstantOp::create(
          rewriter, op.getLoc(), rankedType,
          DenseElementsAttr::get(rankedType,
                                 rewriter.getFloatAttr(elemType, 2.0)));
      Value scaledUniform =
          stablehlo::MulOp::create(rewriter, op.getLoc(), rankedType,
                                   randUniform, twoConst)
              .getResult();
      scaledUniform =
          stablehlo::SubtractOp::create(rewriter, op.getLoc(), rankedType,
                                        scaledUniform, oneConst)
              .getResult();
      auto probit = chlo::ErfInvOp::create(rewriter, op.getLoc(), rankedType,
                                           scaledUniform);
      double sqrt2 = std::sqrt(2.0);
      auto sqrt2Const = stablehlo::ConstantOp::create(
          rewriter, op.getLoc(), rankedType,
          DenseElementsAttr::get(rankedType,
                                 rewriter.getFloatAttr(elemType, sqrt2)));
      auto standardNormal = stablehlo::MulOp::create(
          rewriter, op.getLoc(), rankedType, probit, sqrt2Const);

      auto mu = adaptor.getA();
      auto sigma = adaptor.getB();
      auto muBroadcast = stablehlo::BroadcastInDimOp::create(
          rewriter, op.getLoc(), rankedType, mu,
          rewriter.getDenseI64ArrayAttr({}));
      auto sigmaBroadcast = stablehlo::BroadcastInDimOp::create(
          rewriter, op.getLoc(), rankedType, sigma,
          rewriter.getDenseI64ArrayAttr({}));
      auto scaled = stablehlo::MulOp::create(rewriter, op.getLoc(), rankedType,
                                             sigmaBroadcast, standardNormal);
      result = stablehlo::AddOp::create(rewriter, op.getLoc(), rankedType,
                                        muBroadcast, scaled);
    } else if (distribution == enzyme::RngDistribution::MULTINORMAL) {
      // Multivariate normal: x ~ N(mean, cov)
      // Algorithm: x = mean + chol(cov) * z, where z ~ N(0, I)

      Value mean = adaptor.getA();
      Value cov = adaptor.getB();

      auto meanType = cast<RankedTensorType>(mean.getType());
      auto covType = cast<RankedTensorType>(cov.getType());

      bool scalarMean =
          meanType.getShape().empty() ||
          (meanType.getShape().size() == 1 && meanType.getShape()[0] == 1);

      int64_t dim = covType.getShape()[0];
      auto vectorType = RankedTensorType::get({dim}, elemType);

      // Sample z ~ N(0, I)
      unsigned mantissaBits;
      if (nbits == 16)
        mantissaBits = 10; // TODO bfloat16
      else if (nbits == 32)
        mantissaBits = 23;
      else if (nbits == 64)
        mantissaBits = 52;
      else
        return rewriter.notifyMatchFailure(
            op, "Unsupported float type for MULTINORMAL");
      auto uintVectorType = RankedTensorType::get({dim}, uintType);

      auto shiftAmount = stablehlo::ConstantOp::create(
          rewriter, op.getLoc(), uintVectorType,
          DenseElementsAttr::get(
              uintVectorType,
              rewriter.getIntegerAttr(uintType, nbits - mantissaBits)));
      auto shiftedBits = stablehlo::ShiftRightLogicalOp::create(
          rewriter, op.getLoc(), uintVectorType, randomBits, shiftAmount);

      uint64_t onePattern;
      if (nbits == 16)
        onePattern = 0x3C00;
      else if (nbits == 32)
        onePattern = 0x3F800000;
      else if (nbits == 64)
        onePattern = 0x3FF0000000000000ULL;
      else
        return rewriter.notifyMatchFailure(
            op, "Unsupported float type for MULTINORMAL");

      auto onePatternConst = stablehlo::ConstantOp::create(
          rewriter, op.getLoc(), uintVectorType,
          DenseElementsAttr::get(
              uintVectorType, rewriter.getIntegerAttr(uintType, onePattern)));
      auto floatBits = stablehlo::OrOp::create(
          rewriter, op.getLoc(), uintVectorType, shiftedBits, onePatternConst);

      Value randUniform = stablehlo::BitcastConvertOp::create(
                              rewriter, op.getLoc(), vectorType, floatBits)
                              .getResult();
      auto oneConst = stablehlo::ConstantOp::create(
          rewriter, op.getLoc(), vectorType,
          DenseElementsAttr::get(vectorType,
                                 rewriter.getFloatAttr(elemType, 1.0)));
      randUniform =
          stablehlo::SubtractOp::create(rewriter, op.getLoc(), vectorType,
                                        randUniform, oneConst)
              .getResult();

      auto twoConst = stablehlo::ConstantOp::create(
          rewriter, op.getLoc(), vectorType,
          DenseElementsAttr::get(vectorType,
                                 rewriter.getFloatAttr(elemType, 2.0)));
      Value scaledUniform =
          stablehlo::MulOp::create(rewriter, op.getLoc(), vectorType,
                                   randUniform, twoConst)
              .getResult();
      scaledUniform =
          stablehlo::SubtractOp::create(rewriter, op.getLoc(), vectorType,
                                        scaledUniform, oneConst)
              .getResult();

      auto probit = chlo::ErfInvOp::create(rewriter, op.getLoc(), vectorType,
                                           scaledUniform);

      double sqrt2 = std::sqrt(2.0);
      auto sqrt2Const = stablehlo::ConstantOp::create(
          rewriter, op.getLoc(), vectorType,
          DenseElementsAttr::get(vectorType,
                                 rewriter.getFloatAttr(elemType, sqrt2)));
      Value z = stablehlo::MulOp::create(rewriter, op.getLoc(), vectorType,
                                         probit, sqrt2Const)
                    .getResult();

      auto choleskyOp =
          stablehlo::CholeskyOp::create(rewriter, op.getLoc(), covType, cov,
                                        /*lower=*/rewriter.getBoolAttr(true));
      Value L = choleskyOp.getResult();

      auto dotDimensionNumbers = stablehlo::DotDimensionNumbersAttr::get(
          rewriter.getContext(),
          /*lhs_batching_dimensions=*/{},
          /*rhs_batching_dimensions=*/{},
          /*lhs_contracting_dimensions=*/{1},
          /*rhs_contracting_dimensions=*/{0});

      auto Lz = stablehlo::DotGeneralOp::create(
          rewriter, op.getLoc(), vectorType, L, z, dotDimensionNumbers,
          /*precision_config=*/ArrayAttr(),
          /*algorithm=*/stablehlo::DotAlgorithmAttr());

      if (scalarMean) {
        auto meanBroadcast = stablehlo::BroadcastInDimOp::create(
            rewriter, op.getLoc(), vectorType, mean,
            rewriter.getDenseI64ArrayAttr({}));
        result = stablehlo::AddOp::create(rewriter, op.getLoc(), vectorType,
                                          meanBroadcast, Lz);
      } else {
        result = stablehlo::AddOp::create(rewriter, op.getLoc(), vectorType,
                                          mean, Lz);
      }
    } else {
      return rewriter.notifyMatchFailure(op, "Unknown RNG distribution");
    }

    rewriter.replaceOp(op, {outputState, result});
    return success();
  }
};

// Reference (_rbg_split):
// https://github.com/jax-ml/jax/blob/3aa8a6b0d4de5e554f45db638b0f3056e4c520f1/jax/_src/prng.py#L1271
struct RandomSplitOpConversion
    : public OpConversionPattern<enzyme::RandomSplitOp> {
  using OpConversionPattern::OpConversionPattern;

  std::string backend;
  bool debugDump;
  RandomSplitOpConversion(std::string backend, bool debugDump,
                          TypeConverter &typeConverter, MLIRContext *context,
                          PatternBenefit benefit = 1)
      : OpConversionPattern(typeConverter, context, benefit), backend(backend),
        debugDump(debugDump) {}

  LogicalResult
  matchAndRewrite(enzyme::RandomSplitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto rngState = adaptor.getRngState();
    auto rngStateType = cast<RankedTensorType>(rngState.getType());
    auto elemType = cast<IntegerType>(rngStateType.getElementType());
    auto loc = op.getLoc();

    // Check RNG state is a tensor<2xui64>
    if (rngStateType.getShape().size() != 1 ||
        rngStateType.getShape()[0] != 2 || elemType.getWidth() != 64 ||
        !elemType.isUnsigned()) {
      return rewriter.notifyMatchFailure(op, "Unsupported RNG state");
    }

    size_t numOutputs = op.getNumResults();

    auto ui32Type =
        IntegerType::get(rewriter.getContext(), 32, IntegerType::Unsigned);
    auto ui32x2x2Type = RankedTensorType::get({2, 2}, ui32Type);
    auto ui32x4Type = RankedTensorType::get({4}, ui32Type);

    // tensor<2xui64> -> tensor<4xui32>
    // little endian [low32([0]), high32([0]), low32([1]), high32([1])]
    auto bitcastTo2x2 = stablehlo::BitcastConvertOp::create(
        rewriter, loc, ui32x2x2Type, rngState);
    auto reshapedKey =
        stablehlo::ReshapeOp::create(rewriter, loc, ui32x4Type, bitcastTo2x2);
    auto ui32TensorType = RankedTensorType::get({}, ui32Type);
    auto ui32x1Type = RankedTensorType::get({1}, ui32Type);

    // Extract keys
    auto key0_0 = stablehlo::ReshapeOp::create(
        rewriter, op.getLoc(), ui32TensorType,
        stablehlo::SliceOp::create(rewriter, op.getLoc(), ui32x1Type,
                                   reshapedKey,
                                   rewriter.getDenseI64ArrayAttr({0}),
                                   rewriter.getDenseI64ArrayAttr({1}),
                                   rewriter.getDenseI64ArrayAttr({1})));
    auto key0_1 = stablehlo::ReshapeOp::create(
        rewriter, op.getLoc(), ui32TensorType,
        stablehlo::SliceOp::create(rewriter, op.getLoc(), ui32x1Type,
                                   reshapedKey,
                                   rewriter.getDenseI64ArrayAttr({1}),
                                   rewriter.getDenseI64ArrayAttr({2}),
                                   rewriter.getDenseI64ArrayAttr({1})));
    auto key1_0 = stablehlo::ReshapeOp::create(
        rewriter, op.getLoc(), ui32TensorType,
        stablehlo::SliceOp::create(rewriter, op.getLoc(), ui32x1Type,
                                   reshapedKey,
                                   rewriter.getDenseI64ArrayAttr({2}),
                                   rewriter.getDenseI64ArrayAttr({3}),
                                   rewriter.getDenseI64ArrayAttr({1})));
    auto key1_1 = stablehlo::ReshapeOp::create(
        rewriter, op.getLoc(), ui32TensorType,
        stablehlo::SliceOp::create(rewriter, op.getLoc(), ui32x1Type,
                                   reshapedKey,
                                   rewriter.getDenseI64ArrayAttr({3}),
                                   rewriter.getDenseI64ArrayAttr({4}),
                                   rewriter.getDenseI64ArrayAttr({1})));

    // Construct counters
    auto counterType =
        RankedTensorType::get({static_cast<int64_t>(numOutputs)}, ui32Type);

    auto counts1 = stablehlo::ConstantOp::create(
        rewriter, loc, counterType,
        DenseElementsAttr::get(counterType,
                               rewriter.getIntegerAttr(ui32Type, 0)));
    auto counts2 = stablehlo::IotaOp::create(rewriter, loc, counterType,
                                             rewriter.getI64IntegerAttr(0));

    // Broadcast keys
    auto key0_0_bcast = stablehlo::BroadcastInDimOp::create(
        rewriter, loc, counterType, key0_0, rewriter.getDenseI64ArrayAttr({}));
    auto key0_1_bcast = stablehlo::BroadcastInDimOp::create(
        rewriter, loc, counterType, key0_1, rewriter.getDenseI64ArrayAttr({}));
    auto key1_0_bcast = stablehlo::BroadcastInDimOp::create(
        rewriter, loc, counterType, key1_0, rewriter.getDenseI64ArrayAttr({}));
    auto key1_1_bcast = stablehlo::BroadcastInDimOp::create(
        rewriter, loc, counterType, key1_1, rewriter.getDenseI64ArrayAttr({}));

    auto [h0_0, h0_1] = threefry2x32Hash(rewriter, loc, key0_0_bcast,
                                         key0_1_bcast, counts1, counts2);
    auto [h1_0, h1_1] = threefry2x32Hash(rewriter, loc, key1_0_bcast,
                                         key1_1_bcast, counts1, counts2);

    // Compute output keys
    //   output[i][0] = combine(h0_0[i], h0_1[i])
    //   output[i][1] = combine(h1_0[i], h1_1[i])
    SmallVector<Value> outputKeys;
    for (size_t i = 0; i < numOutputs; ++i) {
      auto h0_0_i = stablehlo::SliceOp::create(
          rewriter, loc, ui32x1Type, h0_0,
          rewriter.getDenseI64ArrayAttr({static_cast<int64_t>(i)}),
          rewriter.getDenseI64ArrayAttr({static_cast<int64_t>(i + 1)}),
          rewriter.getDenseI64ArrayAttr({1}));
      auto h0_1_i = stablehlo::SliceOp::create(
          rewriter, loc, ui32x1Type, h0_1,
          rewriter.getDenseI64ArrayAttr({static_cast<int64_t>(i)}),
          rewriter.getDenseI64ArrayAttr({static_cast<int64_t>(i + 1)}),
          rewriter.getDenseI64ArrayAttr({1}));
      auto h1_0_i = stablehlo::SliceOp::create(
          rewriter, loc, ui32x1Type, h1_0,
          rewriter.getDenseI64ArrayAttr({static_cast<int64_t>(i)}),
          rewriter.getDenseI64ArrayAttr({static_cast<int64_t>(i + 1)}),
          rewriter.getDenseI64ArrayAttr({1}));
      auto h1_1_i = stablehlo::SliceOp::create(
          rewriter, loc, ui32x1Type, h1_1,
          rewriter.getDenseI64ArrayAttr({static_cast<int64_t>(i)}),
          rewriter.getDenseI64ArrayAttr({static_cast<int64_t>(i + 1)}),
          rewriter.getDenseI64ArrayAttr({1}));

      // Concatenate [h0_0[i], h0_1[i], h1_0[i], h1_1[i]] -> tensor<4xui32>
      SmallVector<Value> parts = {h0_0_i, h0_1_i, h1_0_i, h1_1_i};
      auto concatType = RankedTensorType::get({4}, ui32Type);
      auto concated = stablehlo::ConcatenateOp::create(
          rewriter, loc, concatType, parts, rewriter.getI64IntegerAttr(0));

      // Restore: tensor<4xui32> -> tensor<2xui64>
      auto reshaped =
          stablehlo::ReshapeOp::create(rewriter, loc, ui32x2x2Type, concated);
      auto outputKey = stablehlo::BitcastConvertOp::create(
          rewriter, loc, rngStateType, reshaped);

      outputKeys.push_back(outputKey);
    }

    rewriter.replaceOp(op, outputKeys);
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
      auto symbolConst = stablehlo::ConstantOp::create(
          rewriter, op.getLoc(), i64TensorType,
          cast<ElementsAttr>(makeAttr(i64TensorType, symbolValue)));

      auto subtracePtr = stablehlo::ConstantOp::create(
          rewriter, op.getLoc(), loweredTraceType,
          cast<ElementsAttr>(makeAttr(loweredTraceType, 0)));

      std::string wrapperFn = getOrCreateWrapper(getSubtraceFn);

      if (!moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(wrapperFn)) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(moduleOp.getBody());

        auto funcType = LLVM::LLVMFunctionType::get(
            llvmVoidType, {llvmPtrType, llvmPtrType, llvmPtrType},
            /*isVarArg=*/false);
        auto func = LLVM::LLVMFuncOp::create(rewriter, op.getLoc(), wrapperFn,
                                             funcType);

        rewriter.setInsertionPointToStart(func.addEntryBlock(rewriter));
        LLVM::CallOp::create(rewriter, op.getLoc(), TypeRange{},
                             SymbolRefAttr::get(ctx, getSubtraceFn),
                             ValueRange{func.getArgument(0),
                                        func.getArgument(1),
                                        func.getArgument(2)});
        LLVM::ReturnOp::create(rewriter, op.getLoc(), ValueRange{});
      }

      if (!moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(getSubtraceFn)) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(moduleOp.getBody());
        auto funcType = LLVM::LLVMFunctionType::get(
            llvmVoidType, {llvmPtrType, llvmPtrType, llvmPtrType},
            /*isVarArg=*/false);
        LLVM::LLVMFuncOp::create(rewriter, op.getLoc(), getSubtraceFn, funcType,
                                 LLVM::Linkage::External);
      }

      SmallVector<Attribute> aliases;
      aliases.push_back(stablehlo::OutputOperandAliasAttr::get(
          ctx, std::vector<int64_t>{}, 2, std::vector<int64_t>{}));

      auto jitCall = enzymexla::JITCallOp::create(
          rewriter, op.getLoc(), TypeRange{loweredTraceType},
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
      auto weightConst = stablehlo::ConstantOp::create(
          rewriter, op.getLoc(), weightType,
          cast<ElementsAttr>(makeAttr(weightType, 0)));

      if (!moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(wrapperFn)) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(moduleOp.getBody());

        auto funcType = LLVM::LLVMFunctionType::get(
            llvmVoidType, {llvmPtrType, llvmPtrType}, /*isVarArg=*/false);
        auto func = LLVM::LLVMFuncOp::create(rewriter, op.getLoc(), wrapperFn,
                                             funcType);

        rewriter.setInsertionPointToStart(func.addEntryBlock(rewriter));
        LLVM::CallOp::create(rewriter, op.getLoc(), TypeRange{},
                             SymbolRefAttr::get(ctx, getWeightFn),
                             func.getArguments());
        LLVM::ReturnOp::create(rewriter, op.getLoc(), ValueRange{});
      }

      if (!moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(getWeightFn)) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(moduleOp.getBody());
        auto funcType = LLVM::LLVMFunctionType::get(
            llvmVoidType, {llvmPtrType, llvmPtrType}, /*isVarArg=*/false);
        LLVM::LLVMFuncOp::create(rewriter, op.getLoc(), getWeightFn, funcType,
                                 LLVM::Linkage::External);
      }

      SmallVector<Attribute> aliases;
      aliases.push_back(stablehlo::OutputOperandAliasAttr::get(
          ctx, std::vector<int64_t>{}, 1, std::vector<int64_t>{}));

      auto jitCall = enzymexla::JITCallOp::create(
          rewriter, op.getLoc(), TypeRange{weightType},
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

struct GetFlattenedSamplesFromTraceOpConversion
    : public OpConversionPattern<enzyme::GetFlattenedSamplesFromTraceOp> {
  using OpConversionPattern::OpConversionPattern;

  std::string backend;
  GetFlattenedSamplesFromTraceOpConversion(std::string backend,
                                           TypeConverter &typeConverter,
                                           MLIRContext *context,
                                           PatternBenefit benefit = 1)
      : OpConversionPattern(typeConverter, context, benefit), backend(backend) {
  }

  LogicalResult
  matchAndRewrite(enzyme::GetFlattenedSamplesFromTraceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto ctx = op->getContext();

    Value trace = adaptor.getTrace();
    auto positionType = cast<RankedTensorType>(op.getPosition().getType());
    auto selection = op.getSelectionAttr();

    if (backend == "cpu") {
      auto moduleOp = op->getParentOfType<ModuleOp>();

      auto llvmPtrType = LLVM::LLVMPointerType::get(ctx);
      auto llvmVoidType = LLVM::LLVMVoidType::get(ctx);
      auto llvmI64Type = IntegerType::get(ctx, 64);

      std::string getFlattenedFn =
          "enzyme_probprog_get_flattened_samples_from_trace";

      SmallVector<uint64_t> flattenedSymbols;
      SmallVector<uint64_t> addressLengths;

      for (auto addr : selection) {
        auto address = cast<ArrayAttr>(addr);
        addressLengths.push_back(address.size());
        for (auto sym : address) {
          auto symbolAttr = cast<enzyme::SymbolAttr>(sym);
          flattenedSymbols.push_back(symbolAttr.getPtr());
        }
      }

      size_t numAddresses = addressLengths.size();
      size_t totalSymbols = flattenedSymbols.size();

      auto i64TensorType = RankedTensorType::get({}, llvmI64Type);
      auto numAddressesConst = stablehlo::ConstantOp::create(
          rewriter, op.getLoc(), i64TensorType,
          cast<ElementsAttr>(makeAttr(i64TensorType, numAddresses)));

      auto totalSymbolsConst = stablehlo::ConstantOp::create(
          rewriter, op.getLoc(), i64TensorType,
          cast<ElementsAttr>(makeAttr(i64TensorType, totalSymbols)));

      auto addressLengthsArrType = RankedTensorType::get(
          {static_cast<int64_t>(numAddresses)}, llvmI64Type);
      SmallVector<APInt> addressLengthsAPInt;
      for (auto len : addressLengths) {
        addressLengthsAPInt.push_back(APInt(64, len));
      }
      auto addressLengthsConst = stablehlo::ConstantOp::create(
          rewriter, op.getLoc(), addressLengthsArrType,
          DenseIntElementsAttr::get(addressLengthsArrType,
                                    addressLengthsAPInt));

      auto flattenedSymbolsArrType = RankedTensorType::get(
          {static_cast<int64_t>(totalSymbols)}, llvmI64Type);
      SmallVector<APInt> flattenedSymbolsAPInt;
      for (auto sym : flattenedSymbols) {
        flattenedSymbolsAPInt.push_back(APInt(64, sym));
      }
      auto flattenedSymbolsConst = stablehlo::ConstantOp::create(
          rewriter, op.getLoc(), flattenedSymbolsArrType,
          DenseIntElementsAttr::get(flattenedSymbolsArrType,
                                    flattenedSymbolsAPInt));

      SmallVector<Type> llvmArgTypes;
      llvmArgTypes.push_back(llvmPtrType); // trace
      llvmArgTypes.push_back(llvmPtrType); // num_addresses
      llvmArgTypes.push_back(llvmPtrType); // total_symbols
      llvmArgTypes.push_back(llvmPtrType); // address_lengths array
      llvmArgTypes.push_back(llvmPtrType); // flattened_symbols array
      llvmArgTypes.push_back(llvmPtrType); // position output buffer

      auto funcType = LLVM::LLVMFunctionType::get(llvmVoidType, llvmArgTypes,
                                                  /*isVarArg=*/false);

      SmallVector<Type> originalTypes = {positionType};
      std::string wrapperFn = getOrCreateWrapper(getFlattenedFn, originalTypes);

      if (!moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(wrapperFn)) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(moduleOp.getBody());

        auto func = LLVM::LLVMFuncOp::create(rewriter, op.getLoc(), wrapperFn,
                                             funcType);
        rewriter.setInsertionPointToStart(func.addEntryBlock(rewriter));

        LLVM::CallOp::create(
            rewriter, op.getLoc(), TypeRange{},
            SymbolRefAttr::get(ctx, getFlattenedFn),
            ValueRange{func.getArgument(0), func.getArgument(1),
                       func.getArgument(2), func.getArgument(3),
                       func.getArgument(4), func.getArgument(5)});

        LLVM::ReturnOp::create(rewriter, op.getLoc(), ValueRange{});
      }

      if (!moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(getFlattenedFn)) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(moduleOp.getBody());
        auto funcTypeExt =
            LLVM::LLVMFunctionType::get(llvmVoidType,
                                        {llvmPtrType, llvmPtrType, llvmPtrType,
                                         llvmPtrType, llvmPtrType, llvmPtrType},
                                        /*isVarArg=*/false);
        LLVM::LLVMFuncOp::create(rewriter, op.getLoc(), getFlattenedFn,
                                 funcTypeExt, LLVM::Linkage::External);
      }

      auto positionConst = stablehlo::ConstantOp::create(
          rewriter, op.getLoc(), positionType,
          cast<ElementsAttr>(makeAttr(positionType, 0)));

      SmallVector<Value> jitOperands;
      jitOperands.push_back(trace);
      jitOperands.push_back(numAddressesConst);
      jitOperands.push_back(totalSymbolsConst);
      jitOperands.push_back(addressLengthsConst);
      jitOperands.push_back(flattenedSymbolsConst);
      jitOperands.push_back(positionConst);

      SmallVector<Attribute> aliases;
      aliases.push_back(stablehlo::OutputOperandAliasAttr::get(
          ctx, std::vector<int64_t>{}, 5, std::vector<int64_t>{}));

      auto jitCall = enzymexla::JITCallOp::create(
          rewriter, op.getLoc(), TypeRange{positionType},
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
    }

    return rewriter.notifyMatchFailure(op, "Unknown backend " + backend);
  }
};

struct ForLoopOpConversion : public OpConversionPattern<enzyme::ForLoopOp> {
  using OpConversionPattern::OpConversionPattern;

  std::string backend;
  ForLoopOpConversion(std::string backend, TypeConverter &typeConverter,
                      MLIRContext *context, PatternBenefit benefit = 1)
      : OpConversionPattern(typeConverter, context, benefit), backend(backend) {
  }

  LogicalResult
  matchAndRewrite(enzyme::ForLoopOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Value> initVals = {adaptor.getLowerBound()};
    initVals.append(adaptor.getInitArgs().begin(), adaptor.getInitArgs().end());

    SmallVector<Type> loopTypes = {adaptor.getLowerBound().getType()};
    for (auto result : op.getResults())
      loopTypes.push_back(typeConverter->convertType(result.getType()));

    auto whileOp =
        stablehlo::WhileOp::create(rewriter, op.getLoc(), loopTypes, initVals);

    Block *condBlock = rewriter.createBlock(&whileOp.getCond());
    for (auto type : loopTypes)
      condBlock->addArgument(type, op.getLoc());

    rewriter.setInsertionPointToStart(condBlock);
    Value iv = condBlock->getArgument(0);
    auto cond = stablehlo::CompareOp::create(
        rewriter, op.getLoc(), iv, adaptor.getUpperBound(),
        stablehlo::ComparisonDirection::LT);
    stablehlo::ReturnOp::create(rewriter, op.getLoc(), cond.getResult());

    Block *bodyBlock = rewriter.createBlock(&whileOp.getBody());
    for (auto type : loopTypes)
      bodyBlock->addArgument(type, op.getLoc());

    rewriter.setInsertionPointToStart(bodyBlock);

    Block &origBody = op.getRegion().front();
    rewriter.mergeBlocks(&origBody, bodyBlock, bodyBlock->getArguments());
    auto yieldOp = cast<enzyme::YieldOp>(bodyBlock->getTerminator());
    rewriter.setInsertionPoint(yieldOp);

    // Return values: [ivNext, yielded_values...]
    Value ivNext = stablehlo::AddOp::create(
        rewriter, op.getLoc(), bodyBlock->getArgument(0), adaptor.getStep());
    SmallVector<Value> yieldedVals;
    yieldedVals.push_back(ivNext);
    for (auto val : yieldOp.getOperands()) {
      Value remappedVal = rewriter.getRemappedValue(val);
      yieldedVals.push_back(remappedVal);
    }

    stablehlo::ReturnOp::create(rewriter, op.getLoc(), yieldedVals);
    rewriter.eraseOp(yieldOp);

    // Drop iv
    rewriter.replaceOp(op, whileOp.getResults().drop_front());
    return success();
  }
};

struct WhileLoopOpConversion : public OpConversionPattern<enzyme::WhileLoopOp> {
  using OpConversionPattern::OpConversionPattern;

  std::string backend;
  WhileLoopOpConversion(std::string backend, TypeConverter &typeConverter,
                        MLIRContext *context, PatternBenefit benefit = 1)
      : OpConversionPattern(typeConverter, context, benefit), backend(backend) {
  }

  LogicalResult
  matchAndRewrite(enzyme::WhileLoopOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Type> loopTypes;
    for (auto result : op.getResults())
      loopTypes.push_back(typeConverter->convertType(result.getType()));

    auto whileOp = stablehlo::WhileOp::create(rewriter, op.getLoc(), loopTypes,
                                              adaptor.getInitArgs());

    Block *condBlock = rewriter.createBlock(&whileOp.getCond());
    for (auto type : loopTypes)
      condBlock->addArgument(type, op.getLoc());

    rewriter.setInsertionPointToStart(condBlock);

    Block &origCond = op.getConditionRegion().front();
    rewriter.mergeBlocks(&origCond, condBlock, condBlock->getArguments());
    auto condYieldOp = cast<enzyme::YieldOp>(condBlock->getTerminator());
    rewriter.setInsertionPoint(condYieldOp);

    if (condYieldOp.getOperands().size() != 1) {
      return rewriter.notifyMatchFailure(
          op, "Condition region must yield exactly one boolean value");
    }

    Value condValue = rewriter.getRemappedValue(condYieldOp.getOperand(0));
    stablehlo::ReturnOp::create(rewriter, op.getLoc(), condValue);
    rewriter.eraseOp(condYieldOp);

    Block *bodyBlock = rewriter.createBlock(&whileOp.getBody());
    for (auto type : loopTypes)
      bodyBlock->addArgument(type, op.getLoc());

    rewriter.setInsertionPointToStart(bodyBlock);

    Block &origBody = op.getBodyRegion().front();
    rewriter.mergeBlocks(&origBody, bodyBlock, bodyBlock->getArguments());
    auto bodyYieldOp = cast<enzyme::YieldOp>(bodyBlock->getTerminator());
    rewriter.setInsertionPoint(bodyYieldOp);

    SmallVector<Value> yieldedVals;
    for (auto val : bodyYieldOp.getOperands()) {
      Value remappedVal = rewriter.getRemappedValue(val);
      yieldedVals.push_back(remappedVal);
    }

    stablehlo::ReturnOp::create(rewriter, op.getLoc(), yieldedVals);
    rewriter.eraseOp(bodyYieldOp);

    rewriter.replaceOp(op, whileOp.getResults());
    return success();
  }
};

struct PopcountOpConversion : public OpConversionPattern<enzyme::PopcountOp> {
  using OpConversionPattern::OpConversionPattern;

  std::string backend;
  PopcountOpConversion(std::string backend, TypeConverter &typeConverter,
                       MLIRContext *context, PatternBenefit benefit = 1)
      : OpConversionPattern(typeConverter, context, benefit), backend(backend) {
  }

  LogicalResult
  matchAndRewrite(enzyme::PopcountOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultType = cast<RankedTensorType>(op.getResult().getType());
    auto popcntOp = stablehlo::PopulationCountOp::create(
        rewriter, op.getLoc(), resultType, adaptor.getOperand());
    rewriter.replaceOp(op, popcntOp.getResult());
    return success();
  }
};

struct DynamicExtractOpConversion
    : public OpConversionPattern<enzyme::DynamicExtractOp> {
  using OpConversionPattern::OpConversionPattern;

  std::string backend;
  DynamicExtractOpConversion(std::string backend, TypeConverter &typeConverter,
                             MLIRContext *context, PatternBenefit benefit = 1)
      : OpConversionPattern(typeConverter, context, benefit), backend(backend) {
  }

  LogicalResult
  matchAndRewrite(enzyme::DynamicExtractOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto input = adaptor.getInput();
    auto index = adaptor.getIndex();
    auto inputType = cast<RankedTensorType>(input.getType());
    auto resultType = cast<RankedTensorType>(op.getResult().getType());

    if (inputType.getRank() != 2) {
      return rewriter.notifyMatchFailure(op, "Input must be 2D tensor");
    }

    int64_t positionSize = inputType.getShape()[1];
    auto elemType = inputType.getElementType();

    auto indexType = cast<RankedTensorType>(index.getType());
    auto zeroConst = stablehlo::ConstantOp::create(
        rewriter, op.getLoc(), indexType,
        DenseElementsAttr::get(indexType, rewriter.getI64IntegerAttr(0)));

    auto slicedType = RankedTensorType::get({1, positionSize}, elemType);
    auto dynamicSlice = stablehlo::DynamicSliceOp::create(
        rewriter, op.getLoc(), slicedType, input, ValueRange{index, zeroConst},
        rewriter.getDenseI64ArrayAttr({1, positionSize}));

    auto reshapeOp = stablehlo::ReshapeOp::create(rewriter, op.getLoc(),
                                                  resultType, dynamicSlice);

    rewriter.replaceOp(op, reshapeOp.getResult());
    return success();
  }
};

struct DynamicUpdateOpConversion
    : public OpConversionPattern<enzyme::DynamicUpdateOp> {
  using OpConversionPattern::OpConversionPattern;

  std::string backend;
  DynamicUpdateOpConversion(std::string backend, TypeConverter &typeConverter,
                            MLIRContext *context, PatternBenefit benefit = 1)
      : OpConversionPattern(typeConverter, context, benefit), backend(backend) {
  }

  LogicalResult
  matchAndRewrite(enzyme::DynamicUpdateOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto input = adaptor.getInput();
    auto index = adaptor.getIndex();
    auto value = adaptor.getValue();
    auto inputType = cast<RankedTensorType>(input.getType());
    auto valueType = cast<RankedTensorType>(value.getType());

    if (inputType.getRank() == 1 && valueType.getRank() == 0) {
      auto elemType = valueType.getElementType();

      auto reshapedValueType = RankedTensorType::get({1}, elemType);
      auto reshapedValue = stablehlo::ReshapeOp::create(
          rewriter, op.getLoc(), reshapedValueType, value);

      auto dynamicUpdateSlice = stablehlo::DynamicUpdateSliceOp::create(
          rewriter, op.getLoc(), inputType, input, reshapedValue,
          ValueRange{index});

      rewriter.replaceOp(op, dynamicUpdateSlice.getResult());
      return success();
    }

    if (inputType.getRank() == 2 && valueType.getRank() == 1) {
      int64_t positionSize = valueType.getShape()[0];
      auto elemType = valueType.getElementType();

      auto reshapedValueType =
          RankedTensorType::get({1, positionSize}, elemType);
      auto reshapedValue = stablehlo::ReshapeOp::create(
          rewriter, op.getLoc(), reshapedValueType, value);

      auto indexType = cast<RankedTensorType>(index.getType());
      auto zeroConst = stablehlo::ConstantOp::create(
          rewriter, op.getLoc(), indexType,
          DenseElementsAttr::get(indexType, rewriter.getI64IntegerAttr(0)));

      auto dynamicUpdateSlice = stablehlo::DynamicUpdateSliceOp::create(
          rewriter, op.getLoc(), inputType, input, reshapedValue,
          ValueRange{index, zeroConst});

      rewriter.replaceOp(op, dynamicUpdateSlice.getResult());
      return success();
    }

    return rewriter.notifyMatchFailure(
        op, "Unsupported input/value tensor ranks for dynamic_update");
  }
};

struct UnflattenSliceOpConversion
    : public OpConversionPattern<enzyme::UnflattenSliceOp> {
  using OpConversionPattern::OpConversionPattern;

  std::string backend;
  UnflattenSliceOpConversion(std::string backend, TypeConverter &typeConverter,
                             MLIRContext *context, PatternBenefit benefit = 1)
      : OpConversionPattern(typeConverter, context, benefit), backend(backend) {
  }

  LogicalResult
  matchAndRewrite(enzyme::UnflattenSliceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultType = cast<RankedTensorType>(op.getResult().getType());
    auto positionType = cast<RankedTensorType>(adaptor.getPosition().getType());
    auto elemType = resultType.getElementType();
    int64_t offset = op.getOffset();

    // Handle both 1D (single sample) and 2D (batched samples) inputs
    if (positionType.getRank() == 1) {
      // 1D input: position[positionSize] -> result[...shape...]
      int64_t numElements = 1;
      for (auto dim : resultType.getShape()) {
        if (dim == ShapedType::kDynamic) {
          return rewriter.notifyMatchFailure(op,
                                             "Dynamic shapes not supported");
        }
        numElements *= dim;
      }

      SmallVector<int64_t> startIndices = {offset};
      SmallVector<int64_t> limitIndices = {offset + numElements};
      SmallVector<int64_t> strides = {1};

      auto slicedType = RankedTensorType::get({numElements}, elemType);
      auto sliceOp = stablehlo::SliceOp::create(
          rewriter, op.getLoc(), slicedType, adaptor.getPosition(),
          rewriter.getDenseI64ArrayAttr(startIndices),
          rewriter.getDenseI64ArrayAttr(limitIndices),
          rewriter.getDenseI64ArrayAttr(strides));

      auto reshapeOp = stablehlo::ReshapeOp::create(rewriter, op.getLoc(),
                                                    resultType, sliceOp);
      rewriter.replaceOp(op, reshapeOp.getResult());
    } else if (positionType.getRank() == 2) {
      // 2D input: position[batchSize, positionSize] -> result[batchSize,
      // ...shape...]
      int64_t batchSize = positionType.getShape()[0];

      // Result shape should be [batchSize, ...originalShape...]
      // Compute numElements from result shape excluding batch dimension
      if (resultType.getRank() < 1 || resultType.getShape()[0] != batchSize) {
        return rewriter.notifyMatchFailure(
            op, "Result type must have batch dimension matching input");
      }

      int64_t numElements = 1;
      SmallVector<int64_t> originalShape;
      for (int64_t i = 1; i < resultType.getRank(); ++i) {
        auto dim = resultType.getShape()[i];
        if (dim == ShapedType::kDynamic) {
          return rewriter.notifyMatchFailure(op,
                                             "Dynamic shapes not supported");
        }
        originalShape.push_back(dim);
        numElements *= dim;
      }

      // Slice columns: [batchSize, positionSize] -> [batchSize, numElements]
      SmallVector<int64_t> startIndices = {0, offset};
      SmallVector<int64_t> limitIndices = {batchSize, offset + numElements};
      SmallVector<int64_t> strides = {1, 1};

      auto slicedType =
          RankedTensorType::get({batchSize, numElements}, elemType);
      auto sliceOp = stablehlo::SliceOp::create(
          rewriter, op.getLoc(), slicedType, adaptor.getPosition(),
          rewriter.getDenseI64ArrayAttr(startIndices),
          rewriter.getDenseI64ArrayAttr(limitIndices),
          rewriter.getDenseI64ArrayAttr(strides));

      // Reshape to [batchSize, ...originalShape...]
      auto reshapeOp = stablehlo::ReshapeOp::create(rewriter, op.getLoc(),
                                                    resultType, sliceOp);
      rewriter.replaceOp(op, reshapeOp.getResult());
    } else {
      return rewriter.notifyMatchFailure(op,
                                         "Position must be 1D or 2D tensor");
    }

    return success();
  }
};

struct LowerProbProgToStableHLOPass
    : public enzyme::impl::LowerProbProgToStableHLOPassBase<
          LowerProbProgToStableHLOPass> {
  using LowerProbProgToStableHLOPassBase::LowerProbProgToStableHLOPassBase;

  void runOnOperation() override {
    auto context = getOperation()->getContext();

    TypeConverter typeConverter;
    typeConverter.addConversion([](Type t) { return t; });

    typeConverter.addConversion([&](enzyme::TraceType t) {
      return RankedTensorType::get(
          {}, IntegerType::get(context, 64, IntegerType::Unsigned));
    });
    typeConverter.addConversion([&](enzyme::ConstraintType t) {
      return RankedTensorType::get(
          {}, IntegerType::get(context, 64, IntegerType::Unsigned));
    });
    typeConverter.addSourceMaterialization(
        [&](OpBuilder &builder, Type resultType, ValueRange inputs,
            Location loc) -> Value {
          return UnrealizedConversionCastOp::create(builder, loc, resultType,
                                                    inputs[0])
              .getResult(0);
        });
    typeConverter.addTargetMaterialization(
        [&](OpBuilder &builder, Type resultType, ValueRange inputs,
            Location loc) -> Value {
          return UnrealizedConversionCastOp::create(builder, loc, resultType,
                                                    inputs[0])
              .getResult(0);
        });

    ConversionTarget target(*context);

    target.addLegalDialect<stablehlo::StablehloDialect>();
    target.addLegalDialect<chlo::ChloDialect>();
    target.addLegalDialect<arith::ArithDialect>();
    target.addLegalDialect<func::FuncDialect>();
    target.addLegalDialect<tensor::TensorDialect>();
    target.addLegalDialect<enzyme::EnzymeDialect>();

    target.addIllegalOp<enzyme::RandomOp>();
    target.addIllegalOp<enzyme::RandomSplitOp>();
    target.addIllegalOp<enzyme::CholeskyOp>();
    target.addIllegalOp<enzyme::TriangularSolveOp>();
    target.addIllegalOp<enzyme::DotOp>();
    target.addIllegalOp<enzyme::LogAddExpOp>();
    target.addIllegalOp<enzyme::UnflattenSliceOp>();
    target.addIllegalOp<enzyme::ForLoopOp>();
    target.addIllegalOp<enzyme::WhileLoopOp>();
    target.addIllegalOp<enzyme::PopcountOp>();
    target.addIllegalOp<enzyme::DynamicExtractOp>();
    target.addIllegalOp<enzyme::DynamicUpdateOp>();

    target.addLegalOp<UnrealizedConversionCastOp>();

    RewritePatternSet patterns(context);

    patterns.add<
        RandomOpConversion, CholeskyOpConversion, TriangularSolveOpConversion,
        DotOpConversion, LogAddExpOpConversion, UnflattenSliceOpConversion,
        ForLoopOpConversion, WhileLoopOpConversion, PopcountOpConversion,
        DynamicExtractOpConversion, DynamicUpdateOpConversion>(
        backend, typeConverter, context);
    patterns.add<RandomSplitOpConversion>(backend, debugDump, typeConverter,
                                          context);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

struct LowerProbProgTraceOpsPass
    : public enzyme::impl::LowerProbProgTraceOpsPassBase<
          LowerProbProgTraceOpsPass> {
  using LowerProbProgTraceOpsPassBase::LowerProbProgTraceOpsPassBase;

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
    target.addIllegalOp<enzyme::GetFlattenedSamplesFromTraceOp>();
    target.addIllegalOp<enzyme::SelectOp>();
    target.addIllegalOp<enzyme::DumpOp>();

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

    patterns
        .add<InitTraceOpConversion, AddSampleToTraceOpConversion,
             AddSubtraceOpConversion, AddWeightToTraceOpConversion,
             AddRetvalToTraceOpConversion, GetSampleFromConstraintOpConversion,
             GetSubconstraintOpConversion, GetSampleFromTraceOpConversion,
             GetSubtraceOpConversion, GetWeightFromTraceOpConversion,
             GetFlattenedSamplesFromTraceOpConversion, SelectOpConversion,
             DumpOpConversion, UnrealizedConversionCastOpConversion>(
            backend, typeConverter, context);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
