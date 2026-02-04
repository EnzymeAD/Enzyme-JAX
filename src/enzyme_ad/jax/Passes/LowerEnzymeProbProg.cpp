#include "Enzyme/MLIR/Dialect/Dialect.h"
#include "Enzyme/MLIR/Dialect/Ops.h"
#include "mhlo/IR/hlo_ops.h"
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

struct SelectOpConversion : public OpConversionPattern<enzyme::SelectOp> {
  using OpConversionPattern::OpConversionPattern;

  std::string backend;
  SelectOpConversion(std::string backend, MLIRContext *context,
                     PatternBenefit benefit = 1)
      : OpConversionPattern(context, benefit), backend(backend) {}

  LogicalResult
  matchAndRewrite(enzyme::SelectOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<stablehlo::SelectOp>(
        op, adaptor.getTrueValue().getType(), adaptor.getCondition(),
        adaptor.getTrueValue(), adaptor.getFalseValue());
    return success();
  }
};

struct ReshapeOpConversion : public OpConversionPattern<enzyme::ReshapeOp> {
  using OpConversionPattern::OpConversionPattern;

  std::string backend;
  ReshapeOpConversion(std::string backend, MLIRContext *context,
                      PatternBenefit benefit = 1)
      : OpConversionPattern(context, benefit), backend(backend) {}

  LogicalResult
  matchAndRewrite(enzyme::ReshapeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultType = cast<RankedTensorType>(op.getResult().getType());
    rewriter.replaceOpWithNewOp<stablehlo::ReshapeOp>(op, resultType,
                                                      adaptor.getInput());
    return success();
  }
};

struct DumpOpConversion : public OpConversionPattern<enzyme::DumpOp> {
  using OpConversionPattern::OpConversionPattern;

  std::string backend;
  DumpOpConversion(std::string backend, MLIRContext *context,
                   PatternBenefit benefit = 1)
      : OpConversionPattern(context, benefit), backend(backend) {}

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

struct CholeskyOpConversion : public OpConversionPattern<enzyme::CholeskyOp> {
  using OpConversionPattern::OpConversionPattern;

  std::string backend;
  CholeskyOpConversion(std::string backend, MLIRContext *context,
                       PatternBenefit benefit = 1)
      : OpConversionPattern(context, benefit), backend(backend) {}

  LogicalResult
  matchAndRewrite(enzyme::CholeskyOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto input = adaptor.getInput();
    auto resultType = cast<RankedTensorType>(op.getResult().getType());

    rewriter.replaceOpWithNewOp<stablehlo::CholeskyOp>(
        op, resultType, input, rewriter.getBoolAttr(op.getLower()));
    return success();
  }
};

struct TriangularSolveOpConversion
    : public OpConversionPattern<enzyme::TriangularSolveOp> {
  using OpConversionPattern::OpConversionPattern;

  std::string backend;
  TriangularSolveOpConversion(std::string backend, MLIRContext *context,
                              PatternBenefit benefit = 1)
      : OpConversionPattern(context, benefit), backend(backend) {}

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
    default:
      return failure();
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
  DotOpConversion(std::string backend, MLIRContext *context,
                  PatternBenefit benefit = 1)
      : OpConversionPattern(context, benefit), backend(backend) {}

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

    rewriter.replaceOpWithNewOp<stablehlo::DotGeneralOp>(
        op, resultType, lhs, rhs, dotDimensionNumbers,
        /*precision_config=*/ArrayAttr(),
        /*algorithm=*/stablehlo::DotAlgorithmAttr());
    return success();
  }
};

// Reference:
// https://github.com/jax-ml/jax/blob/e9b487238f0cfe932200bae842d26826f19ba2bc/jax/_src/lax/other.py#L262
struct LogAddExpOpConversion : public OpConversionPattern<enzyme::LogAddExpOp> {
  using OpConversionPattern::OpConversionPattern;

  std::string backend;
  LogAddExpOpConversion(std::string backend, MLIRContext *context,
                        PatternBenefit benefit = 1)
      : OpConversionPattern(context, benefit), backend(backend) {}

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

struct LogisticOpConversion : public OpConversionPattern<enzyme::LogisticOp> {
  using OpConversionPattern::OpConversionPattern;

  std::string backend;
  LogisticOpConversion(std::string backend, MLIRContext *context,
                       PatternBenefit benefit = 1)
      : OpConversionPattern(context, benefit), backend(backend) {}

  LogicalResult
  matchAndRewrite(enzyme::LogisticOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto operand = adaptor.getOperand();
    auto resultType = cast<RankedTensorType>(op.getResult().getType());

    rewriter.replaceOpWithNewOp<stablehlo::LogisticOp>(op, resultType, operand);
    return success();
  }
};

struct RandomOpConversion : public OpConversionPattern<enzyme::RandomOp> {
  using OpConversionPattern::OpConversionPattern;

  std::string backend;
  RandomOpConversion(std::string backend, MLIRContext *context,
                     PatternBenefit benefit = 1)
      : OpConversionPattern(context, benefit), backend(backend) {}

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
      double lo = std::nextafter(-1.0, 0.0);
      double hi = 1.0;
      double range = hi - lo;
      auto rangeConst = stablehlo::ConstantOp::create(
          rewriter, op.getLoc(), rankedType,
          DenseElementsAttr::get(rankedType,
                                 rewriter.getFloatAttr(elemType, range)));
      auto loConst = stablehlo::ConstantOp::create(
          rewriter, op.getLoc(), rankedType,
          DenseElementsAttr::get(rankedType,
                                 rewriter.getFloatAttr(elemType, lo)));
      Value scaledUniform =
          stablehlo::MulOp::create(rewriter, op.getLoc(), rankedType,
                                   randUniform, rangeConst)
              .getResult();
      scaledUniform =
          stablehlo::AddOp::create(rewriter, op.getLoc(), rankedType,
                                   scaledUniform, loConst)
              .getResult();
      scaledUniform =
          stablehlo::MaxOp::create(rewriter, op.getLoc(), rankedType,
                                   scaledUniform, loConst)
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

      double lo = std::nextafter(-1.0, 0.0);
      double hi = 1.0;
      double range = hi - lo;
      auto rangeConst = stablehlo::ConstantOp::create(
          rewriter, op.getLoc(), vectorType,
          DenseElementsAttr::get(vectorType,
                                 rewriter.getFloatAttr(elemType, range)));
      auto loConst = stablehlo::ConstantOp::create(
          rewriter, op.getLoc(), vectorType,
          DenseElementsAttr::get(vectorType,
                                 rewriter.getFloatAttr(elemType, lo)));
      Value scaledUniform =
          stablehlo::MulOp::create(rewriter, op.getLoc(), vectorType,
                                   randUniform, rangeConst)
              .getResult();
      scaledUniform =
          stablehlo::AddOp::create(rewriter, op.getLoc(), vectorType,
                                   scaledUniform, loConst)
              .getResult();
      scaledUniform =
          stablehlo::MaxOp::create(rewriter, op.getLoc(), vectorType,
                                   scaledUniform, loConst)
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
                          MLIRContext *context, PatternBenefit benefit = 1)
      : OpConversionPattern(context, benefit), backend(backend),
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

struct ForLoopOpConversion : public OpConversionPattern<enzyme::ForLoopOp> {
  using OpConversionPattern::OpConversionPattern;

  std::string backend;
  ForLoopOpConversion(std::string backend, MLIRContext *context,
                      PatternBenefit benefit = 1)
      : OpConversionPattern(context, benefit), backend(backend) {}

  LogicalResult
  matchAndRewrite(enzyme::ForLoopOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Value> initVals = {adaptor.getLowerBound()};
    initVals.append(adaptor.getInitArgs().begin(), adaptor.getInitArgs().end());

    SmallVector<Type> loopTypes = {adaptor.getLowerBound().getType()};
    for (auto result : op.getResults())
      loopTypes.push_back(result.getType());

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
  WhileLoopOpConversion(std::string backend, MLIRContext *context,
                        PatternBenefit benefit = 1)
      : OpConversionPattern(context, benefit), backend(backend) {}

  LogicalResult
  matchAndRewrite(enzyme::WhileLoopOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Type> loopTypes;
    for (auto result : op.getResults())
      loopTypes.push_back(result.getType());

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

struct IfOpConversion : public OpConversionPattern<enzyme::IfOp> {
  using OpConversionPattern::OpConversionPattern;

  std::string backend;
  IfOpConversion(std::string backend, MLIRContext *context,
                 PatternBenefit benefit = 1)
      : OpConversionPattern(context, benefit), backend(backend) {}

  LogicalResult
  matchAndRewrite(enzyme::IfOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Type> resultTypes;
    for (auto result : op.getResults())
      resultTypes.push_back(result.getType());

    auto ifOp = stablehlo::IfOp::create(rewriter, op.getLoc(), resultTypes,
                                        adaptor.getPredicate());
    {
      Block *trueBlock = rewriter.createBlock(&ifOp.getTrueBranch());
      rewriter.setInsertionPointToStart(trueBlock);

      Block &origTrue = op.getTrueBranch().front();
      rewriter.mergeBlocks(&origTrue, trueBlock, {});
      auto trueYieldOp = cast<enzyme::YieldOp>(trueBlock->getTerminator());
      rewriter.setInsertionPoint(trueYieldOp);

      SmallVector<Value> trueYieldedVals;
      for (auto val : trueYieldOp.getOperands()) {
        Value remappedVal = rewriter.getRemappedValue(val);
        trueYieldedVals.push_back(remappedVal);
      }

      stablehlo::ReturnOp::create(rewriter, op.getLoc(), trueYieldedVals);
      rewriter.eraseOp(trueYieldOp);
    }
    {
      Block *falseBlock = rewriter.createBlock(&ifOp.getFalseBranch());
      rewriter.setInsertionPointToStart(falseBlock);

      Block &origFalse = op.getFalseBranch().front();
      rewriter.mergeBlocks(&origFalse, falseBlock, {});
      auto falseYieldOp = cast<enzyme::YieldOp>(falseBlock->getTerminator());
      rewriter.setInsertionPoint(falseYieldOp);

      SmallVector<Value> falseYieldedVals;
      for (auto val : falseYieldOp.getOperands()) {
        Value remappedVal = rewriter.getRemappedValue(val);
        falseYieldedVals.push_back(remappedVal);
      }

      stablehlo::ReturnOp::create(rewriter, op.getLoc(), falseYieldedVals);
      rewriter.eraseOp(falseYieldOp);
    }

    rewriter.replaceOp(op, ifOp.getResults());
    return success();
  }
};

struct PopcountOpConversion : public OpConversionPattern<enzyme::PopcountOp> {
  using OpConversionPattern::OpConversionPattern;

  std::string backend;
  PopcountOpConversion(std::string backend, MLIRContext *context,
                       PatternBenefit benefit = 1)
      : OpConversionPattern(context, benefit), backend(backend) {}

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

struct SliceOpConversion : public OpConversionPattern<enzyme::SliceOp> {
  using OpConversionPattern::OpConversionPattern;

  std::string backend;
  SliceOpConversion(std::string backend, MLIRContext *context,
                    PatternBenefit benefit = 1)
      : OpConversionPattern(context, benefit), backend(backend) {}

  LogicalResult
  matchAndRewrite(enzyme::SliceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultType = cast<RankedTensorType>(op.getResult().getType());
    rewriter.replaceOpWithNewOp<stablehlo::SliceOp>(
        op, resultType, adaptor.getOperand(), op.getStartIndices(),
        op.getLimitIndices(), op.getStrides());
    return success();
  }
};

struct DynamicSliceOpConversion
    : public OpConversionPattern<enzyme::DynamicSliceOp> {
  using OpConversionPattern::OpConversionPattern;

  std::string backend;
  DynamicSliceOpConversion(std::string backend, MLIRContext *context,
                           PatternBenefit benefit = 1)
      : OpConversionPattern(context, benefit), backend(backend) {}

  LogicalResult
  matchAndRewrite(enzyme::DynamicSliceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultType = cast<RankedTensorType>(op.getResult().getType());
    rewriter.replaceOpWithNewOp<stablehlo::DynamicSliceOp>(
        op, resultType, adaptor.getOperand(), adaptor.getStartIndices(),
        op.getSliceSizes());
    return success();
  }
};

struct DynamicUpdateSliceOpConversion
    : public OpConversionPattern<enzyme::DynamicUpdateSliceOp> {
  using OpConversionPattern::OpConversionPattern;

  std::string backend;
  DynamicUpdateSliceOpConversion(std::string backend, MLIRContext *context,
                                 PatternBenefit benefit = 1)
      : OpConversionPattern(context, benefit), backend(backend) {}

  LogicalResult
  matchAndRewrite(enzyme::DynamicUpdateSliceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultType = cast<RankedTensorType>(op.getResult().getType());
    auto newOp = stablehlo::DynamicUpdateSliceOp::create(
        rewriter, op.getLoc(), resultType, adaptor.getOperand(),
        adaptor.getUpdate(), adaptor.getStartIndices());
    rewriter.replaceOp(op, newOp.getResult());
    return success();
  }
};

struct LowerProbProgToStableHLOPass
    : public enzyme::impl::LowerProbProgToStableHLOPassBase<
          LowerProbProgToStableHLOPass> {
  using LowerProbProgToStableHLOPassBase::LowerProbProgToStableHLOPassBase;

  void runOnOperation() override {
    auto context = getOperation()->getContext();

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
    target.addIllegalOp<enzyme::LogisticOp>();
    target.addIllegalOp<enzyme::ForLoopOp>();
    target.addIllegalOp<enzyme::WhileLoopOp>();
    target.addIllegalOp<enzyme::IfOp>();
    target.addIllegalOp<enzyme::PopcountOp>();
    target.addIllegalOp<enzyme::SliceOp>();
    target.addIllegalOp<enzyme::DynamicSliceOp>();
    target.addIllegalOp<enzyme::DynamicUpdateSliceOp>();
    target.addIllegalOp<enzyme::SelectOp>();
    target.addIllegalOp<enzyme::ReshapeOp>();

    target.addLegalOp<UnrealizedConversionCastOp>();

    RewritePatternSet patterns(context);

    patterns.add<RandomOpConversion, CholeskyOpConversion,
                 TriangularSolveOpConversion, DotOpConversion,
                 LogAddExpOpConversion, LogisticOpConversion,
                 ForLoopOpConversion, WhileLoopOpConversion, IfOpConversion,
                 PopcountOpConversion, SliceOpConversion,
                 DynamicSliceOpConversion, DynamicUpdateSliceOpConversion,
                 SelectOpConversion, ReshapeOpConversion>(backend, context);
    patterns.add<RandomSplitOpConversion>(backend, debugDump, context);

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

    ConversionTarget target(*context);

    target.addLegalDialect<LLVM::LLVMDialect>();
    target.addLegalDialect<func::FuncDialect>();
    target.addLegalDialect<enzymexla::EnzymeXLADialect>();
    target.addLegalDialect<stablehlo::StablehloDialect>();

    target.addIllegalOp<enzyme::DumpOp>();

    RewritePatternSet patterns(context);

    patterns.add<DumpOpConversion>(context);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
