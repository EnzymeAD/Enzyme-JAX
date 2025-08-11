#include "mhlo/IR/hlo_ops.h"
#include "mhlo/transforms/map_mhlo_to_scalar_op.h"
#include "mhlo/transforms/rewriters.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/DynamicAPInt.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/MathExtras.h"
#include <algorithm>
#include <cstdint>

#include "src/enzyme_ad/jax/Dialect/Dialect.h"
#include "src/enzyme_ad/jax/Dialect/Ops.h"
#include "src/enzyme_ad/jax/Passes/Passes.h"
#include "src/enzyme_ad/jax/Utils.h"

#include "mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#define DEBUG_TYPE "enzymexla-stablehlo-to-triton-compatible-dialect"

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_STABLEHLOTOTRITONCOMPATIBLEDIALECT
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

using namespace mlir;
using namespace mlir::enzyme;

bool isPowerOfTwo(uint64_t n) { return (n & (n - 1)) == 0; }

struct ConstantOpToArithmetic : public OpRewritePattern<stablehlo::ConstantOp> {
  using OpRewritePattern<stablehlo::ConstantOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(stablehlo::ConstantOp op,
                                PatternRewriter &rewriter) const override {
    auto denseAttr = dyn_cast<DenseElementsAttr>(op.getValue());
    if (!denseAttr || !denseAttr.isSplat())
      return failure();

    auto splatValue = denseAttr.getSplatValue<Attribute>();
    if (!splatValue)
      return failure();

    auto elementType = cast<RankedTensorType>(op.getType()).getElementType();
    if (dyn_cast<ComplexType>(elementType))
      return failure();

    rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, op.getType(), denseAttr);
    return success();
  }
};

void replaceWithTritonMakeRange(PatternRewriter &rewriter, uint32_t start,
                                uint32_t end, Type resultElementType,
                                Operation *op) {
  auto rangeOp = rewriter.create<triton::MakeRangeOp>(
      op->getLoc(),
      RankedTensorType::get({static_cast<int64_t>(end - start)},
                            rewriter.getI32Type()),
      start, end);
  if (resultElementType.getIntOrFloatBitWidth() == 32) {
    rewriter.replaceOp(op, rangeOp.getResult());
  } else {
    rewriter.replaceOpWithNewOp<stablehlo::ConvertOp>(
        op, op->getResult(0).getType(), rangeOp.getResult());
  }
  return;
}

struct ConstantOpToTritonMakeRange
    : public OpRewritePattern<stablehlo::ConstantOp> {
  using OpRewritePattern<stablehlo::ConstantOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(stablehlo::ConstantOp op,
                                PatternRewriter &rewriter) const override {
    auto tensorType = cast<RankedTensorType>(op.getType());
    if (tensorType.getRank() != 1)
      return failure();

    // Get the DenseElementsAttr value of the constant
    auto denseAttr = dyn_cast<DenseElementsAttr>(op.getValue());
    if (!denseAttr)
      return failure();

    // Ensure that the values are integers (we assume tt.make_range works for
    // integers)
    auto elementType = tensorType.getElementType();
    if (!isa<IntegerType>(elementType) ||
        dyn_cast<IntegerType>(elementType).isSigned())
      return failure();

    SmallVector<uint32_t> values;
    for (auto value : denseAttr.getValues<APInt>())
      values.push_back(value.getZExtValue());

    // Check if it's a contiguous range; e.g., [0, 1, 2, ..., n-1]
    for (size_t i = 1; i < values.size(); ++i) {
      if (values[i] != values[i - 1] + 1)
        return failure();
    }

    if (!isPowerOfTwo(values.size()))
      return failure();

    replaceWithTritonMakeRange(rewriter, values[0],
                               values[values.size() - 1] + 1, elementType, op);
    return success();
  }
};

struct IotaOpToTritonMakeRange : public OpRewritePattern<stablehlo::IotaOp> {
  using OpRewritePattern<stablehlo::IotaOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(stablehlo::IotaOp op,
                                PatternRewriter &rewriter) const override {
    auto tensorType = cast<RankedTensorType>(op.getType());
    auto length = tensorType.getDimSize(0);
    if (tensorType.getRank() != 1)
      return failure();

    auto elementType = tensorType.getElementType();
    if (!isa<IntegerType>(elementType) ||
        dyn_cast<IntegerType>(elementType).isSigned())
      return failure();

    if (!isPowerOfTwo(length))
      return failure();

    replaceWithTritonMakeRange(rewriter, 0, length, elementType, op);
    return success();
  }
};

struct ConvertOpToArithmetic : public OpRewritePattern<stablehlo::ConvertOp> {
  using OpRewritePattern<stablehlo::ConvertOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(stablehlo::ConvertOp op,
                                PatternRewriter &rewriter) const override {
    auto inputType = op.getOperand().getType();
    auto outputType = op.getType();

    auto inputElementType = getElementTypeOrSelf(inputType);
    auto outputElementType = getElementTypeOrSelf(outputType);

    auto inputBitWidth = inputElementType.getIntOrFloatBitWidth();
    auto outputBitWidth = outputElementType.getIntOrFloatBitWidth();

    if (isa<FloatType>(inputElementType) && isa<FloatType>(outputElementType)) {
      if (inputBitWidth == outputBitWidth) {
        rewriter.replaceOp(op, op.getOperand());
      } else if (inputBitWidth > outputBitWidth) {
        rewriter.replaceOpWithNewOp<arith::TruncFOp>(op, outputType,
                                                     op.getOperand());
      } else {
        rewriter.replaceOpWithNewOp<arith::ExtFOp>(op, outputType,
                                                   op.getOperand());
      }
      return success();
    }

    if (isa<IntegerType>(inputElementType) &&
        isa<IntegerType>(outputElementType)) {
      if (inputBitWidth == outputBitWidth) {
        rewriter.replaceOp(op, op.getOperand());
      } else if (inputBitWidth > outputBitWidth) {
        rewriter.replaceOpWithNewOp<arith::TruncIOp>(op, outputType,
                                                     op.getOperand());
      } else {
        if (cast<IntegerType>(outputElementType).isSigned()) {
          rewriter.replaceOpWithNewOp<arith::ExtSIOp>(op, outputType,
                                                      op.getOperand());
        } else {
          rewriter.replaceOpWithNewOp<arith::ExtUIOp>(op, outputType,
                                                      op.getOperand());
        }
      }
      return success();
    }

    return failure();
  }
};

struct BcastInDimToTritonBroadcast
    : public OpRewritePattern<stablehlo::BroadcastInDimOp> {
  using OpRewritePattern<stablehlo::BroadcastInDimOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(stablehlo::BroadcastInDimOp op,
                                PatternRewriter &rewriter) const override {
    // bcast_in_dim -> tt.broadcast(tt.trans(tt.reshape))
    auto loc = op.getLoc();

    auto input = op.getOperand();
    auto broadcastDims = op.getBroadcastDimensions();
    auto inputType = cast<RankedTensorType>(input.getType());
    auto inputShape = inputType.getShape();
    auto outputType = cast<RankedTensorType>(op.getType());
    auto outputShape = outputType.getShape();

    auto inputElementType = getElementTypeOrSelf(inputType);
    SmallVector<int64_t> reshapeOutputShape;
    for (auto dim : inputShape)
      reshapeOutputShape.push_back(dim);
    for (int i = 0; i < outputType.getRank() - inputType.getRank(); ++i)
      reshapeOutputShape.push_back(1);
    auto reshapeOutputType =
        RankedTensorType::get(reshapeOutputShape, inputElementType);

    auto reshapeOp =
        rewriter.create<triton::ReshapeOp>(loc, reshapeOutputType, input);

    SmallVector<int32_t> permutation(outputType.getRank(), -1);
    int32_t mapped = broadcastDims.size();
    for (int i = 0; i < outputType.getRank(); ++i) {
      auto found = std::find(broadcastDims.begin(), broadcastDims.end(), i);
      if (found != broadcastDims.end()) {
        permutation[i] = std::distance(broadcastDims.begin(), found);
      } else {
        permutation[i] = mapped++;
      }
    }
    auto transposeOp = rewriter.create<triton::TransOp>(
        loc, reshapeOp, rewriter.getDenseI32ArrayAttr(permutation));

    rewriter.replaceOpWithNewOp<triton::BroadcastOp>(op, outputType,
                                                     transposeOp);
    return success();
  }
};

struct StableHLOToTritonCompatibleDialectPass
    : public enzyme::impl::StableHLOToTritonCompatibleDialectBase<
          StableHLOToTritonCompatibleDialectPass> {
  using Base::Base;

  void runOnOperation() override {
    auto context = getOperation()->getContext();
    RewritePatternSet patterns(context);

    // Arithmetic dialect
    patterns.add<ConstantOpToArithmetic, ConvertOpToArithmetic>(context);

    // Math dialect

    // Control flow dialect

    // Structured control flow dialect

    // Triton dialect
    patterns.add<ConstantOpToTritonMakeRange, IotaOpToTritonMakeRange,
                 BcastInDimToTritonBroadcast>(context);

    // XXX: Reuse upstream conversion patterns from xla. Debug segfaults?
    // We can't directly use the pass since those are restricted to func.func
    // auto linalgTypeConverter = mhlo::createHloToLinalgTypeConverter();
    // mhlo::populateHloToLinalgConversionPattern(context, *linalgTypeConverter,
    //                                            &patterns, false);

    GreedyRewriteConfig config;
    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns),
                                            config))) {
      signalPassFailure();
    }
  }
};
