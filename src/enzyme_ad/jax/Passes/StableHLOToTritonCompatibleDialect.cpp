#include "mhlo/IR/hlo_ops.h"
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

    GreedyRewriteConfig config;
    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns),
                                            config))) {
      signalPassFailure();
    }
  }
};
