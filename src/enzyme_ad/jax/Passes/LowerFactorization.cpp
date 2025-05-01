#include "mhlo/IR/hlo_ops.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
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

#define DEBUG_TYPE "lower-factorization"

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_LOWERFACTORIZATIONPASS
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

struct LUFactorizationOpLowering
    : public OpRewritePattern<enzymexla::LUFactorizationOp> {

  std::string backend;
  LUFactorizationOpLowering(std::string backend, MLIRContext *context,
                            PatternBenefit benefit = 1)
      : OpRewritePattern(context, benefit), backend(backend) {}

  LogicalResult matchAndRewrite(enzymexla::LUFactorizationOp op,
                                PatternRewriter &rewriter) const override {
    auto input = op.getOperand();
    auto inputShape = cast<RankedTensorType>(input.getType()).getShape();
    auto inputRank = inputShape.size();
    auto inputElementType =
        cast<RankedTensorType>(input.getType()).getElementType();

    const int64_t m = inputShape[inputRank - 2];
    const int64_t n = inputShape[inputRank - 1];
    const int64_t numBatchDims = inputRank - 2;
    auto inputType = input.getType();

    auto indexType =
        cast<RankedTensorType>(op.getResult(1).getType()).getElementType();

    auto infoType = RankedTensorType::get({}, indexType);

    SmallVector<int64_t> pivotShape;
    for (int i = 0; i < numBatchDims; i++) {
      pivotShape.push_back(inputShape[i]);
    }
    pivotShape.push_back(std::min(m, n));
    auto pivotType = RankedTensorType::get(pivotShape, indexType);

    SmallVector<int64_t> permutationShape;
    for (int i = 0; i < numBatchDims; i++) {
      permutationShape.push_back(inputShape[i]);
    }
    permutationShape.push_back(m);
    auto permutationType = RankedTensorType::get(permutationShape, indexType);

    if (numBatchDims > 0 && (backend == "cuda" || backend == "cpu")) {
      // TODO: Implement batched LU factorizations???
      return rewriter.notifyMatchFailure(
          op,
          "Batched LU factorizations not yet implemented for " + backend + ".");
    }

    if (backend == "cpu") {
      return rewriter.notifyMatchFailure(
          op, "CPU backend lowering not yet implemented.");
    } else if (backend == "cuda") {
      return rewriter.notifyMatchFailure(
          op, "CUDA backend lowering not yet implemented.");
    } else if (backend == "tpu") {
      // TPU returns (LU, pivots, permutation). info isn't returned. based on
      // how JAX operates, I am assuming info = 0 when there is a nan in the
      // output.
      auto customCall = rewriter.create<stablehlo::CustomCallOp>(
          op.getLoc(), TypeRange{inputType, pivotType, permutationType},
          ValueRange{input}, rewriter.getStringAttr("LUFactorization"),
          /*has_side_effect*/ nullptr,
          /*backend_config*/ nullptr,
          /*api_version*/ nullptr,
          /*calledcomputations*/ nullptr,
          /*operand_layouts*/ nullptr,
          /*result_layouts*/ nullptr,
          /*output_operand_aliases*/ nullptr);

      // LAPACK returns 1-indexed pivots, while XLA returns 0-indexed pivots. We
      // make it consistent with LAPACK by adding 1 to the pivots.
      auto pivots1Indexed = rewriter.create<stablehlo::AddOp>(
          op.getLoc(),
          rewriter.create<stablehlo::ConstantOp>(
              op.getLoc(), pivotType,
              cast<ElementsAttr>(makeAttr(pivotType, 1))),
          customCall.getResult(1));

      auto isFinite = rewriter.create<stablehlo::IsFiniteOp>(
          op.getLoc(), customCall.getResult(0));

      SmallVector<int64_t> reductionDims;
      for (int i = 0; i < inputRank; i++) {
        reductionDims.push_back(i);
      }
      auto initValType = RankedTensorType::get({}, rewriter.getI1Type());
      auto initVal = rewriter.create<stablehlo::ConstantOp>(
          op.getLoc(), initValType,
          cast<ElementsAttr>(makeAttr(initValType, 1)));

      auto allFinite = rewriter.create<stablehlo::ReduceOp>(
          op.getLoc(), initValType, ValueRange{isFinite.getResult()},
          ValueRange{initVal}, rewriter.getDenseI64ArrayAttr(reductionDims));

      {
        OpBuilder::InsertionGuard guard(rewriter);
        auto &region = allFinite.getBody();
        auto *block =
            rewriter.createBlock(&region, {}, {initValType, initValType},
                                 {op.getLoc(), op.getLoc()});

        rewriter.setInsertionPointToStart(block);
        auto lhs = block->getArgument(0);
        auto rhs = block->getArgument(1);
        auto andOp = rewriter.create<stablehlo::AndOp>(op.getLoc(), lhs, rhs);

        rewriter.create<stablehlo::ReturnOp>(op.getLoc(),
                                             ValueRange{andOp.getResult()});
      }

      auto info = rewriter.create<stablehlo::ConvertOp>(op.getLoc(), infoType,
                                                        allFinite.getResult(0));

      rewriter.replaceAllUsesWith(op.getResult(0), customCall.getResult(0));
      rewriter.replaceAllUsesWith(op.getResult(1), pivots1Indexed);
      rewriter.replaceAllUsesWith(op.getResult(2), info);
      rewriter.eraseOp(op);

      return success();
    } else {
      return rewriter.notifyMatchFailure(op, "Unknown backend " + backend);
    }
  }
};

struct LowerFactorizationPass
    : public enzyme::impl::LowerFactorizationPassBase<LowerFactorizationPass> {
  using Base::Base;

  void runOnOperation() override {
    auto context = getOperation()->getContext();
    RewritePatternSet patterns(context);

    patterns.add<LUFactorizationOpLowering>(backend, context);

    GreedyRewriteConfig config;
    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns),
                                            config))) {
      signalPassFailure();
    }
  }
};
