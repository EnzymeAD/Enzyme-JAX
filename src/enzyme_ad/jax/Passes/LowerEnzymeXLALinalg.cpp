#include "Enzyme/MLIR/Dialect/Dialect.h"
#include "Enzyme/MLIR/Passes/EnzymeBatchPass.h"
#include "mhlo/IR/hlo_ops.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "src/enzyme_ad/jax/Dialect/Dialect.h"
#include "src/enzyme_ad/jax/Dialect/Ops.h"
#include "src/enzyme_ad/jax/Passes/LinalgUtils.h"
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
#include <limits>
#include <numeric>

#define DEBUG_TYPE "lower-enzymexla-linalg"

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_LOWERENZYMEXLALINALGPASS
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

using namespace mlir;
using namespace mlir::enzyme;
using namespace mlir::enzymexla;

struct LUFactorizationOpLowering
    : public OpRewritePattern<enzymexla::LUFactorizationOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(enzymexla::LUFactorizationOp op,
                                PatternRewriter &rewriter) const override {
    auto getrfOp = enzymexla::GetrfOp::create(
        rewriter, op.getLoc(), op->getResultTypes(), op.getInput());
    rewriter.replaceOp(op, getrfOp);
    return success();
  }
};

struct QRFactorizationOpLowering
    : public OpRewritePattern<enzymexla::QRFactorizationOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(enzymexla::QRFactorizationOp op,
                                PatternRewriter &rewriter) const override {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(op);

    auto geqrfOp = enzymexla::GeqrfOp::create(
        rewriter, op.getLoc(), op->getResultTypes(), op.getInput());
    auto R = geqrfOp.getOutput();
    auto tau = geqrfOp.getTau();
    auto info = geqrfOp.getInfo();

    auto orgqrOp = enzymexla::OrgqrOp::create(
        rewriter, op.getLoc(), op->getResult(0).getType(), R, tau);
    auto Q = orgqrOp.getOutput();

    rewriter.replaceAllOpUsesWith(op, ValueRange{Q, R, info});

    return success();
  }
};

struct SVDFactorizationOpLowering
    : public OpRewritePattern<enzymexla::SVDFactorizationOp> {
  std::string backend;

  SVDFactorizationOpLowering(std::string backend, MLIRContext *context)
      : OpRewritePattern(context), backend(backend) {}

  LogicalResult matchAndRewrite(enzymexla::SVDFactorizationOp op,
                                PatternRewriter &rewriter) const override {
    SVDAlgorithm algorithm = op.getAlgorithm();
    if (algorithm == SVDAlgorithm::DEFAULT) {
      if (backend == "cpu") {
        algorithm = SVDAlgorithm::DivideAndConquer;
      } else if (backend == "cuda" || backend == "tpu" || backend == "rocm") {
        algorithm = SVDAlgorithm::Jacobi;
      } else {
        op->emitOpError() << "Unsupported backend: " << backend;
        return failure();
      }
    }

    bool computeUv =
        !op.getResult(0).use_empty() || !op.getResult(2).use_empty();

    switch (algorithm) {
    case SVDAlgorithm::QRIteration: {
      auto gesddOp = enzymexla::GesvdOp::create(
          rewriter, op.getLoc(), op->getResultTypes(), op.getInput(),
          /*full=*/op.getFull(), computeUv);
      rewriter.replaceOp(op, gesddOp);
      break;
    }
    case SVDAlgorithm::DivideAndConquer: {
      auto gesvjOp = enzymexla::GesddOp::create(
          rewriter, op.getLoc(), op->getResultTypes(), op.getInput(),
          /*full=*/op.getFull(), computeUv);
      rewriter.replaceOp(op, gesvjOp);
      break;
    }
    case SVDAlgorithm::Jacobi: {
      auto gesvdOp = enzymexla::GesvjOp::create(
          rewriter, op.getLoc(), op->getResultTypes(), op.getInput(),
          /*full=*/op.getFull(), computeUv);
      rewriter.replaceOp(op, gesvdOp);
      break;
    }
    case SVDAlgorithm::DEFAULT:
      llvm_unreachable("Default should have already been handled");
    }

    return success();
  }
};

struct TridiagonalSolveOpLowering
    : public OpRewritePattern<enzymexla::TridiagonalSolveOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(enzymexla::TridiagonalSolveOp op,
                                PatternRewriter &rewriter) const override {
    auto bType = cast<RankedTensorType>(op.getB().getType());
    auto bRank = bType.getRank();
    auto batchShape = bType.getShape().drop_back(2);

    auto infoType = RankedTensorType::get(batchShape, rewriter.getI32Type());
    auto gtsvOp = enzymexla::GtsvOp::create(
        rewriter, op.getLoc(), TypeRange{bType, infoType}, op.getDl(),
        op.getD(), op.getDu(), op.getB());

    // https://github.com/jax-ml/jax/blob/9b99988ae92f408251459b9a4e8249b7ac58b42d/jax/_src/lax/linalg.py#L2977-L2994
    // systems that failed to solve are replaced by NaNs
    auto ok = stablehlo::CompareOp::create(
        rewriter, op.getLoc(), gtsvOp.getInfo(),
        stablehlo::ConstantOp::create(
            rewriter, op.getLoc(), infoType,
            cast<ElementsAttr>(makeAttr(infoType, 0))),
        stablehlo::ComparisonDirection::EQ);

    SmallVector<int64_t> broadcastDims(bRank - 2);
    std::iota(broadcastDims.begin(), broadcastDims.end(), 0);
    auto okBroadcast = stablehlo::BroadcastInDimOp::create(
        rewriter, op.getLoc(),
        RankedTensorType::get(bType.getShape(), rewriter.getI1Type()), ok,
        rewriter.getDenseI64ArrayAttr(broadcastDims));

    auto nan = stablehlo::ConstantOp::create(
        rewriter, op.getLoc(), bType,
        cast<ElementsAttr>(
            makeAttr(bType, std::numeric_limits<double>::quiet_NaN())));

    rewriter.replaceOpWithNewOp<stablehlo::SelectOp>(op, okBroadcast,
                                                     gtsvOp.getX(), nan);
    return success();
  }
};

struct LowerEnzymeXLALinalgPass
    : public enzyme::impl::LowerEnzymeXLALinalgPassBase<
          LowerEnzymeXLALinalgPass> {
  using Base::Base;

  void runOnOperation() override {
    auto context = getOperation()->getContext();
    RewritePatternSet patterns(context);

    patterns.add<LUFactorizationOpLowering, QRFactorizationOpLowering,
                 TridiagonalSolveOpLowering>(context);
    patterns.add<SVDFactorizationOpLowering>(backend, context);

    GreedyRewriteConfig config;
    config.enableFolding();
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns),
                                     config))) {
      signalPassFailure();
    }

    // Verify that all illegal ops have been lowered
    auto walkResult = getOperation()->walk([&](Operation *op) {
      if (isa<enzymexla::LUFactorizationOp, enzymexla::QRFactorizationOp,
              enzymexla::SVDFactorizationOp, enzymexla::TridiagonalSolveOp>(
              op)) {
        op->emitError("Failed to lower enzymexla linalg operation");
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });

    if (walkResult.wasInterrupted()) {
      signalPassFailure();
    }
  }
};
