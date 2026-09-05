//===- StablehloToLapack.cpp - stablehlo linalg -> LAPACK ops ------------===//

#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "src/enzyme_ad/jax/Dialect/Dialect.h"
#include "src/enzyme_ad/jax/Dialect/Ops.h"
#include "src/enzyme_ad/jax/Passes/Passes.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_STABLEHLOTOLAPACKPASS
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

using namespace mlir;

namespace {

struct CholeskyToPotrf : public OpRewritePattern<stablehlo::CholeskyOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(stablehlo::CholeskyOp op,
                                PatternRewriter &rewriter) const override {
    auto ty = dyn_cast<RankedTensorType>(op.getType());
    if (!ty || ty.getRank() != 2 || !ty.hasStaticShape())
      return failure();
    Type elemType = ty.getElementType();
    if (!elemType.isF32() && !elemType.isF64())
      return failure();

    Location loc = op.getLoc();
    bool lower = op.getLower();
    auto infoType = RankedTensorType::get({}, rewriter.getI64Type());
    auto potrf = enzymexla::PotrfOp::create(
        rewriter, loc, TypeRange{ty, infoType}, op.getA(),
        enzymexla::LapackUploAttr::get(rewriter.getContext(),
                                       lower ? enzymexla::LapackUplo::L
                                             : enzymexla::LapackUplo::U));

    // potrf leaves the input in the opposite triangle; cholesky expects zeros.
    auto idxType = RankedTensorType::get(ty.getShape(), rewriter.getI64Type());
    Value rows = stablehlo::IotaOp::create(rewriter, loc, idxType, 0);
    Value cols = stablehlo::IotaOp::create(rewriter, loc, idxType, 1);
    Value keep = stablehlo::CompareOp::create(
        rewriter, loc, rows, cols,
        lower ? stablehlo::ComparisonDirection::GE
              : stablehlo::ComparisonDirection::LE);
    Value zero = stablehlo::ConstantOp::create(
        rewriter, loc, ty, cast<ElementsAttr>(rewriter.getZeroAttr(ty)));
    rewriter.replaceOpWithNewOp<stablehlo::SelectOp>(op, keep,
                                                     potrf.getResult(0), zero);
    return success();
  }
};

struct TriangularSolveToTrsm
    : public OpRewritePattern<stablehlo::TriangularSolveOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(stablehlo::TriangularSolveOp op,
                                PatternRewriter &rewriter) const override {
    auto ty = dyn_cast<RankedTensorType>(op.getType());
    if (!ty || ty.getRank() != 2 || !ty.hasStaticShape())
      return failure();
    Type elemType = ty.getElementType();
    if (!elemType.isF32() && !elemType.isF64())
      return failure();

    enzymexla::LapackTranspose transa;
    switch (op.getTransposeA()) {
    case stablehlo::Transpose::NO_TRANSPOSE:
      transa = enzymexla::LapackTranspose::none;
      break;
    case stablehlo::Transpose::TRANSPOSE:
      transa = enzymexla::LapackTranspose::transpose;
      break;
    case stablehlo::Transpose::ADJOINT:
      transa = enzymexla::LapackTranspose::adjoint;
      break;
    default:
      return failure();
    }

    Location loc = op.getLoc();
    auto scalarType = RankedTensorType::get({}, elemType);
    Value alpha = stablehlo::ConstantOp::create(
        rewriter, loc, scalarType,
        DenseElementsAttr::get(scalarType,
                               rewriter.getFloatAttr(elemType, 1.0)));
    rewriter.replaceOpWithNewOp<enzymexla::TrsmOp>(
        op, ty, alpha, op.getA(), op.getB(),
        enzymexla::LapackSideAttr::get(rewriter.getContext(),
                                       op.getLeftSide()
                                           ? enzymexla::LapackSide::left
                                           : enzymexla::LapackSide::right),
        enzymexla::LapackUploAttr::get(
            rewriter.getContext(), op.getLower() ? enzymexla::LapackUplo::L
                                                 : enzymexla::LapackUplo::U),
        enzymexla::LapackTransposeAttr::get(rewriter.getContext(), transa),
        op.getUnitDiagonal() ? rewriter.getUnitAttr() : nullptr);
    return success();
  }
};

struct StablehloToLapackPass
    : public enzyme::impl::StablehloToLapackPassBase<StablehloToLapackPass> {
  using Base::Base;

  void runOnOperation() override {
    if (backend != "cpu")
      return;
    RewritePatternSet patterns(&getContext());
    patterns.add<CholeskyToPotrf, TriangularSolveToTrsm>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace
