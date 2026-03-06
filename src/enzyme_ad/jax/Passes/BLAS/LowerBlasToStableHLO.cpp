#include "src/enzyme_ad/jax/Dialect/BLAS/Dialect.h"
#include "src/enzyme_ad/jax/Dialect/BLAS/Utils.h"
#include "src/enzyme_ad/jax/Dialect/Dialect.h"
#include "src/enzyme_ad/jax/Dialect/Ops.h"
#include "src/enzyme_ad/jax/Passes/BLAS/Passes.h"

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "src/enzyme_ad/jax/Utils.h"
#include "stablehlo/dialect/StablehloOps.h"

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallVector.h"
#include <mlir/IR/BuiltinAttributes.h>

#define DEBUG_TYPE "lower-blas-to-stablehlo"

namespace mlir::blas {
#define GEN_PASS_DEF_LOWERBLASTOSTABLEHLOPASS
#include "src/enzyme_ad/jax/Passes/BLAS/Passes.h.inc"
} // namespace mlir::blas

using namespace mlir;
// using namespace mlir::enzyme;
namespace blas = mlir::blas;
using namespace mlir::blas;
using namespace mlir::stablehlo;

struct SymmOpLowering : public OpRewritePattern<blas::SymmOp> {

  using OpRewritePattern<blas::SymmOp>::OpRewritePattern;

  std::string backend;
  int64_t blasIntWidth;
  SymmOpLowering(std::string backend, int64_t blasIntWidth,
                 MLIRContext *context, PatternBenefit benefit = 1)
      : OpRewritePattern(context, benefit), backend(backend),
        blasIntWidth(blasIntWidth) {}

  LogicalResult matchAndRewrite(blas::SymmOp op,
                                PatternRewriter &rewriter) const override {
    auto AType = cast<RankedTensorType>(op.getA().getType());
    auto nBatchDims = AType.getRank() - 2;
    SmallVector<int64_t> batchDims(nBatchDims, 0);
    std::iota(batchDims.begin(), batchDims.end(), 0);

    Value A = op.getA();
    if (!stablehlo::IsTensorFilled(A)) {
      // If the tensor is not filled, we copy to the non-uplo region for safety
      A = stablehlo::copyTriangularPart(rewriter, A, op.getUplo());
      if (!A) {
        return failure();
      }
    }

    // fallback to emitting a stablehlo.dot_general that computes:
    //   alpha*A*B + beta*C if side = 'L'
    //   alpha*B*A + beta*C if side = 'R'
    stablehlo::DotDimensionNumbersAttr dotDims;
    dotDims = stablehlo::DotDimensionNumbersAttr::get(
        op.getContext(), batchDims, batchDims, {nBatchDims + 1}, {nBatchDims});

    stablehlo::DotGeneralOp dotGeneralOp;
    if (op.getSide() == BlasSide::left) {
      dotGeneralOp = stablehlo::DotGeneralOp::create(
          rewriter, op.getLoc(), cast<RankedTensorType>(op.getC().getType()),
          op.getA(), op.getB(), dotDims, nullptr, nullptr);
    } else {
      dotGeneralOp = stablehlo::DotGeneralOp::create(
          rewriter, op.getLoc(), cast<RankedTensorType>(op.getC().getType()),
          op.getB(), op.getA(), dotDims, nullptr, nullptr);
    }

    auto mul0 = stablehlo::MulOpCreate(rewriter, op->getLoc(), op.getAlpha(),
                                       dotGeneralOp);
    auto mul1 =
        stablehlo::MulOpCreate(rewriter, op->getLoc(), op.getBeta(), op.getC());
    auto res = stablehlo::AddOpCreate(rewriter, op->getLoc(), mul0, mul1);
    rewriter.replaceOp(op, res);
    return success();
  }
};

struct SyrkOpLowering : public OpRewritePattern<blas::SyrkOp> {
  using OpRewritePattern<blas::SyrkOp>::OpRewritePattern;

  SyrkOpLowering(std::string backend, int64_t blasIntWidth,
                 MLIRContext *context, PatternBenefit benefit = 1)
      : OpRewritePattern(context, benefit), backend(backend),
        blasIntWidth(blasIntWidth){};

  LogicalResult matchAndRewrite(blas::SyrkOp op,
                                PatternRewriter &rewriter) const override {
    auto AType = cast<RankedTensorType>(op.getA().getType());
    auto nBatchDims = AType.getRank() - 2;
    SmallVector<int64_t> batchDims(nBatchDims, 0);
    std::iota(batchDims.begin(), batchDims.end(), 0);

    Value C = op.getC();
    if (!stablehlo::IsTensorFilled(C)) {
      // If the tensor is not filled, we copy to the non-uplo region for safety
      C = stablehlo::copyTriangularPart(rewriter, C, op.getUplo());
      if (!C) {
        return failure();
      }
    }

    // fallback to emitting a stablehlo.dot_general that computes:
    //   alpha * A * A^T + beta * C
    //   alpha * A^T * A + beta * C
    stablehlo::DotDimensionNumbersAttr dotDims;
    switch (op.getTranspose()) {
    case BlasTranspose::none:
      dotDims = stablehlo::DotDimensionNumbersAttr::get(
          op.getContext(), batchDims, batchDims, {nBatchDims + 1},
          {nBatchDims + 1});
      break;
    case BlasTranspose::adjoint:
      LLVM_FALLTHROUGH;
    case BlasTranspose::transpose:
      dotDims = stablehlo::DotDimensionNumbersAttr::get(
          op.getContext(), batchDims, batchDims, {nBatchDims}, {nBatchDims});
      break;
    }

    auto AAT = stablehlo::DotGeneralOp::create(
        rewriter, op.getLoc(), cast<RankedTensorType>(op.getC().getType()),
        op.getA(), op.getA(), dotDims, nullptr, nullptr);

    auto aop =
        stablehlo::MulOpCreate(rewriter, op->getLoc(), op.getAlpha(), AAT);
    auto bop = stablehlo::MulOpCreate(rewriter, op->getLoc(), op.getBeta(), C);

    auto res = stablehlo::AddOpCreate(rewriter, op->getLoc(), aop, bop);
    rewriter.replaceOp(op, res);
    return success();
  }

private:
  std::string backend;
  int64_t blasIntWidth;
};

struct TrsmOpLowering : public OpRewritePattern<blas::TrsmOp> {
  TrsmOpLowering(MLIRContext *context, PatternBenefit benefit = 1)
      : OpRewritePattern(context, benefit) {}

  LogicalResult matchAndRewrite(blas::TrsmOp op,
                                PatternRewriter &rewriter) const override {
    auto ctx = op->getContext();
    LLVMTypeConverter typeConverter(ctx);

    auto alpha = op.getOperand(0);
    auto A = op.getOperand(1);
    auto B = op.getOperand(2);
    auto type_alpha = cast<RankedTensorType>(alpha.getType());
    auto type_A = cast<RankedTensorType>(A.getType());
    auto type_B = cast<RankedTensorType>(B.getType());
    auto type_element = type_alpha.getElementType();

    // (C1)
    if (type_A.getElementType() != type_element ||
        type_B.getElementType() != type_element) {
      return rewriter.notifyMatchFailure(
          op, "Element types of alpha, A and B must match");
    }
    auto rank = type_A.getRank();

    // (C2)
    if (type_A.getRank() != type_B.getRank()) {
      return rewriter.notifyMatchFailure(op, "Ranks of A and B must match");
    }

    // (C3)
    auto shape_A = type_A.getShape();
    auto shape_B = type_B.getShape();
    if (shape_A.drop_back(2) != shape_B.drop_back(2)) {
      return rewriter.notifyMatchFailure(
          op, "Batch dimensions of A and B must match");
    }

    if (shape_A[rank - 1] != shape_A[rank - 2]) {
      return rewriter.notifyMatchFailure(
          op, "Inner two dimensions of A must be square");
    }

    if (shape_A[rank - 1] !=
        shape_B[rank - (op.getSide() == BlasSide::left ? 2 : 1)]) {
      return rewriter.notifyMatchFailure(
          op, "Inner dimensions of A and B must match");
    }

    // (C4)
    if (op.getResult().getType() != type_B) {
      return rewriter.notifyMatchFailure(op, "Result type must match B's type");
    }

    auto scaledB = stablehlo::MulOpCreate(rewriter, op.getLoc(), B, alpha);

    auto transa = stablehlo::Transpose::NO_TRANSPOSE;
    switch (op.getTransa()) {
    case BlasTranspose::none:
      transa = stablehlo::Transpose::NO_TRANSPOSE;
      break;
    case BlasTranspose::transpose:
      transa = stablehlo::Transpose::TRANSPOSE;
      break;
    case BlasTranspose::adjoint:
      transa = stablehlo::Transpose::ADJOINT;
      break;
    }

    auto trisolve_op = stablehlo::TriangularSolveOp::create(
        rewriter, op.getLoc(), TypeRange{type_B}, A, scaledB,
        /*left_side=*/op.getSide() == BlasSide::left,
        /*lower=*/op.getUplo() == BlasUplo::lower,
        /*unit_diagonal=*/op.getDiag() == BlasDiag::unit,
        /*transpose_a=*/transa);

    // replace enzymexla.blas.trsm with the trisolve_op
    rewriter.replaceAllUsesWith(op.getResult(), trisolve_op.getResult());
    rewriter.eraseOp(op);

    return success();
  }
};

struct LowerBlasToStableHLOPass
    : public mlir::blas::impl::LowerBlasToStableHLOPassBase<
          LowerBlasToStableHLOPass> {
  using Base::Base;

  void runOnOperation() override {
    auto context = getOperation()->getContext();
    RewritePatternSet patterns(context);

    patterns.add<SymmOpLowering, TrsmOpLowering>(context);

    GreedyRewriteConfig config;
    config.setUseTopDownTraversal(true);
    config.enableFolding();
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns),
                                     config))) {
      signalPassFailure();
    }

    // NOTE we don't verify because we might want to lower to jit_call
    // Verify that all illegal ops have been lowered
    // auto walkResult = getOperation()->walk([&](Operation *op) {
    //   if (isa<blas::TrsmOp>(op)) {
    //     op->emitError("Failed to lower enzymexla.blas operation");
    //     return WalkResult::interrupt();
    //   }
    //   return WalkResult::advance();
    // });

    // if (walkResult.wasInterrupted()) {
    //   signalPassFailure();
    // }
  }
};
