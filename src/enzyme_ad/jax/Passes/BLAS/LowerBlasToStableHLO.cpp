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

    patterns.add<TrsmOpLowering>(context);

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

    if (walkResult.wasInterrupted()) {
      signalPassFailure();
    }
  }
};
