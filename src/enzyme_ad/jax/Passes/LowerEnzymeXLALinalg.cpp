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

struct TridiagonalSolveOpLowering
    : public OpRewritePattern<enzymexla::TridiagonalSolveOp> {
  using OpRewritePattern::OpRewritePattern;

  std::string backend;
  int64_t blasIntWidth;
  TridiagonalSolveOpLowering(std::string backend, int64_t blasIntWidth,
                             MLIRContext *context, PatternBenefit benefit = 1)
      : OpRewritePattern(context, benefit), backend(backend),
        blasIntWidth(blasIntWidth) {}

  LogicalResult matchAndRewrite(enzymexla::TridiagonalSolveOp op,
                                PatternRewriter &rewriter) const override {
    if (backend == "cpu") {
      return matchAndRewriteCPU(op, rewriter);
    } else if (backend == "cuda") {
      return matchAndRewriteCUDA(op, rewriter);
    }
    return matchAndRewriteFallback(op, rewriter);
  }

  LogicalResult matchAndRewriteCPU(enzymexla::TridiagonalSolveOp op,
                                   PatternRewriter &rewriter) const {
    // TODO
    return matchAndRewriteFallback(op, rewriter);
  }

  LogicalResult matchAndRewriteCUDA(enzymexla::TridiagonalSolveOp op,
                                    PatternRewriter &rewriter) const {
    auto dType = cast<RankedTensorType>(op.getDl().getType());
    auto dRank = dType.getRank();
    auto bType = cast<RankedTensorType>(op.getBl().getType());
    auto bRank = bType.getRank();

    // Build operands list - use empty tensors for constant alpha/beta
    SmallVector<Value> operands;
    SmallVector<int64_t> operandRanks;
    SmallVector<bool> areColMajor;

    auto dl = op.getDl();
    auto d = op.getD();
    auto du = op.getDu();
    auto B = op.getB();

    StringAttr customCallTarget;
    ArrayAttr aliases;

    customCallTarget = rewriter.getStringAttr("cusparse_gtsv2_ffi");
    operands = {dl, d, du, B};
    operandRanks = {dRank, dRank, dRank, bRank};
    aliases = rewriter.getArrayAttr(
        {stablehlo::OutputOperandAliasAttr::get(op.getContext(), {}, 3, {})});
    areColMajor = {true, true, true, true};

    DictionaryAttr backendConfig = rewriter.getDictionaryAttr({});

    auto customCall = stablehlo::CustomCallOp::create(
        rewriter, op.getLoc(), TypeRange{bType}, operands, customCallTarget,
        /*has_side_effect*/ nullptr,
        /*backend_config*/ backendConfig,
        /*api_version*/
        stablehlo::CustomCallApiVersionAttr::get(
            rewriter.getContext(),
            mlir::stablehlo::CustomCallApiVersion::API_VERSION_TYPED_FFI),
        /*calledcomputations*/ nullptr,
        /*operand_layouts*/
        getSHLOLayout(rewriter, operandRanks, areColMajor, rank),
        /*result_layouts*/
        getSHLOLayout(rewriter, {rank}, SmallVector<bool>(rank, true), rank),
        /*output_operand_aliases*/ aliases);

    Value result = customCall.getResult(0);
    rewriter.replaceAllUsesWith(op.getResult(), result);

    return success();
  }

  LogicalResult matchAndRewriteFallback(enzymexla::TridiagonalSolveOp op,
                                        PatternRewriter &rewriter) const {
    // TODO
  }
};

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

struct LowerEnzymeXLALinalgPass
    : public enzyme::impl::LowerEnzymeXLALinalgPassBase<
          LowerEnzymeXLALinalgPass> {
  using Base::Base;

  void runOnOperation() override {
    auto context = getOperation()->getContext();
    RewritePatternSet patterns(context);

    patterns.add<LUFactorizationOpLowering>(context);
    patterns.add<SVDFactorizationOpLowering>(backend, context);

    GreedyRewriteConfig config;
    config.enableFolding();
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns),
                                     config))) {
      signalPassFailure();
    }

    // Verify that all illegal ops have been lowered
    auto walkResult = getOperation()->walk([&](Operation *op) {
      if (isa<enzymexla::LUFactorizationOp, enzymexla::SVDFactorizationOp>(
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
