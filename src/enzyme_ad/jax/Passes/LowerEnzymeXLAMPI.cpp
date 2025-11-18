// #include "mhlo/IR/hlo_ops.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "src/enzyme_ad/jax/Dialect/Dialect.h"
#include "src/enzyme_ad/jax/Dialect/Ops.h"
#include "src/enzyme_ad/jax/Passes/LinalgUtils.h"
#include "src/enzyme_ad/jax/Passes/Passes.h"
#include "src/enzyme_ad/jax/Utils.h"
// #include "stablehlo/dialect/StablehloOps.h"
// #include "llvm/ADT/DynamicAPInt.h"
// #include "llvm/ADT/SetVector.h"
// #include "llvm/ADT/SmallVector.h"
// #include "llvm/Support/ErrorHandling.h"
// #include "llvm/Support/LogicalResult.h"
// #include "llvm/Support/MathExtras.h"
// #include <algorithm>
// #include <cstdint>

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_LOWERENZYMEXLAMPIPASS
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

using namespace mlir;

struct MPICommRankOpLowering
    : public OpRewritePattern<enzymexla::MPICommRankOp> {

  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(enzymexla::MPICommRankOp op,
                                PatternRewriter &rewriter) const override {
  }
};

struct LowerEnzymeXLAMPIPass
    : public enzyme::impl::LowerEnzymeXLAMPIPassBase<
          LowerEnzymeXLAMPIPass> {
  using Base::Base;

  void runOnOperation() override {
    auto context = getOperation()->getContext();
    RewritePatternSet patterns(context);

    patterns.add<MPICommRankOpLowering>(context);

    GreedyRewriteConfig config;
    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns),
                                            config))) {
      signalPassFailure();
    }
  }
};
