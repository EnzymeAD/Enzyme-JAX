#include "Enzyme/MLIR/Dialect/Dialect.h"
#include "Enzyme/MLIR/Passes/EnzymeBatchPass.h"
#include "src/enzyme_ad/jax/Dialect/Dialect.h"
#include "src/enzyme_ad/jax/Dialect/Ops.h"
#include "src/enzyme_ad/jax/Passes/Passes.h"
#include "src/enzyme_ad/jax/Utils.h"

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/dialect/StablehloOps.h"

#define DEBUG_TYPE "lower-enzymexla-bessel"

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_LOWERENZYMEXLABESSELPASS
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

using namespace mlir;
using namespace mlir::enzyme;
using namespace mlir::enzymexla;
using namespace mlir::stablehlo;

namespace {

template <typename BesselOp>
struct BesselOpLowering : public OpRewritePattern<BesselOp> {
  BesselOpLowering(std::string backend, MLIRContext *context,
                   PatternBenefit benefit = 1)
      : OpRewritePattern<BesselOp>(context, benefit), backend(backend){};

  LogicalResult matchAndRewrite(BesselOp op,
                                PatternRewriter &rewriter) const override {

    auto moduleOp = op->template getParentOfType<ModuleOp>();
    auto loc = op.getLoc();
    auto fnName = op.getOperation()->getName().stripDialect().str();
    std::replace(fnName.begin(), fnName.end(), '.', '_');

    func::FuncOp besselFunc;
    if (!(besselFunc = moduleOp.template lookupSymbol<func::FuncOp>(fnName))) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());
      auto fnType = rewriter.getFunctionType(
          op->getOperandTypes(), op->getResultTypes());
      besselFunc = func::FuncOp::create(rewriter, loc, fnName, fnType);
      besselFunc.setPrivate();

      auto &entryBlock = *besselFunc.addEntryBlock();
      rewriter.setInsertionPointToStart(&entryBlock);

      // Placeholder implementation: return the last argument
      func::ReturnOp::create(rewriter, loc,
                             entryBlock.getArgument(op->getNumOperands() - 1));
    }

    rewriter.replaceOpWithNewOp<func::CallOp>(op, besselFunc,
                                              op->getOperands());
    return success();
  }

private:
  std::string backend;
};

struct LowerEnzymeXLABesselPass
    : public enzyme::impl::LowerEnzymeXLABesselPassBase<
          LowerEnzymeXLABesselPass> {
  using Base::Base;

  void runOnOperation() override {
    auto context = getOperation()->getContext();
    RewritePatternSet patterns(context);

    patterns.add<BesselOpLowering<enzymexla::BesselIOp>>(backend, context);
    patterns.add<BesselOpLowering<enzymexla::BesselJOp>>(backend, context);
    patterns.add<BesselOpLowering<enzymexla::BesselJXOp>>(backend, context);
    patterns.add<BesselOpLowering<enzymexla::SphericalBesselJOp>>(backend, context);
    patterns.add<BesselOpLowering<enzymexla::BesselYOp>>(backend, context);
    patterns.add<BesselOpLowering<enzymexla::BesselYXOp>>(backend, context);
    patterns.add<BesselOpLowering<enzymexla::SphericalBesselYOp>>(backend, context);
    patterns.add<BesselOpLowering<enzymexla::BesselHOp>>(backend, context);
    patterns.add<BesselOpLowering<enzymexla::HankelH1XOp>>(backend, context);
    patterns.add<BesselOpLowering<enzymexla::HankelH2XOp>>(backend, context);
    patterns.add<BesselOpLowering<enzymexla::BesselIXOp>>(backend, context);
    patterns.add<BesselOpLowering<enzymexla::BesselKOp>>(backend, context);
    patterns.add<BesselOpLowering<enzymexla::BesselKXOp>>(backend, context);
    patterns.add<BesselOpLowering<enzymexla::JincOp>>(backend, context);

    GreedyRewriteConfig config;
    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns),
                                            config))) {
      signalPassFailure();
    }
  }
};
} // namespace
