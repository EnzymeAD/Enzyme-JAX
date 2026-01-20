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

static void buildBesselIBody(OpBuilder &builder, Location loc,
                             Block &entryBlock, StringRef backend) {
  // TODO: Implement actual BesselI lowering
  builder.create<func::ReturnOp>(loc, entryBlock.getArgument(1));
}

static void buildBesselJBody(OpBuilder &builder, Location loc,
                             Block &entryBlock, StringRef backend) {
  // TODO: Implement actual BesselJ lowering
  builder.create<func::ReturnOp>(loc, entryBlock.getArgument(1));
}

static void buildBesselJXBody(OpBuilder &builder, Location loc,
                              Block &entryBlock, StringRef backend) {
  // TODO: Implement actual BesselJX lowering
  builder.create<func::ReturnOp>(loc, entryBlock.getArgument(1));
}

static void buildSphericalBesselJBody(OpBuilder &builder, Location loc,
                                      Block &entryBlock, StringRef backend) {
  // TODO: Implement actual SphericalBesselJ lowering
  builder.create<func::ReturnOp>(loc, entryBlock.getArgument(1));
}

static void buildBesselYBody(OpBuilder &builder, Location loc,
                             Block &entryBlock, StringRef backend) {
  // TODO: Implement actual BesselY lowering
  builder.create<func::ReturnOp>(loc, entryBlock.getArgument(1));
}

static void buildBesselYXBody(OpBuilder &builder, Location loc,
                              Block &entryBlock, StringRef backend) {
  // TODO: Implement actual BesselYX lowering
  builder.create<func::ReturnOp>(loc, entryBlock.getArgument(1));
}

static void buildSphericalBesselYBody(OpBuilder &builder, Location loc,
                                      Block &entryBlock, StringRef backend) {
  // TODO: Implement actual SphericalBesselY lowering
  builder.create<func::ReturnOp>(loc, entryBlock.getArgument(1));
}

static void buildBesselHBody(OpBuilder &builder, Location loc,
                             Block &entryBlock, StringRef backend) {
  // TODO: Implement actual BesselH lowering
  builder.create<func::ReturnOp>(loc, entryBlock.getArgument(1));
}

static void buildHankelH1XBody(OpBuilder &builder, Location loc,
                               Block &entryBlock, StringRef backend) {
  // TODO: Implement actual HankelH1X lowering
  builder.create<func::ReturnOp>(loc, entryBlock.getArgument(1));
}

static void buildHankelH2XBody(OpBuilder &builder, Location loc,
                               Block &entryBlock, StringRef backend) {
  // TODO: Implement actual HankelH2X lowering
  builder.create<func::ReturnOp>(loc, entryBlock.getArgument(1));
}

static void buildBesselIXBody(OpBuilder &builder, Location loc,
                              Block &entryBlock, StringRef backend) {
  // TODO: Implement actual BesselIX lowering
  builder.create<func::ReturnOp>(loc, entryBlock.getArgument(1));
}

static void buildBesselKBody(OpBuilder &builder, Location loc,
                             Block &entryBlock, StringRef backend) {
  // TODO: Implement actual BesselK lowering
  builder.create<func::ReturnOp>(loc, entryBlock.getArgument(1));
}

static void buildBesselKXBody(OpBuilder &builder, Location loc,
                              Block &entryBlock, StringRef backend) {
  // TODO: Implement actual BesselKX lowering
  builder.create<func::ReturnOp>(loc, entryBlock.getArgument(1));
}

static void buildJincBody(OpBuilder &builder, Location loc, Block &entryBlock,
                          StringRef backend) {
  // TODO: Implement actual Jinc lowering
  builder.create<func::ReturnOp>(loc, entryBlock.getArgument(0));
}

template <typename BesselOp>
struct BesselOpLowering : public OpRewritePattern<BesselOp> {
  using BodyBuilderFn = void (*)(OpBuilder &, Location, Block &, StringRef);

  BesselOpLowering(std::string backend, MLIRContext *context,
                   BodyBuilderFn bodyBuilder, PatternBenefit benefit = 1)
      : OpRewritePattern<BesselOp>(context, benefit), backend(backend),
        bodyBuilder(bodyBuilder){};

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
      auto fnType =
          rewriter.getFunctionType(op->getOperandTypes(), op->getResultTypes());
      besselFunc = func::FuncOp::create(rewriter, loc, fnName, fnType);
      besselFunc.setPrivate();

      auto &entryBlock = *besselFunc.addEntryBlock();
      rewriter.setInsertionPointToStart(&entryBlock);

      bodyBuilder(rewriter, loc, entryBlock, backend);
    }

    rewriter.replaceOpWithNewOp<func::CallOp>(op, besselFunc,
                                              op->getOperands());
    return success();
  }

private:
  std::string backend;
  BodyBuilderFn bodyBuilder;
};

struct LowerEnzymeXLABesselPass
    : public enzyme::impl::LowerEnzymeXLABesselPassBase<
          LowerEnzymeXLABesselPass> {
  using Base::Base;

  void runOnOperation() override {
    auto context = getOperation()->getContext();
    RewritePatternSet patterns(context);

    patterns.add<BesselOpLowering<enzymexla::BesselIOp>>(backend, context,
                                                         buildBesselIBody);
    patterns.add<BesselOpLowering<enzymexla::BesselJOp>>(backend, context,
                                                         buildBesselJBody);
    patterns.add<BesselOpLowering<enzymexla::BesselJXOp>>(backend, context,
                                                          buildBesselJXBody);
    patterns.add<BesselOpLowering<enzymexla::SphericalBesselJOp>>(
        backend, context, buildSphericalBesselJBody);
    patterns.add<BesselOpLowering<enzymexla::BesselYOp>>(backend, context,
                                                         buildBesselYBody);
    patterns.add<BesselOpLowering<enzymexla::BesselYXOp>>(backend, context,
                                                          buildBesselYXBody);
    patterns.add<BesselOpLowering<enzymexla::SphericalBesselYOp>>(
        backend, context, buildSphericalBesselYBody);
    patterns.add<BesselOpLowering<enzymexla::BesselHOp>>(backend, context,
                                                         buildBesselHBody);
    patterns.add<BesselOpLowering<enzymexla::HankelH1XOp>>(backend, context,
                                                           buildHankelH1XBody);
    patterns.add<BesselOpLowering<enzymexla::HankelH2XOp>>(backend, context,
                                                           buildHankelH2XBody);
    patterns.add<BesselOpLowering<enzymexla::BesselIXOp>>(backend, context,
                                                          buildBesselIXBody);
    patterns.add<BesselOpLowering<enzymexla::BesselKOp>>(backend, context,
                                                         buildBesselKBody);
    patterns.add<BesselOpLowering<enzymexla::BesselKXOp>>(backend, context,
                                                          buildBesselKXBody);
    patterns.add<BesselOpLowering<enzymexla::JincOp>>(backend, context,
                                                      buildJincBody);

    GreedyRewriteConfig config;
    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns),
                                            config))) {
      signalPassFailure();
    }
  }
};
} // namespace
