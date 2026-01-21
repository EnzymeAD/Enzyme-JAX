#include "Enzyme/MLIR/Dialect/Dialect.h"
#include "Enzyme/MLIR/Passes/EnzymeBatchPass.h"
#include "src/enzyme_ad/jax/Dialect/Dialect.h"
#include "src/enzyme_ad/jax/Dialect/Ops.h"
#include "src/enzyme_ad/jax/Passes/Passes.h"
#include "src/enzyme_ad/jax/Utils.h"

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/dialect/ChloOps.h"
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

static Value createSplatConstant(OpBuilder &builder, Location loc,
                                 RankedTensorType type, double value) {
  auto attr = mlir::enzyme::makeAttr(type, value);
  return stablehlo::ConstantOp::create(builder, loc, type,
                                       cast<ElementsAttr>(attr));
}

static Value createSplatConstantI32(OpBuilder &builder, Location loc,
                                    RankedTensorType type, int32_t value) {
  auto attr = mlir::enzyme::makeAttr(type, value);
  return stablehlo::ConstantOp::create(builder, loc, type,
                                       cast<ElementsAttr>(attr));
}

static void buildBesselIBody(OpBuilder &builder, Location loc,
                             Block &entryBlock, StringRef backend) {
  Value nu = entryBlock.getArgument(0);
  Value x = entryBlock.getArgument(1);
  auto tensorType = cast<RankedTensorType>(x.getType());
  auto elemType = tensorType.getElementType();

  Value absNu = stablehlo::AbsOp::create(builder, loc, nu);
  Value absX = stablehlo::AbsOp::create(builder, loc, x);

  Value zero = createSplatConstant(builder, loc, tensorType, 0.0);
  Value one = createSplatConstant(builder, loc, tensorType, 1.0);
  Value half = createSplatConstant(builder, loc, tensorType, 0.5);
  Value quarter = createSplatConstant(builder, loc, tensorType, 0.25);
  Value two = createSplatConstant(builder, loc, tensorType, 2.0);
  Value pi = createSplatConstant(builder, loc, tensorType, M_PI);
  Value twoDivPi =
      createSplatConstant(builder, loc, tensorType, 2.0 / M_PI);

  Value nuGe0 = stablehlo::CompareOp::create(builder, loc, nu, zero,
                                             stablehlo::ComparisonDirection::GE);
  Value xGe0 = stablehlo::CompareOp::create(builder, loc, x, zero,
                                            stablehlo::ComparisonDirection::GE);

  Value xx = stablehlo::MulOp::create(builder, loc, absX, absX);
  xx = stablehlo::MulOp::create(builder, loc, xx, quarter);

  auto iterType =
      RankedTensorType::get(tensorType.getShape(), builder.getI32Type());
  Value iterZero = createSplatConstantI32(builder, loc, iterType, 0);
  Value iterOne = createSplatConstantI32(builder, loc, iterType, 1);
  int64_t maxIter = 200;
  Value iterMax = createSplatConstantI32(builder, loc, iterType, maxIter);

  auto boolType =
      RankedTensorType::get(tensorType.getShape(), builder.getI1Type());
  Value trueVal = stablehlo::ConstantOp::create(
      builder, loc, boolType,
      cast<ElementsAttr>(DenseElementsAttr::get(boolType, true)));

  TypeRange whileTypes = {iterType, tensorType, tensorType, boolType};
  ValueRange whileInits = {iterZero, zero, one, trueVal};
  auto whileOp =
      stablehlo::WhileOp::create(builder, loc, whileTypes, whileInits);

  {
    OpBuilder::InsertionGuard guard(builder);
    Block *condBlock = builder.createBlock(&whileOp.getCond());
    for (auto type : whileTypes)
      condBlock->addArgument(type, loc);
    builder.setInsertionPointToStart(condBlock);

    Value iter = condBlock->getArgument(0);
    Value active = condBlock->getArgument(3);

    Value iterLtMax = stablehlo::CompareOp::create(
        builder, loc, iter, iterMax, stablehlo::ComparisonDirection::LT);
    auto scalarBoolType = RankedTensorType::get({}, builder.getI1Type());
    stablehlo::ReduceOp anyActive = stablehlo::ReduceOp::create(
        builder, loc, TypeRange{scalarBoolType}, ValueRange{active},
        ValueRange{stablehlo::ConstantOp::create(
            builder, loc, scalarBoolType,
            cast<ElementsAttr>(DenseElementsAttr::get(
                scalarBoolType, false)))},
        builder.getDenseI64ArrayAttr(
            llvm::to_vector(llvm::seq<int64_t>(0, tensorType.getRank()))));
    {
      OpBuilder::InsertionGuard reduceGuard(builder);
      Block *reduceBody = builder.createBlock(&anyActive.getBody());
      reduceBody->addArgument(scalarBoolType, loc);
      reduceBody->addArgument(scalarBoolType, loc);
      builder.setInsertionPointToStart(reduceBody);
      Value orResult = stablehlo::OrOp::create(builder, loc,
                                               reduceBody->getArgument(0),
                                               reduceBody->getArgument(1));
      stablehlo::ReturnOp::create(builder, loc, orResult);
    }

    Value anyActiveBroadcast = stablehlo::BroadcastInDimOp::create(
        builder, loc, boolType, anyActive.getResult(0),
        builder.getDenseI64ArrayAttr({}));
    Value continueLoop =
        stablehlo::AndOp::create(builder, loc, iterLtMax, anyActiveBroadcast);

    stablehlo::ReduceOp anyContinue = stablehlo::ReduceOp::create(
        builder, loc, TypeRange{scalarBoolType}, ValueRange{continueLoop},
        ValueRange{stablehlo::ConstantOp::create(
            builder, loc, scalarBoolType,
            cast<ElementsAttr>(DenseElementsAttr::get(
                scalarBoolType, false)))},
        builder.getDenseI64ArrayAttr(
            llvm::to_vector(llvm::seq<int64_t>(0, tensorType.getRank()))));
    {
      OpBuilder::InsertionGuard reduceGuard(builder);
      Block *reduceBody = builder.createBlock(&anyContinue.getBody());
      reduceBody->addArgument(scalarBoolType, loc);
      reduceBody->addArgument(scalarBoolType, loc);
      builder.setInsertionPointToStart(reduceBody);
      Value orResult = stablehlo::OrOp::create(builder, loc,
                                               reduceBody->getArgument(0),
                                               reduceBody->getArgument(1));
      stablehlo::ReturnOp::create(builder, loc, orResult);
    }

    stablehlo::ReturnOp::create(builder, loc, anyContinue.getResult(0));
  }

  {
    OpBuilder::InsertionGuard guard(builder);
    Block *bodyBlock = builder.createBlock(&whileOp.getBody());
    for (auto type : whileTypes)
      bodyBlock->addArgument(type, loc);
    builder.setInsertionPointToStart(bodyBlock);

    Value iter = bodyBlock->getArgument(0);
    Value s = bodyBlock->getArgument(1);
    Value t = bodyBlock->getArgument(2);
    Value active = bodyBlock->getArgument(3);

    Value newS = stablehlo::AddOp::create(builder, loc, s, t);

    Value iterFloat = stablehlo::ConvertOp::create(builder, loc, tensorType, iter);
    Value iterPlusOne =
        stablehlo::AddOp::create(builder, loc, iterFloat, one);
    Value nuPlusIterPlusOne =
        stablehlo::AddOp::create(builder, loc, absNu, iterPlusOne);
    Value denom =
        stablehlo::MulOp::create(builder, loc, nuPlusIterPlusOne, iterPlusOne);
    Value factor = stablehlo::DivOp::create(builder, loc, xx, denom);
    Value newT = stablehlo::MulOp::create(builder, loc, t, factor);

    Value eps = createSplatConstant(
        builder, loc, tensorType,
        isa<Float32Type>(elemType) ? 1e-7 : 1e-15);
    Value absT = stablehlo::AbsOp::create(builder, loc, newT);
    Value absS = stablehlo::AbsOp::create(builder, loc, newS);
    Value relChange = stablehlo::DivOp::create(builder, loc, absT, absS);
    Value notConverged = stablehlo::CompareOp::create(
        builder, loc, relChange, eps, stablehlo::ComparisonDirection::GT);
    Value sIsZero = stablehlo::CompareOp::create(
        builder, loc, absS, eps, stablehlo::ComparisonDirection::LT);
    Value stillActive =
        stablehlo::OrOp::create(builder, loc, notConverged, sIsZero);
    Value newActive = stablehlo::AndOp::create(builder, loc, active, stillActive);

    Value newIter = stablehlo::AddOp::create(builder, loc, iter, iterOne);

    stablehlo::ReturnOp::create(builder, loc,
                                ValueRange{newIter, newS, newT, newActive});
  }

  Value s = whileOp.getResult(1);

  Value xHalf = stablehlo::MulOp::create(builder, loc, absX, half);
  Value xHalfPowNu = stablehlo::PowOp::create(builder, loc, xHalf, absNu);
  Value nuPlusOne = stablehlo::AddOp::create(builder, loc, absNu, one);
  Value lgammaNuPlusOne = chlo::LgammaOp::create(builder, loc, nuPlusOne);
  Value gammaNuPlusOne = stablehlo::ExpOp::create(builder, loc, lgammaNuPlusOne);

  Value prefactor = stablehlo::DivOp::create(builder, loc, xHalfPowNu, gammaNuPlusOne);
  Value besselIPositive = stablehlo::MulOp::create(builder, loc, s, prefactor);

  Value nuTimesFloor = stablehlo::FloorOp::create(builder, loc, absNu);
  Value nuIsInteger = stablehlo::CompareOp::create(
      builder, loc, absNu, nuTimesFloor, stablehlo::ComparisonDirection::EQ);

  Value nuTimesHalf = stablehlo::MulOp::create(builder, loc, absNu, half);
  Value nuHalfFloor = stablehlo::FloorOp::create(builder, loc, nuTimesHalf);
  Value nuHalfTimes2 = stablehlo::MulOp::create(builder, loc, nuHalfFloor, two);
  Value nuIsEven = stablehlo::CompareOp::create(
      builder, loc, absNu, nuHalfTimes2, stablehlo::ComparisonDirection::EQ);

  Value negOne = createSplatConstant(builder, loc, tensorType, -1.0);
  Value signForNegX =
      stablehlo::SelectOp::create(builder, loc, nuIsEven, one, negOne);

  Value resultForNegXInteger =
      stablehlo::MulOp::create(builder, loc, signForNegX, besselIPositive);
  Value resultForNegXNonInteger = zero;

  Value resultForNegX = stablehlo::SelectOp::create(
      builder, loc, nuIsInteger, resultForNegXInteger, resultForNegXNonInteger);
  Value resultForPosX = besselIPositive;
  Value besselIPosNu = stablehlo::SelectOp::create(builder, loc, xGe0,
                                                   resultForPosX, resultForNegX);

  Value piTimesNu = stablehlo::MulOp::create(builder, loc, pi, absNu);
  Value sinPiNu = stablehlo::SineOp::create(builder, loc, piTimesNu);

  Value besselK = besselIPositive;
  Value correction = stablehlo::MulOp::create(builder, loc, twoDivPi, sinPiNu);
  correction = stablehlo::MulOp::create(builder, loc, correction, besselK);
  Value besselINegNu =
      stablehlo::AddOp::create(builder, loc, besselIPositive, correction);

  Value result = stablehlo::SelectOp::create(builder, loc, nuGe0, besselIPosNu,
                                             besselINegNu);

  func::ReturnOp::create(builder, loc, result);
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
