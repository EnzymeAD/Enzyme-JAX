#include "Enzyme/MLIR/Dialect/Dialect.h"
#include "Enzyme/MLIR/Passes/EnzymeBatchPass.h"
#include "src/enzyme_ad/jax/Dialect/Dialect.h"
#include "src/enzyme_ad/jax/Dialect/Ops.h"
#include "src/enzyme_ad/jax/Passes/Passes.h"
#include "src/enzyme_ad/jax/Utils.h"

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/dialect/ChloOps.h"

#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

#define DEBUG_TYPE "lower-enzymexla-bessel"

namespace {
// Create a besseli power series
static Value createBesseliPowerSeries(PatternRewriter &rewriter, Location loc,
                                      Type type, Value v, Value x) {
  Value zero = rewriter.create<arith::ConstantOp>(
      loc, FloatAttr::get(type, 0.0)).getResult();
  Value one = rewriter.create<arith::ConstantOp>(
      loc, FloatAttr::get(type, 1.0)).getResult();
  Value two = rewriter.create<arith::ConstantOp>(
      loc, FloatAttr::get(type, 2.0)).getResult();
  Value quarter = rewriter.create<arith::ConstantOp>(
      loc, FloatAttr::get(type, 0.25)).getResult();

  Value s = zero;
  auto t = one;
  arith::MulFOp xx = rewriter.create<arith::MulFOp>(
      loc, rewriter.create<arith::MulFOp>(loc, x, x), quarter);

  arith::ConstantOp lowerBound =
      rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(0));
  arith::ConstantOp upperBound =
      rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(3000));
  arith::ConstantOp step =
      rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(1));

  scf::ForOp loop = rewriter.create<scf::ForOp>(
      loc, lowerBound, upperBound, step, ValueRange{s, t},
      [&](OpBuilder &builder, Location loc, Value i, ValueRange loopOperands) {
        Value s_iter = loopOperands[0];
        Value t_iter = loopOperands[1];

        s_iter = builder.create<arith::AddFOp>(loc, s_iter, t_iter);

        Value ip1 = builder.create<arith::AddIOp>(
            loc, i, builder.create<arith::ConstantOp>(loc, builder.getIndexAttr(1)));
        Value i_float =
            builder.create<arith::IndexCastOp>(loc, type, ip1);

        Value v_plus_i_plus_1 = builder.create<arith::AddFOp>(
            loc, builder.create<arith::AddFOp>(loc, v, i_float), one);
        Value den =
            builder.create<arith::MulFOp>(loc, v_plus_i_plus_1, i_float);
        Value term = builder.create<arith::DivFOp>(loc, xx, den);
        t_iter = builder.create<arith::MulFOp>(loc, t_iter, term);

        builder.create<scf::YieldOp>(loc, ValueRange{s_iter, t_iter});
      });

  s = loop.getResults()[0];

  Value x_over_2 = rewriter.create<stablehlo::AddOp>(loc, x, two).getResult();
  Value log_x_over_2 = rewriter.create<stablehlo::LogOp>(loc, x_over_2).getResult();
  Value v_log_x_over_2 = rewriter.create<stablehlo::MulOp>(loc, v, log_x_over_2).getResult();
  Value x_over_2_pow_v = rewriter.create<stablehlo::ExpOp>(loc, v_log_x_over_2).getResult();

  Value v_plus_1 = rewriter.create<stablehlo::AddOp>(loc, v, one).getResult();
  Value lgamma_v_plus_1 =
      rewriter.create<chlo::LgammaOp>(loc, type, v_plus_1).getResult();
  Value gamma_v_plus_1 = rewriter.create<stablehlo::ExpOp>(loc, lgamma_v_plus_1).getResult();

  Value term1 =
      rewriter.create<stablehlo::DivOp>(loc, x_over_2_pow_v, gamma_v_plus_1).getResult();
  return rewriter.create<stablehlo::MulOp>(loc, s, term1).getResult();
}

// Create a besseli with large arguments
static Value createBesseliLargeArgs(PatternRewriter &rewriter, Location loc,
                                    Type type, Value v, Value x) {
  Value one = rewriter.create<arith::ConstantOp>(
      loc, FloatAttr::get(type, 1.0)).getResult();
  Value two = rewriter.create<arith::ConstantOp>(
      loc, FloatAttr::get(type, 2.0)).getResult();
  Value four = rewriter.create<arith::ConstantOp>(
      loc, FloatAttr::get(type, 4.0)).getResult();
  Value eight = rewriter.create<arith::ConstantOp>(
      loc, FloatAttr::get(type, 8.0)).getResult();
  Value neg_one = rewriter.create<arith::ConstantOp>(
      loc, FloatAttr::get(type, -1.0)).getResult();

  auto t = one;
  auto s = t;

  Value invx = rewriter.create<arith::DivFOp>(
      loc, one, rewriter.create<arith::MulFOp>(loc, eight, x));
  Value neg_invx = rewriter.create<arith::MulFOp>(loc, invx, neg_one);
  Value four_v_sq = rewriter.create<arith::MulFOp>(
      loc, four, rewriter.create<arith::MulFOp>(loc, v, v));

  auto lowerBound =
      rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(1));
  auto upperBound =
      rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(1000));
  auto step =
      rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(1));

  scf::ForOp loop = rewriter.create<scf::ForOp>(
      loc, lowerBound, upperBound, step, ValueRange{s, t},
      [&](OpBuilder &builder, Location loc, Value i, ValueRange loopOperands) {
        Value s_iter = loopOperands[0];
        Value t_iter = loopOperands[1];

        s_iter = builder.create<arith::AddFOp>(loc, s_iter, t_iter);

        Value i_float =
            builder.create<arith::IndexCastOp>(loc, type, i);
        Value two_i =
            builder.create<arith::MulFOp>(loc, two, i_float);
        Value two_i_minus_1 =
            builder.create<arith::SubFOp>(loc, two_i, one);
        Value term_sq =
            builder.create<arith::MulFOp>(loc, two_i_minus_1, two_i_minus_1);
        Value num = builder.create<arith::SubFOp>(loc, four_v_sq, term_sq);
        Value term = builder.create<arith::DivFOp>(loc, num, i_float);

        t_iter =
            builder.create<arith::MulFOp>(loc, t_iter, term);
        t_iter = builder.create<arith::MulFOp>(loc, t_iter, neg_invx);
        s_iter = builder.create<arith::AddFOp>(loc, s_iter, t_iter);

        builder.create<scf::YieldOp>(loc, ValueRange{s_iter, t_iter});
      });

  s = loop.getResults().front();

  Value pi = rewriter.create<arith::ConstantOp>(
      loc, FloatAttr::get(type, M_PI)).getResult();
  Value two_pi_x = rewriter.create<arith::MulFOp>(
      loc, rewriter.create<stablehlo::MulOp>(loc, two, pi), x).getResult();
  Value sqrt_two_pi_x = rewriter.create<stablehlo::SqrtOp>(loc, two_pi_x).getResult();
  Value besselix = rewriter.create<stablehlo::DivOp>(loc, s, sqrt_two_pi_x).getResult();

  Value exp_x = rewriter.create<stablehlo::ExpOp>(loc, x).getResult();

  return rewriter.create<arith::MulFOp>(loc, exp_x, besselix);
}

// Create a besseli for positive args
static Value createBesseliPositiveArgs(PatternRewriter &rewriter, Location loc,
                                       Type type, Value nu, Value x) {
  Value cutoff_val;
  if (type.isF64()) {
    cutoff_val = rewriter.create<arith::ConstantOp>(
        loc, FloatAttr::get(type, 19.0));
  } else {
    cutoff_val = rewriter.create<arith::ConstantOp>(
        loc, FloatAttr::get(type, 9.0));
  }

  auto half = rewriter.create<arith::ConstantOp>(
      loc, FloatAttr::get(type, 0.5)).getResult();
  auto nu_sq = rewriter.create<stablehlo::MulOp>(loc, nu, nu).getResult();
  auto nu_sq_half = rewriter.create<stablehlo::MulOp>(loc, nu_sq, half).getResult();
  auto cutoff = rewriter.create<arith::AddFOp>(loc, nu_sq_half, cutoff_val).getResult();
  auto is_large_arg =
      rewriter.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OGT, x, cutoff).getResult();

  scf::IfOp if_op = rewriter.create<scf::IfOp>(
      loc, type, is_large_arg,
      [&](OpBuilder &builder, Location loc) {
        Value res = createBesseliLargeArgs(builder, loc, type, nu, x);
        builder.create<scf::YieldOp>(loc, res);
      },
      [&](OpBuilder &builder, Location loc) {
        Value res = createBesseliPowerSeries(builder, loc, type, nu, x);
        builder.create<scf::YieldOp>(loc, res);
    });
  return if_op.getResults().front();
}

// Create the besseli function
static void createBesselIFunction(PatternRewriter &rewriter, Location loc,
                                  StringRef fnName, func::FuncOp &besselFunc) {
  auto &entryBlock = *besselFunc.addEntryBlock();
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(&entryBlock);

  BlockArgument nu = entryBlock.getArgument(0);
  BlockArgument x = entryBlock.getArgument(1);
  auto type = x.getType().cast<ShapedType>().getElementType();

  Value s_nu = rewriter.create<enzyme::GetScalar>(loc, nu);
  Value s_x = rewriter.create<enzyme::GetScalar>(loc, x);

  auto zero = rewriter.create<arith::ConstantOp>(
      loc, FloatAttr::get(type, 0.0));
  auto abs_nu = rewriter.create<math::AbsFOp>(loc, s_nu);
  auto abs_x = rewriter.create<math::AbsFOp>(loc, s_x);

  auto res = createBesseliPositiveArgs(rewriter, loc, type, abs_nu, abs_x);

  auto is_int = rewriter.create<arith::CmpFOp>(
      loc, arith::CmpFPredicate::OEQ, s_nu,
      rewriter.create<math::RoundOp>(loc, s_nu));
  auto is_neg_x =
      rewriter.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OLT, s_x, zero);

  auto if_neg_x = rewriter.create<scf::IfOp>(
      loc, type, is_neg_x,
      [&](OpBuilder &builder, Location loc) {
        auto if_is_int = builder.create<scf::IfOp>(
            loc, type, is_int,
            [&](OpBuilder &builder, Location loc) {
              Value nu_rem_2 = builder.create<arith::RemFOp>(
                  loc, s_nu,
                  builder.create<arith::ConstantOp>(
                      loc, FloatAttr::get(type, 2.0)));
              Value is_even = builder.create<arith::CmpFOp>(
                  loc, arith::CmpFPredicate::OEQ, nu_rem_2, zero);
              auto if_is_even = builder.create<scf::IfOp>(
                  loc, type, is_even,
                  [&](OpBuilder &builder, Location loc) {
                    builder.create<scf::YieldOp>(loc, res);
                  },
                  [&](OpBuilder &builder, Location loc) {
                    Value neg_res =
                        builder.create<arith::MulFOp>(loc, res,
                                                      builder.create<arith::ConstantOp>(
                                                          loc, FloatAttr::get(
                                                                   type, -1.0)));
                    builder.create<scf::YieldOp>(loc, neg_res);
                  });
              builder.create<scf::YieldOp>(loc, if_is_even.getResult(0));
            },
            [&](OpBuilder &builder, Location loc) {
              Value nan = builder.create<arith::ConstantOp>(
                  loc, APFloat::getQNaN(type.cast<FloatType>().getFloatSemantics()));
              builder.create<scf::YieldOp>(loc, nan);
            });
        builder.create<scf::YieldOp>(loc, if_is_int.getResult(0));
      },
      [&](OpBuilder &builder, Location loc) {
        builder.create<scf::YieldOp>(loc, res);
      });

  Value result = rewriter.create<enzyme::ReturnScalar>(loc, if_neg_x.getResult(0));
  rewriter.create<func::ReturnOp>(loc, result);
}
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
      auto fnType = rewriter.getFunctionType(op->getOperandTypes(),
                                             op->getResultTypes());
      besselFunc = func::FuncOp::create(rewriter, loc, fnName, fnType);
      besselFunc.setPrivate();

      if (std::is_same<BesselOp, enzymexla::BesselIOp>()) {
        createBesselIFunction(rewriter, loc, fnName, besselFunc);
      } else {
        auto &entryBlock = *besselFunc.addEntryBlock();
        rewriter.setInsertionPointToStart(&entryBlock);
        // Placeholder implementation: return the last argument
        func::ReturnOp::create(
            rewriter, loc, entryBlock.getArgument(op->getNumOperands() - 1));
      }
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
