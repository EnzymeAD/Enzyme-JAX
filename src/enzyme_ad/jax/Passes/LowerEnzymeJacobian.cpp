//===- LowerEnzymeJacobian.cpp  ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements patterns to convert JVP/VJPs originating from an enzyme.jacobian
// to enzyme fwddiff/autodiff calls
//===----------------------------------------------------------------------===//

#include "Enzyme/MLIR/Dialect/Dialect.h"
#include "Enzyme/MLIR/Dialect/Ops.h"
#include "src/enzyme_ad/jax/Passes/Passes.h"
#include "src/enzyme_ad/jax/Utils.h"
#include "stablehlo/dialect/StablehloOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <cstdint>

#define DEBUG_TYPE "lower-enzyme-jacobian"

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_LOWERENZYMEJACOBIANSTABLEHLO
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

using namespace mlir;
using namespace mlir::enzyme;

namespace {

struct DotGeneralLowering : public OpRewritePattern<stablehlo::DotGeneralOp> {
  using OpRewritePattern<stablehlo::DotGeneralOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(stablehlo::DotGeneralOp op,
                                PatternRewriter &rewriter) const override {

    // Recognize Jacobian op
    auto lhsOp = op.getLhs().getDefiningOp<enzyme::JacobianOp>();
    auto rhsOp = op.getRhs().getDefiningOp<enzyme::JacobianOp>();

    if (!lhsOp && !rhsOp)
      return failure();

    if (lhsOp && rhsOp)
      return failure();

    bool isJacobianLHS = false;
    if (lhsOp)
      isJacobianLHS = true;

    enzyme::JacobianOp jacOp = isJacobianLHS ? lhsOp : rhsOp;
    SymbolTableCollection symbolTable;
    auto fn = dyn_cast_or_null<FunctionOpInterface>(
        symbolTable.lookupNearestSymbolFrom(jacOp, jacOp.getFnAttr()));
    if (!fn)
      return failure();
    auto nargs = fn.getNumArguments();
    auto nouts = fn.getNumResults();

    // TODO: handle indexing for mutable arguments
    // jidx = d(out_idx) / d(in_idx), where jidx = out_idx * nargs + in_idx;
    auto J = cast<OpResult>(isJacobianLHS ? op.getLhs() : op.getRhs());
    auto jidx = J.getResultNumber();

    if (jidx >= nargs * nouts)
      return failure();

    auto diffo_idx = jidx / nargs;
    auto diffin_idx = jidx % nargs;

    Value dvec = isJacobianLHS ? op.getRhs() : op.getLhs();

    bool isJVP = true;
    bool isVJP = true;
    ArrayRef<int64_t> jrdims =
        isJacobianLHS
            ? op.getDotDimensionNumbers().getLhsContractingDimensions()
            : op.getDotDimensionNumbers().getRhsContractingDimensions();

    // need to check jrdims == output_dims OR  jrdims == input_dims. Any other
    // case should result in a failure
    if (auto JRType = dyn_cast<RankedTensorType>(J.getType())) {

      auto totaldims = JRType.getNumElements();
      auto nindims =
          dyn_cast<RankedTensorType>(fn.getArgument(diffin_idx).getType())
              .getNumElements();
      auto noutdims = totaldims - nindims;

      for (auto dimid : jrdims) {
        isJVP = isJVP && (dimid < noutdims);
        isVJP = isVJP && (dimid >= noutdims);
      }
    }

    // if we contract a mix of both the input and output dimensions, then this
    // is neither a jvp nor a vjp
    if (!isJVP && !isVJP)
      return failure();

    if (isJVP && isVJP)
      return failure();
    // TODO: add batching support
    if (isJVP) {
      // JVP -> enzyme.fwddiff transform
      // The resulting fwddiff op will only have in_idx -> enzyme_dup, out_idx
      // -> enzyme_dupnoneed

      SmallVector<Value> in_args;
      SmallVector<ActivityAttr, 2> newInActivityArgs;
      SmallVector<ActivityAttr, 2> newRetActivityArgs;
      for (auto [idx, act] :
           llvm::enumerate(jacOp.getActivity().getAsRange<ActivityAttr>())) {
        Value in = jacOp.getInputs()[idx];

        if (idx != diffin_idx) {
          in_args.push_back(in);
          newInActivityArgs.push_back(
              ActivityAttr::get(rewriter.getContext(), Activity::enzyme_const));
        } else {
          // push din and mark as dup
          in_args.push_back(in);
          in_args.push_back(dvec);
          newInActivityArgs.push_back(
              ActivityAttr::get(rewriter.getContext(), Activity::enzyme_dup));
        }
      }

      // construct ret_args
      for (auto [idx, ret_act] :
           llvm::enumerate(jacOp.getRetActivity().getAsRange<ActivityAttr>())) {
        if (idx == diffo_idx) {
          newRetActivityArgs.push_back(ActivityAttr::get(
              rewriter.getContext(), Activity::enzyme_dupnoneed));
        } else {
          newRetActivityArgs.push_back(ActivityAttr::get(
              rewriter.getContext(), Activity::enzyme_constnoneed));
        }
      }

      ArrayAttr newInActivity =
          ArrayAttr::get(rewriter.getContext(),
                         llvm::ArrayRef<Attribute>(newInActivityArgs.begin(),
                                                   newInActivityArgs.end()));
      ArrayAttr newRetActivity =
          ArrayAttr::get(rewriter.getContext(),
                         llvm::ArrayRef<Attribute>(newRetActivityArgs.begin(),
                                                   newRetActivityArgs.end()));

      rewriter.replaceOpWithNewOp<ForwardDiffOp>(
          op, op->getResultTypes(), jacOp.getFnAttr(), in_args, newInActivity,
          newRetActivity, nullptr, jacOp.getStrongZeroAttr());
    } else {
      // VJP -> enzyme.autodiff transform
      // The resulting autodiff op will only have in_idx -> enzyme_dup, out_idx
      // -> enzyme_dupnoneed

      SmallVector<Value> in_args(jacOp.getInputs());
      SmallVector<ActivityAttr, 2> newInActivityArgs;
      SmallVector<ActivityAttr, 2> newRetActivityArgs;
      for (auto [idx, act] :
           llvm::enumerate(jacOp.getActivity().getAsRange<ActivityAttr>())) {

        if (idx != diffin_idx) {
          newInActivityArgs.push_back(
              ActivityAttr::get(rewriter.getContext(), Activity::enzyme_const));
        } else {
          newInActivityArgs.push_back(ActivityAttr::get(
              rewriter.getContext(), Activity::enzyme_active));
        }
      }

      // push dvec
      in_args.push_back(dvec);

      // construct ret_args
      for (auto [idx, ret_act] :
           llvm::enumerate(jacOp.getRetActivity().getAsRange<ActivityAttr>())) {
        if (idx == diffo_idx) {
          // accounts for dvec
          newRetActivityArgs.push_back(ActivityAttr::get(
              rewriter.getContext(), Activity::enzyme_activenoneed));
        } else {
          newRetActivityArgs.push_back(ActivityAttr::get(
              rewriter.getContext(), Activity::enzyme_constnoneed));
        }
      }

      ArrayAttr newInActivity =
          ArrayAttr::get(rewriter.getContext(),
                         llvm::ArrayRef<Attribute>(newInActivityArgs.begin(),
                                                   newInActivityArgs.end()));
      ArrayAttr newRetActivity =
          ArrayAttr::get(rewriter.getContext(),
                         llvm::ArrayRef<Attribute>(newRetActivityArgs.begin(),
                                                   newRetActivityArgs.end()));

      rewriter.replaceOpWithNewOp<AutoDiffOp>(
          op, op->getResultTypes(), jacOp.getFnAttr(), in_args, newInActivity,
          newRetActivity, nullptr, jacOp.getStrongZeroAttr());
    }

    return success();
  }
};

struct LowerEnzymeJacobianStableHLO
    : public mlir::enzyme::impl::LowerEnzymeJacobianStableHLOBase<
          LowerEnzymeJacobianStableHLO> {

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<DotGeneralLowering>(context);

    GreedyRewriteConfig config;
    config.enableFolding();
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns),
                                     config))) {
      signalPassFailure();
    }

    // // Verify that all illegal ops have been lowered
    // auto walkResult = getOperation()->walk([&](Operation *op) {
    //   if (isa<enzyme::ConcatOp, enzyme::ExtractOp>(op)) {
    //     op->emitError("Failed to lower enzyme batch operation");
    //     return WalkResult::interrupt();
    //   }
    //   return WalkResult::advance();
    // });
    //
    // if (walkResult.wasInterrupted()) {
    //   signalPassFailure();
    // }
  };
};
} // namespace
