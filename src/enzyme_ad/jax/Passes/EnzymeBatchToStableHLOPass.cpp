//===- EnzymeBatchToStableHLOPass.cpp  ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to print the MLIR module
//===----------------------------------------------------------------------===//

#include "Enzyme/MLIR/Dialect/Dialect.h"
#include "Enzyme/MLIR/Dialect/Ops.h"
#include "src/enzyme_ad/jax/Passes/Passes.h"
#include "src/enzyme_ad/jax/Utils.h"
#include "stablehlo/dialect/StablehloOps.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_ENZYMEBATCHTOSTABLEHLOPASS
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

using namespace mlir;
using namespace mlir::enzyme;
using namespace enzyme;
namespace {

struct ExtractOpConversion : public OpConversionPattern<enzyme::ExtractOp> {
  using OpConversionPattern<enzyme::ExtractOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(enzyme::ExtractOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto outTy = op.getOutput().getType();
    // stablehlo always has tensor type
    auto outRankTy = dyn_cast<RankedTensorType>(outTy);
    auto rank = outRankTy.getRank();
    return failure(); 
    // stablehlo.dynamic_slice op
  }
};

struct ConcatOpConversion : public OpConversionPattern<enzyme::ConcatOp> {
  using OpConversionPattern<enzyme::ConcatOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(enzyme::ConcatOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Value> inputs = op.getInputs();
    if (inputs.empty())
      return failure();

    auto firstInTy = inputs.front().getType();

    // stablehlo always has tensor type
    // reshape each input to 1xinput_rank and concatenate on dim=0

    SmallVector<Value> expandedInputs;
    for (Value in : inputs) {
      auto inRankTy = cast<RankedTensorType>(in.getType());
      auto inShape = inRankTy.getShape();
      SmallVector<int64_t> newInShape = {1};
      newInShape.append(inShape.begin(), inShape.end());
      auto newInTy = inRankTy.clone(newInShape);
      Value newInput = rewriter.create<stablehlo::ReshapeOp>(
          op->getLoc(), newInTy, in, op->getAttrs());
      expandedInputs.push_back(newInput);
    }

    // concatenate on dim=0
    rewriter.replaceOpWithNewOp<stablehlo::ConcatenateOp>(
        op, op->getResultTypes(), expandedInputs, /*dim=*/0);
    return success();
  }
};
struct EnzymeBatchToStableHLOPass
    : public enzyme::impl::EnzymeBatchToStableHLOPassBase<
          EnzymeBatchToStableHLOPass> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<ConcatOpConversion, ExtractOpConversion>(context);

    ConversionTarget target(*context);
    target.addLegalDialect<stablehlo::StablehloDialect>();
    target.addLegalDialect<enzyme::EnzymeDialect>();
    target.addIllegalOp<enzyme::ConcatOp>();

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      signalPassFailure();
    }
  };
};
} // namespace
