//===- LLVMToControlFlow.cpp - ControlFlow to LLVM dialect conversion -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to convert MLIR standard and builtin dialects
// into the LLVM IR dialect.
//
//===----------------------------------------------------------------------===//

#include "Passes.h"

#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <functional>

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_CONVERTLLVMTOCONTROLFLOWPASS
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

using namespace mlir;
using namespace mlir::enzyme;

#define PASS_NAME "convert-llvm-to-cf"

namespace {

struct BranchOpLifting : public OpRewritePattern<LLVM::BrOp> {
  using OpRewritePattern<LLVM::BrOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(LLVM::BrOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<cf::BranchOp>(op, op.getDest(),
                                              op.getDestOperands());
    return success();
  }
};

struct CondBranchOpLifting : public OpRewritePattern<LLVM::CondBrOp> {
  using OpRewritePattern<LLVM::CondBrOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(LLVM::CondBrOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<cf::CondBranchOp>(
        op, op.getCondition(), op.getTrueDest(), op.getTrueDestOperands(),
        op.getFalseDest(), op.getFalseDestOperands());
    return success();
  }
};

struct SwitchOpLifting : public OpRewritePattern<LLVM::SwitchOp> {
  using OpRewritePattern<LLVM::SwitchOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(LLVM::SwitchOp op,
                                PatternRewriter &rewriter) const override {
    SmallVector<APInt> caseValues;
    SmallVector<ValueRange> caseOperands;
    if (auto cvs = op.getCaseValues())
      for (auto val : *cvs)
        caseValues.push_back(val);
    for (auto val : op.getCaseOperands())
      caseOperands.push_back(val);
    rewriter.replaceOpWithNewOp<cf::SwitchOp>(
        op, op.getValue(), op.getDefaultDestination(), op.getDefaultOperands(),
        caseValues, op.getCaseDestinations(), caseOperands);
    return success();
  }
};

} // namespace

void mlir::cf::populateLLVMToControlFlowConversionPatterns(
    RewritePatternSet &patterns) {
  // clang-format off
  patterns.add<
      BranchOpLifting,
      CondBranchOpLifting,
      SwitchOpLifting>(patterns.getContext());
  // clang-format on
}

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

namespace {
/// A pass converting MLIR operations into the LLVM IR dialect.
struct ConvertLLVMToControlFlow
    : public enzyme::impl::ConvertLLVMToControlFlowPassBase<
          ConvertLLVMToControlFlow> {
  using ConvertLLVMToControlFlowPassBase::ConvertLLVMToControlFlowPassBase;
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    mlir::cf::populateLLVMToControlFlowConversionPatterns(patterns);
    if (failed(
            applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace
