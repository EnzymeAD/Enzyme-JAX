//===- SimplifyControlFlow.cpp - SPMD Pass -----------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This implements the pass SimplifyControlFlow which converts all cf.SwitchOps 
// first to scf.IndexSwitchOps and then all scf.IndexSwitchOps to scf.IfOps.
//
//===----------------------------------------------------------------------===//

#include "src/enzyme_ad/jax/Passes/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
// #include "mlir/Dialect/Func/IR/FuncOps.h"

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::enzyme  {
// Define Base Class and Default Constructor for main Class
#define GEN_PASS_DEF_HANDLELLVMUNREACHABLE
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"

namespace {

struct ConvertUnreachableToReturnPattern : public OpRewritePattern<LLVM::UnreachableOp> {
  using OpRewritePattern<LLVM::UnreachableOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(LLVM::UnreachableOp op, PatternRewriter &rewriter) const override {
    // Get the parent operation (the containing function).
    auto funcOp = op->getParentOfType<LLVM::LLVMFuncOp>();
    if(!funcOp) {
      return failure();
    }
    // Check if the function returns an integer type.
    auto returnType = funcOp.getFunctionType().getReturnType();
    if (!isa<IntegerType>(returnType))
      return failure();  // Only apply the pattern if the return type is an integer.

    // Create a constant value of -42 as the return value.
    auto minus42 = rewriter.create<arith::ConstantOp>(
        op.getLoc(), returnType, rewriter.getIntegerAttr(returnType, -42));

    // Replace the llvm.unreachable with llvm.return and return the value -42.
    rewriter.replaceOpWithNewOp<LLVM::ReturnOp>(op, minus42.getResult());

    return success();
  }
};


struct HandleLLVMUnreachable : impl::HandleLLVMUnreachableBase<HandleLLVMUnreachable> {
  using HandleLLVMUnreachableBase::HandleLLVMUnreachableBase;
  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    OpBuilder builder(moduleOp.getContext());
    RewritePatternSet patterns(&getContext());
    // GreedyRewriteConfig config;
    patterns.insert<ConvertUnreachableToReturnPattern>(&getContext());
    (void)applyPatternsGreedily(moduleOp, std::move(patterns)/*, config*/);
  }
};

} // namespace
} // namespace mlir::enyzme
