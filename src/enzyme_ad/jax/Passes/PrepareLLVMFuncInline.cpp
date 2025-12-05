//===- PrepareLLVMFuncInline.cpp - SPMD Pass -----------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This implements the pass PrepareLLVMFuncInline which marks alls llvm funcs as
// always_inline so that they get inlined with the standard inline pass.
//
//===----------------------------------------------------------------------===//

#include "src/enzyme_ad/jax/Passes/Passes.h"
// #include "mlir/Dialect/Func/IR/FuncOps.h"
// #include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::enzyme {
// Define Base Class and Default Constructor for main Class
#define GEN_PASS_DEF_PREPARELLVMFUNCINLINE
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"



namespace {

struct InlineGlobalCtorsPattern : public OpRewritePattern<LLVM::GlobalCtorsOp> {
  using OpRewritePattern<LLVM::GlobalCtorsOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(LLVM::GlobalCtorsOp globalCtorsOp, PatternRewriter &rewriter) const override {
    // Get the parent module to search for the 'main' function
    auto moduleOp = globalCtorsOp->getParentOfType<ModuleOp>();
    LLVM::LLVMFuncOp mainFuncOp;
    for (auto func : moduleOp.getOps<LLVM::LLVMFuncOp>()) {
      if (func.getSymName() == "main") {
        mainFuncOp = func;
        break;
      }
    }
    if (!mainFuncOp)
      return failure();  // No main function found

    // Get the entry block of main
    Block &entryBlock = mainFuncOp.front();

    // Insert calls to global constructors at the beginning of the entry block
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(&entryBlock);

    // Extract priorities and constructors
    auto priorities = globalCtorsOp.getPriorities();
    auto ctors = globalCtorsOp.getCtors();

    // Prepare to store constructors with their priorities
    SmallVector<std::pair<int32_t, LLVM::LLVMFuncOp>> sortedCtors;

    // Resolve symbol references to LLVMFuncOps
    for (size_t i = 0; i < priorities.size(); ++i) {
      auto priorityAttr = cast<IntegerAttr>(priorities[i]);
      int32_t priority = priorityAttr.getInt();
      auto funcSymbolRef = cast<SymbolRefAttr>(ctors[i]);
      // Look up the function in the module
      auto funcOp = moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(funcSymbolRef);
      if (!funcOp) {
        return failure();  // Function not found in the module
      }
      // Store the constructor with its priority
      sortedCtors.push_back({priority, funcOp});
    }

    // Sort the constructors by priority (ascending)
    llvm::sort(sortedCtors, [](const auto &a, const auto &b) {
      return a.first < b.first;  // Sort by priority
    });

    // Insert constructor calls in the correct order
    for (const auto &ctor : sortedCtors) {
      LLVM::LLVMFuncOp func = ctor.second;
      rewriter.create<LLVM::CallOp>(globalCtorsOp.getLoc(), func, ValueRange());
    }

    // Erase the llvm.global_ctors after inlining
    rewriter.eraseOp(globalCtorsOp);

    return success();
  }
};

struct PrepareLLVMFuncInline : impl::PrepareLLVMFuncInlineBase<PrepareLLVMFuncInline> {
  using PrepareLLVMFuncInlineBase::PrepareLLVMFuncInlineBase;
  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    OpBuilder builder(moduleOp.getContext());

    moduleOp.walk([&](LLVM::LLVMFuncOp llvmFuncOp) {
        // Skip the main external functions, they should/can not be inlined
        if (llvmFuncOp.getName() == "main" || llvmFuncOp.isExternal()) {
          return;
        }

        // Remove no_inline and optimize_none attributes if they exist
        llvmFuncOp->removeAttr("no_inline");
        llvmFuncOp->removeAttr("optimize_none");

        // Set the always_inline attribute
        llvmFuncOp->setAttr("always_inline", builder.getUnitAttr());
      });

    RewritePatternSet patterns(&getContext());
    // GreedyRewriteConfig config;
    patterns.insert<InlineGlobalCtorsPattern>(&getContext());
    (void)applyPatternsGreedily(moduleOp, std::move(patterns)/*, config*/);

  }
};

} // namespace
} // namespace mlir::enyzme
