//===- InvokeToCall.cpp - Call the callees that cannot unwind -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//
//
// The LLVM dialect inliner takes only llvm.call sites. An llvm.invoke of a
// function that cannot unwind has a dead unwind edge, so it becomes an
// llvm.call and a branch to its normal successor; a landing pad left without
// a predecessor goes with it. Whether a function can unwind is decided over
// its body -- it never resumes, and every call in it reaches a function that
// cannot unwind -- the way the pipeline treats every body it inlines, rather
// than left to the attribute LLVM withholds from a definition another unit
// might supersede.
//
//===---------------------------------------------------------------------===//
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/RegionUtils.h"
#include "src/enzyme_ad/jax/Passes/Passes.h"

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_INVOKETOCALLPASS
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

using namespace mlir;

namespace {
struct InvokeToCallPass
    : public enzyme::impl::InvokeToCallPassBase<InvokeToCallPass> {
  using InvokeToCallPassBase::InvokeToCallPassBase;

  void runOnOperation() override {
    SymbolTableCollection symbols;
    SmallVector<LLVM::LLVMFuncOp> funcs;
    DenseSet<Operation *> noUnwind;
    getOperation()->walk([&](LLVM::LLVMFuncOp fn) {
      funcs.push_back(fn);
      if (fn.getNoUnwind())
        noUnwind.insert(fn);
    });
    auto calleeCannotUnwind = [&](CallOpInterface call) {
      Operation *callee = call.resolveCallableInTable(&symbols);
      return callee && noUnwind.contains(callee);
    };
    for (bool changed = true; changed;) {
      changed = false;
      for (LLVM::LLVMFuncOp fn : funcs) {
        if (fn.isExternal() || noUnwind.contains(fn))
          continue;
        bool mayUnwind = fn.walk([&](Operation *op) {
                             if (isa<LLVM::ResumeOp, LLVM::InlineAsmOp>(op))
                               return WalkResult::interrupt();
                             if (auto call = dyn_cast<CallOpInterface>(op))
                               if (!calleeCannotUnwind(call))
                                 return WalkResult::interrupt();
                             return WalkResult::advance();
                           }).wasInterrupted();
        if (mayUnwind)
          continue;
        noUnwind.insert(fn);
        fn.setNoUnwind(true);
        changed = true;
      }
    }

    IRRewriter rewriter(&getContext());
    for (LLVM::LLVMFuncOp fn : funcs) {
      SmallVector<LLVM::InvokeOp> invokes;
      fn.walk([&](LLVM::InvokeOp invoke) {
        if (calleeCannotUnwind(invoke))
          invokes.push_back(invoke);
      });
      if (invokes.empty())
        continue;
      for (LLVM::InvokeOp invoke : invokes) {
        auto callee =
            cast<LLVM::LLVMFuncOp>(invoke.resolveCallableInTable(&symbols));
        rewriter.setInsertionPoint(invoke);
        auto call = LLVM::CallOp::create(rewriter, invoke.getLoc(), callee,
                                         invoke.getCalleeOperands());
        call.setCConv(invoke.getCConv());
        if (auto attrs = invoke.getArgAttrsAttr())
          call.setArgAttrsAttr(attrs);
        if (auto attrs = invoke.getResAttrsAttr())
          call.setResAttrsAttr(attrs);
        rewriter.replaceAllUsesWith(invoke.getResults(), call.getResults());
        rewriter.replaceOpWithNewOp<LLVM::BrOp>(
            invoke, invoke.getNormalDestOperands(), invoke.getNormalDest());
      }
      (void)eraseUnreachableBlocks(rewriter, fn->getRegions());
    }
  }
};
} // namespace
