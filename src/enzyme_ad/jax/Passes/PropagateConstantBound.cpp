//===- RemoveDuplicateFuncDef.cpp - Remove duplicate fund def -------------- //
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//
//
// This file implements a pass to remove duplicate function definitions.
//===---------------------------------------------------------------------===//

#include "src/enzyme_ad/jax/Passes/PassDetails.h"
#include "src/enzyme_ad/jax/Passes/Passes.h"

#include "src/enzyme_ad/jax/Dialect/Ops.h"
#include "stablehlo/dialect/StablehloOps.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Visitors.h"

using namespace mlir;
using namespace mlir::enzyme;

namespace {
struct PropagateConstantBoundsPass
    : public PropagateConstantBoundsPassBase<PropagateConstantBoundsPass> {
  void runOnOperation() override {
    auto moduleOp = getOperation();
    auto *ctx = moduleOp->getContext();
    SymbolTable symTable(moduleOp);

    // nvvm.read.ptx.sreg.tid.x
    auto walkResult = moduleOp->walk([&](enzymexla::KernelCallOp callOp) {
      auto symbolName = callOp.getFn();
      auto callee = symTable.lookup<LLVM::LLVMFuncOp>(symbolName);
      if (!callee)
        return WalkResult::advance();
      Region *reg = callee.getCallableRegion();
      // thread idx
      reg->walk([&](NVVM::ThreadIdXOp idxOp) {
        auto cst = callOp.getGridx().getDefiningOp();
        APInt intValue;
        if (matchPattern(cst, m_ConstantInt(&intValue)))
          idxOp->setAttr("range", LLVM::ConstantRangeAttr::get(
                                      ctx, 32, 0, intValue.getSExtValue()));
      });
      reg->walk([&](NVVM::ThreadIdYOp idyOp) {
        auto cst = callOp.getGridy().getDefiningOp();
        APInt intValue;
        if (matchPattern(cst, m_ConstantInt(&intValue)))
          idyOp->setAttr("range", LLVM::ConstantRangeAttr::get(
                                      ctx, 32, 0, intValue.getSExtValue()));
      });
      reg->walk([&](NVVM::ThreadIdZOp idzOp) {
        auto cst = callOp.getGridz().getDefiningOp();
        APInt intValue;
        if (matchPattern(cst, m_ConstantInt(&intValue)))
          idzOp->setAttr("range", LLVM::ConstantRangeAttr::get(
                                      ctx, 32, 0, intValue.getSExtValue()));
      });
      // block index
      reg->walk([&](NVVM::BlockIdXOp blkIdxOp) {
        auto cst = callOp.getBlockx().getDefiningOp();
        APInt intValue;
        if (matchPattern(cst, m_ConstantInt(&intValue)))
          blkIdxOp->setAttr("range", LLVM::ConstantRangeAttr::get(
                                         ctx, 32, 0, intValue.getSExtValue()));
      });
      reg->walk([&](NVVM::BlockIdYOp blkIdyOp) {
        auto cst = callOp.getBlocky().getDefiningOp();
        APInt intValue;
        if (matchPattern(cst, m_ConstantInt(&intValue)))
          blkIdyOp->setAttr("range", LLVM::ConstantRangeAttr::get(
                                         ctx, 32, 0, intValue.getSExtValue()));
      });
      reg->walk([&](NVVM::BlockIdZOp blkIdzOp) {
        auto cst = callOp.getBlockz().getDefiningOp();
        APInt intValue;
        if (matchPattern(cst, m_ConstantInt(&intValue)))
          blkIdzOp->setAttr("range", LLVM::ConstantRangeAttr::get(
                                         ctx, 32, 0, intValue.getSExtValue()));
      });
    });
  }
};
} // end namespace

namespace mlir {
namespace enzyme {
std::unique_ptr<Pass> createPropagateConstantBoundsPass() {
  return std::make_unique<PropagateConstantBoundsPass>();
}
} // namespace enzyme
} // namespace mlir