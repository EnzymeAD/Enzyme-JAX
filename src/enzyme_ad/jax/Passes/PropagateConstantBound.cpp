//===- PropagateConstantBounds.cpp - Remove duplicate fund def ------------- //
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
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

  static void attachConstantRangeIfConstant(MLIRContext *ctx,
                                            Operation *maybeCst,
                                            Operation *target) {
    APInt intValue;
    if (matchPattern(maybeCst, m_ConstantInt(&intValue)))
      target->setAttr("range", LLVM::ConstantRangeAttr::get(
                                   ctx, 32, 0, intValue.getSExtValue()));
  }

  static void replaceWithConstantIfConstant(OpBuilder &builder,
                                            Operation *maybeCst,
                                            Operation *target) {
    APInt intValue;
    auto loc = target->getLoc();
    if (matchPattern(maybeCst, m_ConstantInt(&intValue))) {
      builder.setInsertionPoint(target);
      auto newCst = builder.create<LLVM::ConstantOp>(
          loc, builder.getI32Type(),
          builder.getIntegerAttr(builder.getI32Type(),
                                 intValue.getSExtValue()));
      target->getResult(0).replaceAllUsesWith(newCst.getResult());
    }
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();
    auto *ctx = moduleOp->getContext();
    OpBuilder builder(ctx);
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
        attachConstantRangeIfConstant(ctx, callOp.getGridx().getDefiningOp(),
                                      idxOp.getOperation());
      });
      reg->walk([&](NVVM::ThreadIdYOp idyOp) {
        attachConstantRangeIfConstant(ctx, callOp.getGridy().getDefiningOp(),
                                      idyOp.getOperation());
      });
      reg->walk([&](NVVM::ThreadIdZOp idzOp) {
        attachConstantRangeIfConstant(ctx, callOp.getGridz().getDefiningOp(),
                                      idzOp.getOperation());
      });
      // thread range
      reg->walk([&](NVVM::BlockDimXOp blockIdxOp) {
        replaceWithConstantIfConstant(builder,
                                      callOp.getGridx().getDefiningOp(),
                                      blockIdxOp.getOperation());
      });
      reg->walk([&](NVVM::BlockDimYOp blockIdyOp) {
        replaceWithConstantIfConstant(builder,
                                      callOp.getGridy().getDefiningOp(),
                                      blockIdyOp.getOperation());
      });
      reg->walk([&](NVVM::BlockDimZOp blockIdzOp) {
        replaceWithConstantIfConstant(builder,
                                      callOp.getGridz().getDefiningOp(),
                                      blockIdzOp.getOperation());
      });
      // block index
      reg->walk([&](NVVM::BlockIdXOp blkIdxOp) {
        attachConstantRangeIfConstant(ctx, callOp.getBlockx().getDefiningOp(),
                                      blkIdxOp.getOperation());
      });
      reg->walk([&](NVVM::BlockIdYOp blkIdyOp) {
        attachConstantRangeIfConstant(ctx, callOp.getBlocky().getDefiningOp(),
                                      blkIdyOp.getOperation());
      });
      reg->walk([&](NVVM::BlockIdZOp blkIdzOp) {
        attachConstantRangeIfConstant(ctx, callOp.getBlockz().getDefiningOp(),
                                      blkIdzOp.getOperation());
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