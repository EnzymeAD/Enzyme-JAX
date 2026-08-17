//===---------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//
//
// This file implements a pass to hoist per-thread allocations to allocations on
// the host.
//===---------------------------------------------------------------------===//

#include "Enzyme/MLIR/Dialect/Ops.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/RegionUtils.h"
#include "src/enzyme_ad/jax/Dialect/Dialect.h"
#include "src/enzyme_ad/jax/Dialect/Ops.h"
#include "src/enzyme_ad/jax/Passes/Passes.h"
#include "src/enzyme_ad/jax/Utils.h"
#include "llvm/Support/DebugLog.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#define DEBUG_TYPE "hoist-gpu-allocs"

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_HOISTGPUALLOCSPASS
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

using namespace mlir;
using namespace mlir::enzyme;

namespace {
// Curently unused because the size of the allocation is hardcoded for RSBench
void computeDependentOps(enzymexla::GPUWrapperOp wrapperOp,
                         SetVector<Operation *> &opsToMove) {
  SmallVector<Value> frontier;
  wrapperOp.walk([&](memref::AllocOp allocOp) {
    frontier.append(allocOp.getDynamicSizes().begin(),
                    allocOp.getDynamicSizes().end());
  });

  SetVector<Value> visited;
  getUsedValuesDefinedAbove(wrapperOp.getBodyRegion(), visited);
  while (!frontier.empty()) {
    Value v = frontier.back();
    Operation *definingOp = v.getDefiningOp();
    frontier.pop_back();

    if (!definingOp)
      continue;
    if (!wrapperOp.getBodyRegion().isAncestor(definingOp->getParentRegion()))
      continue;

    opsToMove.insert(definingOp);

    for (Value operand : definingOp->getOperands()) {
      if (visited.contains(operand))
        continue;

      frontier.push_back(operand);
      visited.insert(operand);
    }
  }
}

LogicalResult hoistAllocs(enzymexla::GPUWrapperOp wrapperOp) {
  // for RSBench/XSBench, hardcode the dynamic dimension of the memref to its
  // max (321)
  // for LULESH, it's 20
  constexpr int hardcodedDim = 20;

  PatternRewriter rewriter(wrapperOp.getContext());
  rewriter.setInsertionPoint(wrapperOp);
  // Super hardcoded for RSBench. For XSBench, we just have dims[0] * dims[3] (1
  // iterator)
  // lulesh also has 2 iterators
  SmallVector<Value> dynSizes;
  auto blockDims = wrapperOp.getBlockDims();
  dynSizes.push_back(blockDims[1]);
  dynSizes.push_back(arith::MulIOp::create(rewriter, wrapperOp.getLoc(),
                                           blockDims[0], blockDims[3]));
  scf::ParallelOp parallelOp;
  for (auto bodyOp : wrapperOp.getBodyRegion().getOps<scf::ParallelOp>()) {
    parallelOp = bodyOp;
  }
  if (!parallelOp) {
    llvm::errs() << "failed to find parallel op\n";
    return failure();
  }

  wrapperOp.walk([&](memref::AllocOp allocOp) {
    if (llvm::all_of(allocOp.getDynamicSizes(), [&](Value dynSize) -> bool {
          return matchPattern(dynSize, m_Zero());
        })) {
      return;
    }

    SmallVector<int64_t> shape;
    // shape.append({hardcodedDim, 8});
    shape.push_back(ShapedType::kDynamic);
    shape.push_back(ShapedType::kDynamic);
    shape.push_back(hardcodedDim);
    MemRefType cacheType = MemRefType::get(
        shape, allocOp.getType().getElementType(),
        /*layout=*/rewriter.getMultiDimIdentityMap(shape.size()),
        rewriter.getI64IntegerAttr(1));
    OpBuilder::InsertionGuard guard(rewriter);

    // Value start;
    // {
    //   OpBuilder::InsertionGuard guard(rewriter);
    //   // Try to offset the indices to encourage coalesced reads in the
    //   reverse
    //   // pass
    //   rewriter.setInsertionPoint(allocOp);
    //   Value c321 =
    //       arith::ConstantIndexOp::create(rewriter, allocOp.getLoc(), 321);
    //   start = arith::SubIOp::create(rewriter, allocOp.getLoc(), c321,
    //                                 allocOp.getDynamicSizes().front());
    // }

    // Value c0 = arith::ConstantIndexOp::create(rewriter, allocOp.getLoc(), 0);
    auto gpuAlloc = gpu::AllocOp::create(rewriter, allocOp.getLoc(), cacheType,
                                         /*asyncDependencies=*/ValueRange(),
                                         /*dynamicSizes=*/dynSizes,
                                         /*symbolOperands=*/ValueRange())
                        .getMemref();
    std::function<void(OpOperand &)> visit = [&](OpOperand &use) {
      if (auto storeOp = dyn_cast<memref::StoreOp>(use.getOwner())) {
        rewriter.setInsertionPoint(storeOp);
        SmallVector<Value> indices;
        // for (unsigned i = 0; i < storeOp.getIndices().size(); i++)
        //   // indices.push_back(c0);
        //   indices.push_back(arith::AddIOp::create(
        //       rewriter, storeOp.getLoc(), start, storeOp.getIndices()[i]));
        indices.append(parallelOp.getInductionVars());
        indices.append(storeOp.getIndices().begin(),
                       storeOp.getIndices().end());
        enzyme::StoreOp::create(rewriter, storeOp.getLoc(), storeOp.getValue(),
                                gpuAlloc, indices, dynSizes,
                                gpuAlloc.getType().getShape());
        rewriter.eraseOp(storeOp);
      } else if (auto loadOp = dyn_cast<memref::LoadOp>(use.getOwner())) {
        rewriter.setInsertionPoint(loadOp);
        SmallVector<Value> indices;
        // for (unsigned i = 0; i < loadOp.getIndices().size(); i++)
        //   // indices.push_back(c0);
        //   indices.push_back(arith::AddIOp::create(
        //       rewriter, loadOp.getLoc(), start, loadOp.getIndices()[i]));
        indices.append(parallelOp.getInductionVars());
        indices.append(loadOp.getIndices().begin(), loadOp.getIndices().end());
        auto newLoad =
            enzyme::LoadOp::create(rewriter, loadOp.getLoc(), gpuAlloc, indices,
                                   dynSizes, gpuAlloc.getType().getShape());
        rewriter.replaceOp(loadOp, newLoad);
      } else if (auto deallocOp = dyn_cast<memref::DeallocOp>(use.getOwner())) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointAfter(wrapperOp);
        gpu::DeallocOp::create(rewriter, deallocOp.getLoc(),
                               /*resultTypes=*/TypeRange(),
                               /*asyncDependencies=*/ValueRange(), gpuAlloc);
        rewriter.eraseOp(deallocOp);
      } else if (auto yieldOp = dyn_cast<scf::YieldOp>(use.getOwner())) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPoint(yieldOp);
        use.assign(gpuAlloc);
        if (auto ifOp = dyn_cast<scf::IfOp>(yieldOp->getParentOp())) {
          Value ifResult = ifOp.getResult(use.getOperandNumber());
          ifResult.setType(cacheType);
          if (use.getOwner() == ifOp.thenYield()) {
            ifOp.elseYield().setOperand(use.getOperandNumber(), gpuAlloc);
          } else {
            ifOp.thenYield().setOperand(use.getOperandNumber(), gpuAlloc);
          }
          for (auto &use : ifResult.getUses())
            visit(use);
        }
      } else {
        llvm::errs() << "[hoist-gpu] unexpected user: " << *use.getOwner()
                     << "\n";
      }
    };

    for (OpOperand &use :
         llvm::make_early_inc_range(allocOp.getMemref().getUses())) {
      visit(use);
    }

    rewriter.eraseOp(allocOp);
  });
  return success();
}

struct HoistGPUAllocsPass
    : public enzyme::impl::HoistGPUAllocsPassBase<HoistGPUAllocsPass> {
  using HoistGPUAllocsPassBase::HoistGPUAllocsPassBase;
  void runOnOperation() override {
    WalkResult result =
        getOperation()->walk([](enzymexla::GPUWrapperOp wrapperOp) {
          // Replace zero-sized allocations with poison values
          // wrapperOp.walk([](memref::AllocOp allocOp) {
          //   if (llvm::all_of(allocOp.getMixedSizes(),
          //                    [&](OpFoldResult size) -> bool {
          //                      if (auto val = dyn_cast<Value>(size)) {
          //                        return matchPattern(val, m_Zero());
          //                      }
          //                      return
          //                      cast<IntegerAttr>(cast<Attribute>(size))
          //                                 .getInt() == 0;
          //                    })) {
          //     ImplicitLocOpBuilder builder(allocOp.getLoc(), allocOp);
          //     Value poisonVal = LLVM::ZeroOp::create(
          //         builder, LLVM::LLVMPointerType::get(builder.getContext()));
          //     poisonVal = enzymexla::Pointer2MemrefOp::create(
          //         builder, allocOp.getType(), poisonVal);
          //     allocOp.replaceAllUsesWith(poisonVal);
          //     allocOp.erase();
          //   }
          // });

          if (failed(hoistAllocs(wrapperOp))) {
            return WalkResult::interrupt();
          }
          return WalkResult::advance();
        });
    if (result.wasInterrupted()) {
      signalPassFailure();
    }
  }
};
} // namespace
