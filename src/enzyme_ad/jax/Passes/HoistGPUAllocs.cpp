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

#include "mlir/IR/Verifier.h"

#include <queue>

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

static Value computeFlatIndex(ValueRange ivs, ValueRange sizes,
                              ImplicitLocOpBuilder &builder) {
  Value flatIndex = ivs[0];
  for (int64_t dim = 1; dim < ivs.size(); dim++) {
    Value bound = sizes[dim];
    flatIndex = arith::MulIOp::create(builder, flatIndex, bound);
    flatIndex = arith::AddIOp::create(builder, flatIndex, ivs[dim]);
  }
  return flatIndex;
}

static FailureOr<scf::ParallelOp>
findParallelOp(enzymexla::GPUWrapperOp wrapperOp) {
  for (auto &op : *wrapperOp.getBody()) {
    if (auto parallelOp = dyn_cast<scf::ParallelOp>(op)) {
      return parallelOp;
    }
  }
  return failure();
}

LogicalResult hoistAllocs(enzymexla::GPUWrapperOp wrapperOp) {
  // for RSBench/XSBench, hardcode the dynamic dimension of the memref to its
  // max (321)
  // for LULESH, it's 20
  constexpr int hardcodedDim = 321;

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
    shape.push_back(ShapedType::kDynamic);
    shape.push_back(ShapedType::kDynamic);
    shape.append({hardcodedDim, 8});
    // shape.push_back(hardcodedDim);
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

void generateLoopIterFuncs(Operation *moduleOp) {
  unsigned counter = 0;
  DenseSet<Operation *> processed;
  DenseSet<Value> processedVals;
  moduleOp->walk([&](enzymexla::GPUWrapperOp wrapperOp) {
    auto maybeParallelOp = findParallelOp(wrapperOp);
    if (failed(maybeParallelOp))
      return;
    scf::ParallelOp parallelOp = maybeParallelOp.value();
    SmallVector<memref::AllocOp> cacheAllocs;
    wrapperOp.walk([&](memref::AllocOp allocOp) {
      if (!allocOp->hasAttr("enzyme.cache_alloc"))
        return;
      cacheAllocs.push_back(allocOp);
    });

    if (cacheAllocs.empty())
      return;
    SetVector<Operation *> ops;
    // Function arguments used to compute the number of iterations
    SmallVector<BlockArgument> args;
    wrapperOp.walk([&](scf::ForOp forOp) {
      // Ignore inner loops of nested loops
      if (forOp->getParentOfType<scf::ForOp>()) {
        return;
      }

      if (!processedVals.insert(forOp.getUpperBound()).second)
        return;

      std::queue<Value> worklist;
      DenseSet<Value> visited;
      worklist.push(forOp.getUpperBound());
      visited.insert(forOp.getUpperBound());
      SetVector<Value> freeVars;
      while (!worklist.empty()) {
        Value curr = worklist.front();
        worklist.pop();

        if (auto arg = dyn_cast<BlockArgument>(curr)) {
          if (arg.getOwner()->getParentOp() == parallelOp) {
            // We don't need to do anything here
            // threadIDs.insert(arg);
          } else if (arg.getOwner()->getParent()->isAncestor(
                         &wrapperOp.getBodyRegion())) {
            freeVars.insert(arg);
          } else {
            llvm::errs() << "[hoist] unexpected block arg: " << arg << "\n";
          }
        }
        if (Operation *definingOp = curr.getDefiningOp()) {
          if (definingOp->isAncestor(wrapperOp)) {
            freeVars.insert(curr);
          } else {
            ops.insert(definingOp);
            SetVector<Value> definedAbove;
            getUsedValuesDefinedAbove(definingOp->getRegions(), definedAbove);
            for (Value val : definedAbove) {
              if (visited.insert(val).second) {
                worklist.push(val);
              }
            }
            for (Value operand : definingOp->getOperands()) {
              if (visited.insert(operand).second) {
                worklist.push(operand);
              }
            }
          }
        }
      }
      ops = topologicalSort(ops);
      SmallVector<Value> args;
      args.append(parallelOp.getInductionVars());
      args.append(freeVars.begin(), freeVars.end());
      SmallVector<Type> argTypes =
          llvm::map_to_vector(args, [](Value arg) { return arg.getType(); });
      OpBuilder builder(wrapperOp);
      builder.setInsertionPoint(
          wrapperOp->getParentOfType<FunctionOpInterface>());
      auto funcType =
          builder.getFunctionType(argTypes, forOp.getUpperBound().getType());
      std::string funcName =
          (Twine("__enzyme_computeloopiter") + Twine(counter)).str();
      counter++;
      auto loopIterFunc =
          func::FuncOp::create(builder, forOp.getLoc(), funcName, funcType);
      loopIterFunc.setPrivate();
      Block *entryBlock = loopIterFunc.addEntryBlock();
      builder.setInsertionPointToStart(entryBlock);
      IRMapping map;
      map.map(args, entryBlock->getArguments());

      for (Operation *op : ops) {
        builder.clone(*op, map);
      }
      func::ReturnOp::create(builder, forOp.getLoc(),
                             map.lookup(forOp.getUpperBound()));

      // Get the total number of threads
      // The cumulative sum will be of size nThreads + 1.
      // e.g. 3 threads that iterate [4, 2, 3], the cumulative sum will be
      // [0, 4, (4+2), (4+2+3)] => [0, 4, 6, 9]
      builder.setInsertionPoint(wrapperOp);
      Location loc = wrapperOp.getLoc();
      Value c1 = arith::ConstantIndexOp::create(builder, loc, 1);
      Value total = c1;
      for (Value blockDim : wrapperOp.getBlockDims()) {
        total = arith::MulIOp::create(builder, loc, total, blockDim);
      }
      total = arith::AddIOp::create(builder, loc, total, c1);
      auto deviceMemorySpaceAttr = builder.getI64IntegerAttr(1);
      auto deviceIndices = gpu::AllocOp::create(
          builder, loc,
          MemRefType::get(ShapedType::kDynamic, builder.getI32Type(),
                          builder.getMultiDimIdentityMap(1),
                          deviceMemorySpaceAttr),
          /*asyncDependencies=*/ValueRange(), total,
          /*symbolOperands=*/ValueRange());
      deviceIndices->setAttr("enzyme.device_indices", builder.getUnitAttr());

      {
        // Create a second wrapper op that just computes the number of times
        // each thread executes. Give it the same launch parameters and
        // iterators as the original wrapper.
        OpBuilder::InsertionGuard guard(builder);
        auto iterCountWrapper = enzymexla::GPUWrapperOp::create(
            builder, loc, wrapperOp.getBlockDims());
        // iterCountWrapper->setAttr("enzyme.iter_counts",
        // builder.getUnitAttr());
        IRMapping map;
        builder.setInsertionPointToStart(iterCountWrapper.getBody());
        for (Operation &op : wrapperOp.getBodyRegion().getOps()) {
          if (op.getNumRegions() == 0 &&
              &op != wrapperOp.getBody()->getTerminator()) {
            builder.clone(op, map);
          }
        }
        auto lowerBounds =
            llvm::map_to_vector(parallelOp.getLowerBound(), [&](Value val) {
              return map.lookupOrDefault(val);
            });
        auto upperBounds =
            llvm::map_to_vector(parallelOp.getUpperBound(), [&](Value val) {
              return map.lookupOrDefault(val);
            });
        auto steps = llvm::map_to_vector(parallelOp.getStep(), [&](Value val) {
          return map.lookupOrDefault(val);
        });
        scf::ParallelOp::create(
            builder, loc, lowerBounds, upperBounds, steps,
            [&](OpBuilder &builder, Location loc, ValueRange ivs) {
              for (unsigned i = 0; i < ivs.size(); i++) {
                args[i] = ivs[i];
              }

              Value iterCount =
                  func::CallOp::create(builder, wrapperOp.getLoc(),
                                       loopIterFunc, args)
                      .getResult(0);

              // Flatten the ivs
              ImplicitLocOpBuilder implBuilder(loc, builder);
              Value flatIV = computeFlatIndex(ivs, upperBounds, implBuilder);

              memref::StoreOp::create(builder, loc, iterCount,
                                      deviceIndices.getMemref(), flatIV);

              // Determine if this is the last thread
              Value lastIdx = arith::SubIOp::create(builder, loc, total, c1);
              Value lastIV = arith::SubIOp::create(builder, loc, lastIdx, c1);
              Value cond = arith::CmpIOp::create(
                  builder, loc, arith::CmpIPredicate::eq, flatIV, lastIV);
              // Have the last thread also write a 0 to the last index
              scf::IfOp::create(
                  builder, loc, cond, [&](OpBuilder &builder, Location loc) {
                    Value c0i32 = arith::ConstantOp::create(
                        builder, loc, builder.getI32IntegerAttr(0));
                    memref::StoreOp::create(builder, loc, c0i32,
                                            deviceIndices.getMemref(), lastIdx);
                    scf::YieldOp::create(builder, loc);
                  });
              scf::ReduceOp::create(builder, loc);
            });

        // Emit the exclusive sum
        builder.setInsertionPoint(
            wrapperOp->getParentOfType<FunctionOpInterface>());
        auto ptrType = LLVM::LLVMPointerType::get(builder.getContext());
        auto funcType =
            builder.getFunctionType({builder.getI32Type(), ptrType}, {});
        auto exclSumFunc = func::FuncOp::create(
            builder, wrapperOp.getLoc(), "cubDeviceExclusiveSum", funcType);
        exclSumFunc.setPrivate();

        builder.setInsertionPoint(wrapperOp);
        Value totalI32 = arith::IndexCastOp::create(
            builder, loc, builder.getI32Type(), total);
        Value deviceIndicesPtr = enzymexla::Memref2PointerOp::create(
            builder, loc, ptrType, deviceIndices.getMemref());
        func::CallOp::create(builder, loc, exclSumFunc,
                             {totalI32, deviceIndicesPtr});

        // Figure out how much data to allocate in total. Copy the size back to
        // the host
        Value lastIdx = arith::SubIOp::create(builder, loc, total, c1);
        lastIdx = arith::IndexCastOp::create(builder, loc, builder.getI64Type(),
                                             lastIdx);
        Value gepRes = LLVM::GEPOp::create(
            builder, loc, ptrType, builder.getI32Type(), deviceIndicesPtr,
            lastIdx, LLVM::GEPNoWrapFlags::inbounds);
        gepRes = enzymexla::Pointer2MemrefOp::create(
            builder, loc, deviceIndices.getMemref().getType(), gepRes);
        auto elementType = deviceIndices.getMemref().getType().getElementType();
        Value dataSizeMem = memref::AllocaOp::create(
            builder, loc, MemRefType::get({}, elementType));
        Value sizeOfInt = arith::ConstantIndexOp::create(
            builder, loc, elementType.getIntOrFloatBitWidth() / 8);
        enzymexla::MemcpyOp::create(builder, loc, /*asyncToken=*/Type(),
                                    /*asyncDependencies=*/ValueRange(),
                                    dataSizeMem, gepRes, sizeOfInt);

        Value dataSize = memref::LoadOp::create(builder, loc, dataSizeMem);
        dataSize = arith::IndexCastOp::create(builder, loc,
                                              builder.getIndexType(), dataSize);
        for (memref::AllocOp allocOp : cacheAllocs) {
          builder.setInsertionPoint(wrapperOp);
          auto gpuAlloc = gpu::AllocOp::create(
              builder, loc,
              MemRefType::get(allocOp.getType().getShape(),
                              allocOp.getType().getElementType(),
                              builder.getMultiDimIdentityMap(1),
                              deviceMemorySpaceAttr),
              /*asyncDependencies=*/ValueRange(), dataSize,
              /*symbolOperands=*/ValueRange());
          gpuAlloc->setAttr("enzyme.hoisted_cache", builder.getUnitAttr());

          // Replace loads and stores
          builder.setInsertionPointToStart(parallelOp.getBody());
          ImplicitLocOpBuilder locBuilder(loc, builder);
          Value flatIV =
              computeFlatIndex(parallelOp.getInductionVars(),
                               parallelOp.getUpperBound(), locBuilder);
          Value startIdx = memref::LoadOp::create(
              builder, loc, deviceIndices.getMemref(), flatIV);
          startIdx = arith::IndexCastOp::create(
              builder, loc, builder.getIndexType(), startIdx);

          PatternRewriter rewriter(builder.getContext());

          std::function<void(OpOperand &)> visit = [&](OpOperand &use) {
            if (auto storeOp = dyn_cast<memref::StoreOp>(use.getOwner())) {
              rewriter.setInsertionPoint(storeOp);
              SmallVector<Value> indices;
              indices.append(storeOp.getIndices().begin(),
                             storeOp.getIndices().end());
              indices[0] = arith::AddIOp::create(rewriter, loc, startIdx,
                                                 storeOp.getIndices().front());
              memref::StoreOp::create(rewriter, storeOp.getLoc(),
                                      storeOp.getValue(), gpuAlloc.getMemref(),
                                      indices);
              rewriter.eraseOp(storeOp);
            } else if (auto loadOp = dyn_cast<memref::LoadOp>(use.getOwner())) {
              rewriter.setInsertionPoint(loadOp);
              SmallVector<Value> indices;
              indices.append(loadOp.getIndices().begin(),
                             loadOp.getIndices().end());
              indices[0] = arith::AddIOp::create(rewriter, loc, startIdx,
                                                 loadOp.getIndices().front());
              auto newLoad = memref::LoadOp::create(
                  rewriter, loc, gpuAlloc.getMemref(), indices);
              rewriter.replaceOp(loadOp, newLoad);
            } else if (auto deallocOp =
                           dyn_cast<memref::DeallocOp>(use.getOwner())) {
              OpBuilder::InsertionGuard guard(rewriter);
              rewriter.setInsertionPointAfter(wrapperOp);
              gpu::DeallocOp::create(rewriter, deallocOp.getLoc(),
                                     /*resultTypes=*/TypeRange(),
                                     /*asyncDependencies=*/ValueRange(),
                                     gpuAlloc.getMemref());
              rewriter.eraseOp(deallocOp);
            } else {
              llvm::errs() << "[debug] found use of alloc op: "
                           << *use.getOwner() << "\n";
            }
          };
          for (OpOperand &use :
               llvm::make_early_inc_range(allocOp.getMemref().getUses())) {
            visit(use);
          }
          allocOp.erase();
        }
      }
    });
  });
}

struct HoistGPUAllocsPass
    : public enzyme::impl::HoistGPUAllocsPassBase<HoistGPUAllocsPass> {
  using HoistGPUAllocsPassBase::HoistGPUAllocsPassBase;
  void runOnOperation() override {
    Operation *moduleOp = getOperation();
    generateLoopIterFuncs(moduleOp);
    WalkResult result = moduleOp->walk([](enzymexla::GPUWrapperOp wrapperOp) {
      // Replace zero-sized allocations with poison values
      // wrapperOp.walk([](memref::AllocOp allocOp) {
      //   if (llvm::all_of(
      //           allocOp.getMixedSizes(), [&](OpFoldResult size) -> bool {
      //             if (auto val = dyn_cast<Value>(size)) {
      //               return matchPattern(val, m_Zero());
      //             }
      //             return cast<IntegerAttr>(cast<Attribute>(size)).getInt()
      //             == 0;
      //           })) {
      //     ImplicitLocOpBuilder builder(allocOp.getLoc(), allocOp);
      //     Value poisonVal = LLVM::ZeroOp::create(
      //         builder, LLVM::LLVMPointerType::get(builder.getContext()));
      //     poisonVal = enzymexla::Pointer2MemrefOp::create(
      //         builder, allocOp.getType(), poisonVal);
      //     allocOp.replaceAllUsesWith(poisonVal);
      //     allocOp.erase();
      //   }
      // });

      // if (failed(hoistAllocs(wrapperOp))) {
      //   return WalkResult::interrupt();
      // }
      return WalkResult::advance();
    });
    if (result.wasInterrupted()) {
      signalPassFailure();
    }
  }
};
} // namespace
