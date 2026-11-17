//===- LoopUndistribute.cpp - Fuse & Interchange GPU Block Loops -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/Polymer/Transforms/LoopUndistribute.h"

#include "mlir/Dialect/Affine/IR/AffineMemoryOpInterfaces.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"

#include "src/enzyme_ad/jax/Dialect/Ops.h"

using namespace mlir;

namespace polymer {

bool isGridParallel(Operation *op) {
  return op && op->hasAttr(kGpuGridParAttr);
}

bool isBlockParallel(Operation *op) {
  return op && op->hasAttr(kGpuBlockParAttr);
}

affine::AffineParallelOp getAffineGridParallel(Operation *op) {
  auto par = dyn_cast_or_null<affine::AffineParallelOp>(op);
  return (par && isGridParallel(par)) ? par : nullptr;
}

affine::AffineParallelOp getAffineBlockParallel(Operation *op) {
  auto par = dyn_cast_or_null<affine::AffineParallelOp>(op);
  return (par && isBlockParallel(par)) ? par : nullptr;
}

namespace {

static bool areEquiv(affine::AffineParallelOp a, affine::AffineParallelOp b) {
  return a.getLowerBoundsValueMap() == b.getLowerBoundsValueMap() &&
         a.getUpperBoundsValueMap() == b.getUpperBoundsValueMap() &&
         a.getSteps() == b.getSteps();
}

static Value stripViews(Value v) {
  while (v) {
    if (auto view = v.getDefiningOp<ViewLikeOpInterface>()) {
      v = view.getViewSource();
      continue;
    }
    if (auto castOp = v.getDefiningOp<memref::MemorySpaceCastOp>()) {
      v = castOp.getSource();
      continue;
    }
    if (auto castOp = v.getDefiningOp<memref::CastOp>()) {
      v = castOp.getSource();
      continue;
    }
    break;
  }
  return v;
}

struct RegionMemSummary {
  bool hasUnknownSideEffects = false;
  SmallPtrSet<Value, 4> reads;
  SmallPtrSet<Value, 4> writes;
};

static bool isThreadPrivateMemref(Value base) {
  if (!base)
    return false;
  auto allocaOp = base.getDefiningOp<memref::AllocaOp>();
  if (!allocaOp)
    return false;
  if (allocaOp->hasAttr("polymer.private"))
    return true;
  for (Operation *p = allocaOp->getParentOp(); p; p = p->getParentOp()) {
    if (isBlockParallel(p))
      return true;
    if (isGridParallel(p))
      break;
  }
  return false;
}

static RegionMemSummary summarizeMemory(Operation *rootOp) {
  RegionMemSummary summary;
  auto recordRead = [&](Value memref) {
    Value base = stripViews(memref);
    if (!isThreadPrivateMemref(base))
      summary.reads.insert(base);
  };
  auto recordWrite = [&](Value memref) {
    Value base = stripViews(memref);
    if (!isThreadPrivateMemref(base))
      summary.writes.insert(base);
  };
  rootOp->walk([&](Operation *op) {
    if (op == rootOp ||
        isa<affine::AffineForOp, affine::AffineIfOp, affine::AffineParallelOp,
            affine::AffineYieldOp, scf::ForOp, scf::IfOp, scf::YieldOp>(op))
      return;

    if (isa<enzymexla::BarrierOp>(op))
      return;

    if (auto readOp = dyn_cast<affine::AffineReadOpInterface>(op)) {
      recordRead(readOp.getMemRef());
      return;
    }
    if (auto writeOp = dyn_cast<affine::AffineWriteOpInterface>(op)) {
      recordWrite(writeOp.getMemRef());
      return;
    }
    if (auto loadOp = dyn_cast<memref::LoadOp>(op)) {
      recordRead(loadOp.getMemRef());
      return;
    }
    if (auto storeOp = dyn_cast<memref::StoreOp>(op)) {
      recordWrite(storeOp.getMemRef());
      return;
    }
    if (isa<memref::AllocaOp>(op))
      return;

    if (!isMemoryEffectFree(op))
      summary.hasUnknownSideEffects = true;
  });
  return summary;
}

static bool provablyDisjointBaseMemrefs(Value a, Value b) {
  if (!a || !b)
    return false;
  if (a == b)
    return false;
  bool aAlloca = a.getDefiningOp<memref::AllocaOp>() != nullptr;
  bool bAlloca = b.getDefiningOp<memref::AllocaOp>() != nullptr;
  // An alloca cannot alias another distinct alloca or an incoming argument.
  if (aAlloca || bAlloca)
    return true;
  return false;
}

static bool needsBarrierBetween(Operation *first, Operation *second) {
  RegionMemSummary s1 = summarizeMemory(first);
  RegionMemSummary s2 = summarizeMemory(second);
  if (s1.hasUnknownSideEffects || s2.hasUnknownSideEffects)
    return true;

  auto mayConflict = [&](const SmallPtrSetImpl<Value> &setA,
                         const SmallPtrSetImpl<Value> &setB) {
    for (Value a : setA) {
      for (Value b : setB) {
        if (!provablyDisjointBaseMemrefs(a, b))
          return true;
      }
    }
    return false;
  };

  // RAW, WAR, WAW across threads between `first` and `second`.
  if (mayConflict(s1.writes, s2.reads))
    return true;
  if (mayConflict(s1.reads, s2.writes))
    return true;
  if (mayConflict(s1.writes, s2.writes))
    return true;
  return false;
}

static LogicalResult parallelizePreceding(affine::AffineParallelOp parOp,
                                          RewriterBase &rewriter) {
  Block *block = parOp->getBlock();

  // Do not sink operations that feed parOp's own bounds.
  SmallPtrSet<Operation *, 4> boundDefs;
  SmallVector<Value, 4> worklist(parOp->getOperands().begin(),
                                 parOp->getOperands().end());
  while (!worklist.empty()) {
    Value v = worklist.pop_back_val();
    Operation *def = v.getDefiningOp();
    if (!def || def->getBlock() != block)
      continue;
    if (boundDefs.insert(def).second)
      worklist.append(def->getOperands().begin(), def->getOperands().end());
  }

  // Stop at any preceding block-parallel op so we don't clone across sibling
  // parallel stages unnecessarily.
  Block::iterator startIt = block->begin();
  for (auto it = parOp->getIterator(); it != block->begin();) {
    --it;
    if (isBlockParallel(&*it)) {
      startIt = std::next(it);
      break;
    }
  }

  bool changed = false;
  IRMapping mapping;
  rewriter.setInsertionPointToStart(parOp.getBody());
  for (Operation &op : llvm::make_range(startIt, parOp->getIterator())) {
    if (boundDefs.contains(&op))
      continue;
    if (isMemoryEffectFree(&op) && !isBlockParallel(&op)) {
      Operation *cloned = rewriter.clone(op, mapping);
      for (auto [oldRes, newRes] :
           llvm::zip(op.getResults(), cloned->getResults())) {
        oldRes.replaceUsesWithIf(newRes, [&](OpOperand &use) {
          return parOp->isProperAncestor(use.getOwner());
        });
      }
      changed = true;
    }
  }

  for (Operation *op = parOp->getPrevNode(); op != nullptr;) {
    Operation *prev = op->getPrevNode();
    if (op == &*startIt) {
      if (isOpTriviallyDead(op)) {
        rewriter.eraseOp(op);
        changed = true;
      }
      break;
    }
    if (isOpTriviallyDead(op)) {
      rewriter.eraseOp(op);
      changed = true;
    }
    op = prev;
  }

  return success(changed);
}

static bool regionNeedsLoopBackBarrier(Region &region) {
  Operation *op = region.getParentOp();
  if (isa<affine::AffineForOp, scf::ForOp>(op))
    return needsBarrierBetween(op, op);
  if (isa<affine::AffineIfOp, scf::IfOp>(op))
    return false;
  return false;
}

static LogicalResult interchange(Operation *parent, RewriterBase &rewriter) {
  if (!parent || isGridParallel(parent))
    return failure();
  if (!isa<affine::AffineForOp, scf::ForOp, affine::AffineIfOp, scf::IfOp>(
          parent))
    return failure();
  if (parent->getNumResults() != 0)
    return failure();

  SmallVector<affine::AffineParallelOp, 2> pars;
  affine::AffineParallelOp canonicalPar = nullptr;

  for (Region &region : parent->getRegions()) {
    if (region.empty()) {
      pars.push_back(nullptr);
      continue;
    }
    if (!llvm::hasSingleElement(region))
      return failure();
    Block &body = region.front();
    Operation *term = body.getTerminator();
    if (term->getNumResults() != 0 || term->getNumOperands() != 0)
      return failure();

    if (body.getOperations().size() == 1) {
      pars.push_back(nullptr);
    } else if (body.getOperations().size() == 2) {
      auto thisPar = getAffineBlockParallel(&body.front());
      if (!thisPar)
        return failure();
      // Bounds of `thisPar` must not depend on `parent`'s block arguments.
      for (Value opr : thisPar->getOperands()) {
        if (auto ba = dyn_cast<BlockArgument>(opr))
          if (ba.getOwner() == &body)
            return failure();
      }
      if (canonicalPar && !areEquiv(canonicalPar, thisPar))
        return failure();
      canonicalPar = thisPar;
      pars.push_back(thisPar);
    } else {
      return failure();
    }
  }

  if (!canonicalPar)
    return failure();

  Location loc = parent->getLoc();
  rewriter.setInsertionPoint(parent);
  auto newPar = cast<affine::AffineParallelOp>(
      rewriter.cloneWithoutRegions(*canonicalPar));
  Block *newParBody = rewriter.createBlock(
      &newPar.getRegion(), newPar.getRegion().begin(),
      canonicalPar.getBody()->getArgumentTypes(),
      llvm::map_to_vector(canonicalPar.getBody()->getArguments(),
                          [](BlockArgument a) { return a.getLoc(); }));

  Operation *child = rewriter.cloneWithoutRegions(*parent);
  affine::AffineYieldOp::create(rewriter, loc);

  for (auto [idx, regions] :
       llvm::enumerate(llvm::zip(child->getRegions(), parent->getRegions()))) {
    auto &[childRegion, parentRegion] = regions;
    if (parentRegion.empty())
      continue;

    bool needsBarrier = regionNeedsLoopBackBarrier(parentRegion);

    Block *parentBody = &parentRegion.front();
    Block *childBody = rewriter.createBlock(
        &childRegion, childRegion.begin(), parentBody->getArgumentTypes(),
        llvm::map_to_vector(parentBody->getArguments(),
                            [](BlockArgument a) { return a.getLoc(); }));

    if (affine::AffineParallelOp innerPar = pars[idx]) {
      Block *innerParBody = innerPar.getBody();
      rewriter.eraseOp(innerParBody->getTerminator());
      rewriter.inlineBlockBefore(innerParBody, childBody, childBody->end(),
                                 newParBody->getArguments());
    }

    rewriter.setInsertionPointToEnd(childBody);
    if (needsBarrier) {
      enzymexla::BarrierOp::create(rewriter, newPar.getLoc(),
                                   newParBody->getArguments());
    }
    rewriter.clone(*parentBody->getTerminator());

    for (auto [oldArg, newArg] :
         llvm::zip(parentBody->getArguments(), childBody->getArguments()))
      rewriter.replaceAllUsesWith(oldArg, newArg);
  }

  rewriter.eraseOp(parent);
  return success();
}

static LogicalResult fuse(affine::AffineParallelOp par,
                          RewriterBase &rewriter) {
  if (!isBlockParallel(par))
    return failure();
  auto nextPar = getAffineBlockParallel(par->getNextNode());
  if (!nextPar)
    return failure();
  if (!areEquiv(par, nextPar))
    return failure();

  bool insertBarrier = needsBarrierBetween(par, nextPar);

  rewriter.eraseOp(nextPar.getBody()->getTerminator());
  rewriter.setInsertionPoint(par.getBody()->getTerminator());
  if (insertBarrier) {
    enzymexla::BarrierOp::create(rewriter, nextPar.getLoc(),
                                 par.getBody()->getArguments());
  }
  rewriter.inlineBlockBefore(nextPar.getBody(), par.getBody()->getTerminator(),
                             par.getBody()->getArguments());
  rewriter.eraseOp(nextPar);
  return success();
}

static SmallVector<affine::AffineParallelOp>
findTopLevelBlockPars(Operation *gridPar) {
  SmallVector<affine::AffineParallelOp> blockPars;
  gridPar->walk([&](Operation *op) {
    if (op == gridPar)
      return WalkResult::advance();
    if (auto par = getAffineBlockParallel(op)) {
      blockPars.push_back(par);
      return WalkResult::skip();
    }
    return WalkResult::advance();
  });
  return blockPars;
}

static LogicalResult undistributeGrid(Operation *gridPar) {
  IRRewriter rewriter(gridPar->getContext());

  while (true) {
    auto blockPars = findTopLevelBlockPars(gridPar);
    if (blockPars.empty())
      return success();

    bool changed = false;

    for (auto blockPar : blockPars) {
      if (succeeded(interchange(blockPar->getParentOp(), rewriter))) {
        changed = true;
        break;
      }
    }
    if (changed)
      continue;

    for (auto blockPar : blockPars) {
      if (succeeded(fuse(blockPar, rewriter))) {
        changed = true;
        break;
      }
    }
    if (changed)
      continue;

    for (auto blockPar : blockPars) {
      if (succeeded(parallelizePreceding(blockPar, rewriter))) {
        changed = true;
        break;
      }
    }
    if (changed)
      continue;

    break;
  }

  return success();
}

} // namespace

LogicalResult undistributeFunctions(Operation *root) {
  SmallVector<Operation *> gridPars;
  root->walk([&](Operation *op) {
    if (isGridParallel(op))
      gridPars.push_back(op);
  });

  for (Operation *gridPar : gridPars) {
    if (failed(undistributeGrid(gridPar)))
      return failure();
  }
  return success();
}

} // namespace polymer
