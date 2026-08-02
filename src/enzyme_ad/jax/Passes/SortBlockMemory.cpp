//===- SortBlockMemory.cpp - Cluster non-overlapping block accesses -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//
//
// This file implements a pass that hoists loads to the start of their block and
// sinks stores to the end, whenever the accesses moved across can be proven not
// to overlap.
//
// This is aimed at scatter-like code -- most notably what `remove-atomics`
// leaves behind, a run of (load, modify, store) triples to disjoint locations.
// Emitted in place, each load must wait for the preceding store, so the run
// becomes a chain of dependent round trips to memory. Clustering the loads lets
// them all be in flight at once.
//
// Unlike `sort-memory`, legality is decided per access rather than for a whole
// region: a load is hoisted past exactly those stores it provably misses, so a
// single unanalyzable access no longer forfeits the transform for everything
// around it.
//
//===---------------------------------------------------------------------===//

#include "Enzyme/MLIR/Interfaces/Utils.h"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "src/enzyme_ad/jax/Passes/Passes.h"

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_SORTBLOCKMEMORYPASS
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

using namespace mlir;
using namespace mlir::affine;

namespace {

// Number of loops enclosing `op`, i.e. the depth at which two accesses in the
// same iteration are compared.
unsigned getNestingDepth(Operation *op) {
  unsigned depth = 0;
  for (Operation *cur = op->getParentOp(); cur; cur = cur->getParentOp()) {
    if (isa<AffineForOp>(cur))
      depth++;
    if (auto par = dyn_cast<AffineParallelOp>(cur))
      depth += par.getNumDims();
  }
  return depth;
}

Value getMemref(Operation *op) {
  if (auto ld = dyn_cast<AffineLoadOp>(op))
    return ld.getMemref();
  if (auto st = dyn_cast<AffineStoreOp>(op))
    return st.getMemref();
  return nullptr;
}

// Whether `a` and `b` may touch the same element. Both must be affine accesses
// living in the same block.
bool mayOverlap(Operation *a, Operation *b) {
  Value memA = getMemref(a), memB = getMemref(b);
  if (!memA || !memB)
    return true;

  // Distinct underlying buffers can never overlap, whatever the indices are.
  if (memA != memB && !enzyme::oputils::mayAlias(memA, memB))
    return false;

  // Same buffer: ask whether the two accesses can name the same element within
  // one iteration of the enclosing loops.
  MemRefAccess accessA(a), accessB(b);
  unsigned depth = getNestingDepth(a) + 1;
  DependenceResult result =
      checkMemrefAccessDependence(accessA, accessB, depth);
  return result.value != DependenceResult::NoDependence;
}

// An op that neither reads nor writes memory can always be moved across; one
// whose effects we cannot enumerate pins everything.
bool isMemoryOpaque(Operation *op) {
  if (isa<AffineLoadOp, AffineStoreOp>(op))
    return false;
  if (isPure(op))
    return false;
  auto iface = dyn_cast<MemoryEffectOpInterface>(op);
  if (!iface)
    return true;
  return iface.hasEffect<MemoryEffects::Read>() ||
         iface.hasEffect<MemoryEffects::Write>() ||
         iface.hasEffect<MemoryEffects::Free>();
}

// The earliest point in `block` that all of `op`'s operands dominate.
Operation *earliestOperandPoint(Operation *op, Block *block) {
  Operation *point = &block->front();
  for (Value operand : op->getOperands()) {
    Operation *def = operand.getDefiningOp();
    if (!def || def->getBlock() != block)
      continue;
    if (point->isBeforeInBlock(def))
      point = def->getNextNode();
  }
  return point;
}

void sortBlock(Block *block, unsigned window) {
  SmallVector<Operation *> loads, stores;
  for (Operation &op : *block) {
    if (isa<AffineLoadOp>(op))
      loads.push_back(&op);
    else if (isa<AffineStoreOp>(op))
      stores.push_back(&op);
  }
  if (loads.size() < 2 && stores.size() < 2)
    return;

  // Hoisting a load keeps its result live until its consumer, so `window` caps
  // how many may be in flight at once; without it a long run would inflate
  // register pressure enough to cost more than the latency it saves.
  if (window) {
    if (loads.size() > window)
      loads.resize(window);
    if (stores.size() > window)
      stores.erase(stores.begin(), stores.end() - window);
  }

  // Hoist loads, in order, as far up as both their operands and the accesses
  // in between allow. `cursor` is the last load already hoisted; keeping the
  // next one behind it preserves their original relative order.
  Operation *cursor = nullptr;
  for (Operation *load : loads) {
    Operation *dest = earliestOperandPoint(load, block);
    for (Operation *cur = dest; cur != load; cur = cur->getNextNode()) {
      if (isMemoryOpaque(cur) ||
          (isa<AffineStoreOp>(cur) && mayOverlap(cur, load)))
        dest = cur->getNextNode();
    }
    if (cursor) {
      Operation *afterCursor = cursor->getNextNode();
      if (dest->isBeforeInBlock(afterCursor))
        dest = afterCursor;
    }
    if (dest != load)
      load->moveBefore(dest);
    cursor = load;
  }

  // Sink stores, last first, each landing just ahead of the one sunk before it
  // so that their original relative order survives too.
  cursor = block->getTerminator();
  for (Operation *store : llvm::reverse(stores)) {
    Operation *dest = cursor;
    for (Operation *cur = store->getNextNode(); cur != cursor;
         cur = cur->getNextNode()) {
      // A store may not pass an access that could observe or overwrite it.
      if (isMemoryOpaque(cur) ||
          (isa<AffineLoadOp, AffineStoreOp>(cur) && mayOverlap(store, cur))) {
        dest = cur;
        break;
      }
    }
    if (dest != store->getNextNode())
      store->moveBefore(dest);
    cursor = store;
  }
}

struct SortBlockMemoryPass
    : public enzyme::impl::SortBlockMemoryPassBase<SortBlockMemoryPass> {
  using SortBlockMemoryPassBase::SortBlockMemoryPassBase;

  void runOnOperation() override {
    getOperation()->walk([&](Block *block) { sortBlock(block, window); });
  }
};

} // end anonymous namespace
