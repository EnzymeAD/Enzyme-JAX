//===- SortMemory.cpp - Print the MLIR module                     ------------
////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to print the MLIR module
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "src/enzyme_ad/jax/Passes/Passes.h"
#include "llvm/ADT/SetVector.h"

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_SORTMEMORY
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

using namespace mlir;
using namespace mlir::affine;

bool definedOutsideOrAt(Value v, Operation *op) {
  return !op->isProperAncestor(v.getParentBlock()->getParentOp());
}

bool affineCmp(AffineExpr lhs, AffineExpr rhs);

bool affineCmp(AffineMap lhs, AffineMap rhs) {
  if (lhs.getNumResults() < rhs.getNumResults())
    return true;
  if (rhs.getNumResults() < lhs.getNumResults())
    return false;

  if (lhs.getNumDims() < rhs.getNumDims())
    return true;
  if (rhs.getNumDims() < lhs.getNumDims())
    return false;

  if (lhs.getNumSymbols() < rhs.getNumSymbols())
    return true;
  if (rhs.getNumSymbols() < lhs.getNumSymbols())
    return false;

  for (auto &&[l, r] : llvm::zip_equal(lhs.getResults(), rhs.getResults())) {
    if (affineCmp(l, r))
      return true;
    if (affineCmp(r, l))
      return false;
  }

  return false;
}

bool affineCmpLoad(AffineLoadOp lhs, AffineLoadOp rhs) {
  return affineCmp(lhs.getMap(), rhs.getMap());
}

bool affineCmpStore(AffineStoreOp lhs, AffineStoreOp rhs) {
  return affineCmp(lhs.getMap(), rhs.getMap());
}

/// Returns the nesting depth of this statement, i.e., the number of loops
/// surrounding this statement.
static unsigned getNestingDepth(Operation *op) {
  Operation *currOp = op;
  unsigned depth = 0;
  while ((currOp = currOp->getParentOp())) {
    if (isa<AffineForOp>(currOp))
      depth++;
    if (auto parOp = dyn_cast<AffineParallelOp>(currOp))
      depth += parOp.getNumDims();
  }
  return depth;
}

void sortParallel(affine::AffineParallelOp par) {
  SmallVector<affine::AffineLoadOp> loads;
  SmallVector<affine::AffineStoreOp> stores;
  par->walk([&](affine::AffineLoadOp ld) { loads.push_back(ld); });
  par->walk([&](affine::AffineStoreOp store) { stores.push_back(store); });

  // Dep check depth within the loop would be number of enclosing loops + number
  // of IVs for this parallel loop + 1.
  unsigned depth = ::getNestingDepth(par) + par.getNumDims() + 1;

  for (auto ld : loads) {
    MemRefAccess dstAccess(ld);
    for (auto st : stores) {
      MemRefAccess srcAccess(st);

      // if there is a store dependent the load for the same memref, then it is
      // not valid to move loads at the beginning of the loop.
      DependenceResult result =
          checkMemrefAccessDependence(srcAccess, dstAccess, depth);

      if (result.value == DependenceResult::Failure) {
        return;
      }
      if (result.value == DependenceResult::HasDependence) {
        return;
      }
    }
  }

  SetVector<Value> memrefs;
  for (auto ld : loads) {
    memrefs.insert(ld.getMemref());
    for (auto v : ld.getMapOperands()) {
      if (!definedOutsideOrAt(v, par))
        return;
    }
  }
  for (auto st : stores) {
    memrefs.insert(st.getMemref());
    for (auto v : st.getMapOperands()) {
      if (!definedOutsideOrAt(v, par))
        return;
    }
  }
  for (auto m : memrefs) {
    if (!definedOutsideOrAt(m, par))
      return;
    for (auto u : m.getUsers()) {
      if (!isa<affine::AffineLoadOp, affine::AffineStoreOp>(u)) {
        return;
      }
    }
  }

  llvm::stable_sort(loads, affineCmpLoad);
  llvm::stable_sort(stores, affineCmpStore);

  Operation *first = &par.getBody()->front();
  for (auto ld : llvm::reverse(loads)) {
    if (ld->getParentOp() != par)
      continue;
    ld->moveBefore(first);
    first = ld;
  }

  Operation *last = par.getBody()->getTerminator();
  for (auto st : stores) {
    if (st->getParentOp() != par)
      continue;
    st->moveBefore(last);
  }
}

namespace {
struct SortMemory : public enzyme::impl::SortMemoryBase<SortMemory> {
  using SortMemoryBase::SortMemoryBase;

  void runOnOperation() override { getOperation()->walk(sortParallel); }
};

} // end anonymous namespace
