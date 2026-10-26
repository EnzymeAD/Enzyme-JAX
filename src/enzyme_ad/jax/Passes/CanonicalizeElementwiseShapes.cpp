//===- CanonicalizeElementwiseShapes.cpp ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Give each maximal chain of elementwise ops a single tensor shape, so the
// chain can be fused as one unit by the backend. See the pass description in
// Passes.td for why a chain ends up straddling two bitcast-equivalent shapes
// in the first place.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "src/enzyme_ad/jax/Passes/Passes.h"
#include "src/enzyme_ad/jax/Utils.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "canonicalize-elementwise-shapes"

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_CANONICALIZEELEMENTWISESHAPESPASS
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

using namespace mlir;
using namespace mlir::enzyme;

namespace {

using Shape = SmallVector<int64_t, 4>;

/// Follow `stablehlo.reshape` producers back to the first value that is not
/// produced by one. Every `stablehlo.reshape` preserves both the element count
/// and the linearized element order, so the result denotes the same buffer.
static Value stripReshapes(Value v) {
  while (auto reshape = v.getDefiningOp<stablehlo::ReshapeOp>())
    v = reshape.getOperand();
  return v;
}

static RankedTensorType getStaticRankedType(Value v) {
  auto ty = dyn_cast<RankedTensorType>(v.getType());
  if (!ty || !ty.hasStaticShape())
    return nullptr;
  return ty;
}

/// An op takes part in a chain if it is elementwise, has a single statically
/// shaped ranked tensor result and no regions.
static bool isChainCandidate(Operation *op) {
  if (!op || op->getNumResults() != 1 || op->getNumRegions() != 0)
    return false;
  if (!stablehlo::hasTraitElementwise(op))
    return false;
  if (!getStaticRankedType(op->getResult(0)))
    return false;
  // Generic op creation below reconstructs the op from its name, operands,
  // result type and attributes; anything carrying extra state is skipped.
  return op->getNumSuccessors() == 0;
}

/// Values reachable from `v` by walking through `stablehlo.reshape` users.
/// These are alternative views of the same buffer.
static void collectReshapeViews(Value v, SmallVectorImpl<Value> &views) {
  SmallVector<Value> worklist{v};
  DenseSet<Value> seen{v};
  while (!worklist.empty()) {
    Value cur = worklist.pop_back_val();
    views.push_back(cur);
    for (Operation *user : cur.getUsers()) {
      auto reshape = dyn_cast<stablehlo::ReshapeOp>(user);
      if (!reshape)
        continue;
      Value res = reshape.getResult();
      if (seen.insert(res).second)
        worklist.push_back(res);
    }
  }
}

/// A maximal set of elementwise ops connected through bitcast reshapes, plus
/// the boundary information needed to cost and rewrite it.
struct Chain {
  SmallVector<Operation *> members;
  DenseSet<Operation *> memberSet;

  /// Values feeding the chain from outside, already stripped of reshapes.
  SetVector<Value> inputs;
  /// (member result, shape) pairs at which the chain is read from outside.
  SmallVector<std::pair<Value, Shape>> externalUses;
  /// Reshape ops currently materializing a view of a chain value, or feeding
  /// one in. Used to make sure a rewrite never increases the reshape count.
  DenseSet<Operation *> boundaryReshapes;

  int64_t numElements = 0;
};

static bool isConstantValue(Value v) {
  DenseElementsAttr attr;
  return matchPattern(v, m_Constant(&attr));
}

/// Number of reshape ops that would exist on the boundary if the whole chain
/// were expressed in `shape`. Constants are free: they are folded into a new
/// constant of the target shape.
static int64_t boundaryCost(const Chain &chain, ArrayRef<int64_t> shape) {
  int64_t cost = 0;
  for (Value input : chain.inputs) {
    if (isConstantValue(input))
      continue;
    auto ty = cast<RankedTensorType>(input.getType());
    if (ty.getShape() != shape)
      ++cost;
  }
  // Distinct (value, shape) pairs each need one reshape; uses sharing both
  // share the reshape.
  DenseSet<std::pair<Value, ArrayRef<int64_t>>> counted;
  for (auto &[value, useShape] : chain.externalUses) {
    if (ArrayRef<int64_t>(useShape) == shape)
      continue;
    if (counted.insert({value, ArrayRef<int64_t>(useShape)}).second)
      ++cost;
  }
  return cost;
}

/// Build the chain containing `seed`, or return failure if `seed` is already
/// part of a previously built chain.
static void buildChains(Operation *root,
                        SmallVectorImpl<std::unique_ptr<Chain>> &chains) {
  llvm::EquivalenceClasses<Operation *> classes;

  root->walk([&](Operation *op) {
    if (!isChainCandidate(op))
      return;
    auto resultTy = getStaticRankedType(op->getResult(0));
    classes.insert(op);

    for (Value operand : op->getOperands()) {
      auto operandTy = getStaticRankedType(operand);
      // Operands of a different size (e.g. an implicitly broadcast scalar) do
      // not denote the same buffer, so they are a chain input, not an edge.
      if (!operandTy || operandTy.getNumElements() != resultTy.getNumElements())
        continue;
      Operation *def = stripReshapes(operand).getDefiningOp();
      if (!isChainCandidate(def))
        continue;
      if (getStaticRankedType(def->getResult(0)).getNumElements() !=
          resultTy.getNumElements())
        continue;
      classes.unionSets(op, def);
    }
  });

  for (auto it = classes.begin(), end = classes.end(); it != end; ++it) {
    if (!(*it)->isLeader())
      continue;

    auto chain = std::make_unique<Chain>();
    for (auto mi = classes.member_begin(**it), me = classes.member_end();
         mi != me; ++mi)
      chain->members.push_back(*mi);
    if (chain->members.size() < 2)
      continue;

    // Walk order gives a deterministic, dominance-respecting order.
    llvm::sort(chain->members, [](Operation *a, Operation *b) {
      return a->isBeforeInBlock(b);
    });
    chain->memberSet.insert(chain->members.begin(), chain->members.end());
    chain->numElements =
        getStaticRankedType(chain->members.front()->getResult(0))
            .getNumElements();

    // All members must live in the same block for the rewrite to be valid.
    Block *block = chain->members.front()->getBlock();
    if (llvm::any_of(chain->members,
                     [&](Operation *op) { return op->getBlock() != block; }))
      continue;

    for (Operation *member : chain->members) {
      for (Value operand : member->getOperands()) {
        Value src = stripReshapes(operand);
        Operation *def = src.getDefiningOp();
        if (def && chain->memberSet.contains(def))
          continue;
        chain->inputs.insert(src);
        if (operand != src)
          if (auto *reshape = operand.getDefiningOp())
            chain->boundaryReshapes.insert(reshape);
      }

      SmallVector<Value> views;
      collectReshapeViews(member->getResult(0), views);
      for (Value view : views) {
        auto viewTy = getStaticRankedType(view);
        if (!viewTy)
          continue;
        for (OpOperand &use : view.getUses()) {
          Operation *user = use.getOwner();
          if (chain->memberSet.contains(user))
            continue;
          // A reshape is transparent: it is accounted for via its own users.
          if (isa<stablehlo::ReshapeOp>(user))
            continue;
          chain->externalUses.push_back(
              {member->getResult(0), Shape(viewTy.getShape())});
        }
        if (view != member->getResult(0))
          chain->boundaryReshapes.insert(view.getDefiningOp());
      }
    }

    chains.push_back(std::move(chain));
  }
}

/// Candidate shapes: every shape the chain is currently written in, read at,
/// or fed from.
static SmallVector<Shape> candidateShapes(const Chain &chain) {
  SmallVector<Shape> candidates;
  auto add = [&](ArrayRef<int64_t> shape) {
    for (const Shape &existing : candidates)
      if (ArrayRef<int64_t>(existing) == shape)
        return;
    candidates.push_back(Shape(shape));
  };

  for (Operation *member : chain.members)
    add(getStaticRankedType(member->getResult(0)).getShape());
  for (Value input : chain.inputs) {
    auto ty = cast<RankedTensorType>(input.getType());
    if (ty.getNumElements() == chain.numElements)
      add(ty.getShape());
  }
  for (auto &[value, useShape] : chain.externalUses)
    add(useShape);

  return candidates;
}

static Value reshapeTo(OpBuilder &builder, Location loc, Value v,
                       ArrayRef<int64_t> shape) {
  auto ty = cast<RankedTensorType>(v.getType());
  if (ty.getShape() == shape)
    return v;
  return stablehlo::ReshapeOp::create(
      builder, loc, RankedTensorType::get(shape, ty.getElementType()), v);
}

static void rewriteChain(Chain &chain, ArrayRef<int64_t> shape) {
  OpBuilder builder(chain.members.front());

  // Materialize every chain input in the target shape.
  DenseMap<Value, Value> inputMap;
  for (Value input : chain.inputs) {
    auto ty = cast<RankedTensorType>(input.getType());
    if (ty.getNumElements() != chain.numElements) {
      // Not a same-buffer operand (implicitly broadcast scalar): leave as is.
      inputMap[input] = input;
      continue;
    }

    DenseElementsAttr attr;
    if (matchPattern(input, m_Constant(&attr))) {
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPoint(chain.members.front());
      inputMap[input] = stablehlo::ConstantOp::create(
          builder, input.getLoc(),
          attr.reshape(RankedTensorType::get(shape, ty.getElementType())));
      continue;
    }

    OpBuilder::InsertionGuard guard(builder);
    if (Operation *def = input.getDefiningOp())
      builder.setInsertionPointAfter(def);
    else
      builder.setInsertionPointToStart(chain.members.front()->getBlock());
    inputMap[input] = reshapeTo(builder, input.getLoc(), input, shape);
  }

  // Rewrite the members in dominance order.
  DenseMap<Value, Value> resultMap;
  for (Operation *member : chain.members) {
    SmallVector<Value> operands;
    for (Value operand : member->getOperands()) {
      Value src = stripReshapes(operand);
      if (Operation *def = src.getDefiningOp();
          def && chain.memberSet.contains(def)) {
        operands.push_back(resultMap.lookup(def->getResult(0)));
        continue;
      }
      operands.push_back(inputMap.lookup(src));
    }

    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(member);
    auto elemTy =
        cast<RankedTensorType>(member->getResult(0).getType()).getElementType();
    Operation *newOp = builder.create(
        member->getLoc(), member->getName().getIdentifier(),
        ValueRange(operands), TypeRange(RankedTensorType::get(shape, elemTy)),
        member->getAttrs(), {}, {});
    resultMap[member->getResult(0)] = newOp->getResult(0);
  }

  // Redirect every outside reader to the rewritten value, reshaping back only
  // where the reader genuinely wants a different shape.
  for (Operation *member : chain.members) {
    Value newValue = resultMap.lookup(member->getResult(0));

    SmallVector<Value> views;
    collectReshapeViews(member->getResult(0), views);
    for (Value view : views) {
      auto viewTy = cast<RankedTensorType>(view.getType());
      SmallVector<OpOperand *> externalUses;
      for (OpOperand &use : view.getUses()) {
        Operation *user = use.getOwner();
        if (chain.memberSet.contains(user) || isa<stablehlo::ReshapeOp>(user))
          continue;
        externalUses.push_back(&use);
      }
      if (externalUses.empty())
        continue;

      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointAfterValue(newValue);
      Value replacement =
          reshapeTo(builder, view.getLoc(), newValue, viewTy.getShape());
      for (OpOperand *use : externalUses)
        use->set(replacement);
    }
  }

  // Drop the old chain. A single op can show up as both a view and a boundary
  // reshape, so this must be a set: erasing the same op twice is a double free.
  SetVector<Operation *> toErase;
  for (Operation *member : chain.members) {
    SmallVector<Value> views;
    collectReshapeViews(member->getResult(0), views);
    for (Value view : views)
      if (view != member->getResult(0))
        toErase.insert(view.getDefiningOp());
  }
  toErase.insert(chain.members.begin(), chain.members.end());
  // The reshapes that used to adapt inputs into the old shape, and any
  // constant that was only there to feed the old chain, are dead too.
  toErase.insert(chain.boundaryReshapes.begin(), chain.boundaryReshapes.end());
  for (Value input : chain.inputs)
    if (Operation *def = input.getDefiningOp())
      if (isConstantValue(input))
        toErase.insert(def);

  // Order-independent: an op may still be used by another entry that is only
  // erased later in the list, so sweep to a fixpoint.
  SmallVector<Operation *> pending(toErase.begin(), toErase.end());
  bool changed = true;
  while (changed) {
    changed = false;
    for (Operation *&op : pending) {
      if (!op || !op->use_empty())
        continue;
      op->erase();
      op = nullptr;
      changed = true;
    }
  }
}

struct CanonicalizeElementwiseShapesPass
    : public enzyme::impl::CanonicalizeElementwiseShapesPassBase<
          CanonicalizeElementwiseShapesPass> {
  using CanonicalizeElementwiseShapesPassBase::
      CanonicalizeElementwiseShapesPassBase;

  void runOnOperation() override {
    SmallVector<std::unique_ptr<Chain>> chains;
    buildChains(getOperation(), chains);

    for (auto &chain : chains) {
      SmallVector<Shape> candidates = candidateShapes(*chain);
      if (candidates.size() < 2)
        continue;

      const Shape *best = nullptr;
      int64_t bestCost = std::numeric_limits<int64_t>::max();
      for (const Shape &candidate : candidates) {
        int64_t cost = boundaryCost(*chain, candidate);
        if (cost < bestCost) {
          bestCost = cost;
          best = &candidate;
        }
      }
      if (!best)
        continue;

      // Never trade one reshape for another: only rewrite when this strictly
      // reduces the number of reshapes on the chain boundary.
      if (bestCost >= (int64_t)chain->boundaryReshapes.size())
        continue;

      LLVM_DEBUG(llvm::dbgs()
                 << "canonicalizing chain of " << chain->members.size()
                 << " ops to " << bestCost << " boundary reshapes (was "
                 << chain->boundaryReshapes.size() << ")\n");
      rewriteChain(*chain, *best);
    }
  }
};

} // namespace
