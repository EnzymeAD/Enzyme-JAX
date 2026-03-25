// StencilAnalysis.h — Stencil shift detection and chain analysis.
//
// Shared between StencilGhostCellWidening pass and OptimizeCommunication.

#ifndef ENZYMEXLA_STENCIL_ANALYSIS_H
#define ENZYMEXLA_STENCIL_ANALYSIS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir::enzyme::stencil {

/// Returns true if the op can be safely widened (elementwise-like).
inline bool isWidenSafe(Operation *op) {
  if (op->hasTrait<OpTrait::Elementwise>())
    return true;
  return isa<stablehlo::SelectOp, stablehlo::CompareOp, stablehlo::ConvertOp,
             stablehlo::ClampOp>(op);
}

/// A stencil shift: slice(removes 1 element on dim d) → [elementwise] →
/// pad(adds 1 on dim d).
struct StencilShift {
  int32_t dim;
  stablehlo::SliceOp slice;
  stablehlo::PadOp pad;
  Value rootField;
  bool isLow;
};

/// A group of connected shifts sharing a root field on the same dimension.
struct StencilGroup {
  int32_t dim;
  Value rootField;
  SmallVector<SmallVector<StencilShift *>> levels;
  SmallVector<StencilShift *> allShifts;
  int64_t bfsDepth = 0;
  int64_t criticalPath = 0; // = required ghost width K
  Operation *firstOp = nullptr;
  Operation *lastOp = nullptr;
};

/// Find all stencil shifts in the IR.
inline SmallVector<StencilShift> findAllShifts(Operation *root) {
  SmallVector<StencilShift> shifts;
  root->walk([&](stablehlo::PadOp pad) {
    auto lo = pad.getEdgePaddingLow();
    auto hi = pad.getEdgePaddingHigh();
    auto in = pad.getInteriorPadding();
    int64_t rank = lo.size();

    int32_t d = -1;
    for (int64_t i = 0; i < rank; ++i) {
      if (in[i])
        return;
      if (lo[i] || hi[i]) {
        if (d != -1)
          return;
        d = i;
      }
    }
    if (d < 0 || lo[d] + hi[d] != 1)
      return;

    std::function<stablehlo::SliceOp(Value, int)> findSlice =
        [&](Value v, int depth) -> stablehlo::SliceOp {
      if (depth <= 0)
        return nullptr;
      if (auto s = v.getDefiningOp<stablehlo::SliceOp>())
        return s;
      auto *def = v.getDefiningOp();
      if (!def || !isWidenSafe(def))
        return nullptr;
      for (Value op : def->getOperands())
        if (auto s = findSlice(op, depth - 1))
          return s;
      return nullptr;
    };
    auto sl = findSlice(pad.getOperand(), 5);
    if (!sl)
      return;

    auto inTy = cast<RankedTensorType>(sl.getOperand().getType());
    if (inTy.getRank() != rank)
      return;
    for (int64_t i = 0; i < rank; ++i) {
      if (sl.getStrides()[i] != 1)
        return;
      if (i == d) {
        if (sl.getLimitIndices()[i] - sl.getStartIndices()[i] !=
            inTy.getDimSize(i) - 1)
          return;
      } else {
        if (sl.getStartIndices()[i] ||
            sl.getLimitIndices()[i] != inTy.getDimSize(i))
          return;
      }
    }
    shifts.push_back({d, sl, pad, sl.getOperand(), lo[d] == 1});
  });
  return shifts;
}

/// Build successor/predecessor edges between shifts.
inline void buildShiftDAG(
    SmallVectorImpl<StencilShift> &shifts, int64_t maxDepth,
    DenseMap<StencilShift *, SmallVector<StencilShift *>> &successors,
    DenseMap<StencilShift *, SmallVector<StencilShift *>> &predecessors,
    DenseMap<Operation *, SmallVector<StencilShift *>> &shiftsBySlice) {

  for (auto &s : shifts)
    shiftsBySlice[s.slice.getOperation()].push_back(&s);

  for (auto &s : shifts) {
    SmallVector<Operation *> wl;
    DenseSet<Operation *> seen;
    for (auto *u : s.pad.getResult().getUsers()) {
      wl.push_back(u);
      seen.insert(u);
    }
    for (int64_t i = 0; i < maxDepth && !wl.empty(); ++i) {
      auto *op = wl.pop_back_val();
      if (auto ns = dyn_cast<stablehlo::SliceOp>(op)) {
        auto it = shiftsBySlice.find(ns.getOperation());
        if (it != shiftsBySlice.end())
          for (auto *s2 : it->second)
            if (s2->dim == s.dim) {
              successors[&s].push_back(s2);
              predecessors[s2].push_back(&s);
            }
      }
      if (isWidenSafe(op) || isa<stablehlo::ReshapeOp>(op))
        for (auto *u : op->getResult(0).getUsers())
          if (seen.insert(u).second)
            wl.push_back(u);
    }
  }
}

/// Discover stencil groups via BFS and compute critical path K.
inline SmallVector<StencilGroup> discoverGroups(
    SmallVectorImpl<StencilShift> &shifts,
    DenseMap<StencilShift *, SmallVector<StencilShift *>> &successors,
    DenseMap<StencilShift *, SmallVector<StencilShift *>> &predecessors,
    DenseMap<Operation *, SmallVector<StencilShift *>> &shiftsBySlice) {

  DenseSet<StencilShift *> visited;
  SmallVector<StencilGroup> groups;

  for (auto &s : shifts) {
    if (visited.contains(&s))
      continue;
    if (predecessors.count(&s) && !predecessors[&s].empty())
      continue;

    StencilGroup g;
    g.dim = s.dim;
    g.rootField = s.rootField;

    SmallVector<StencilShift *> cur = {&s};
    for (auto *s2 : shiftsBySlice[s.slice.getOperation()])
      if (s2 != &s && s2->dim == s.dim && !visited.contains(s2))
        cur.push_back(s2);
    for (auto *cs : cur)
      visited.insert(cs);

    while (!cur.empty()) {
      g.levels.push_back(cur);
      g.bfsDepth++;
      DenseSet<StencilShift *> next;
      for (auto *cs : cur)
        if (successors.count(cs))
          for (auto *ns : successors[cs])
            if (!visited.contains(ns)) {
              next.insert(ns);
              visited.insert(ns);
              for (auto *s2 : shiftsBySlice[ns->slice.getOperation()])
                if (!visited.contains(s2) && s2->dim == ns->dim) {
                  next.insert(s2);
                  visited.insert(s2);
                }
            }
      cur.assign(next.begin(), next.end());
    }

    for (auto &level : g.levels)
      for (auto *sh : level)
        g.allShifts.push_back(sh);

    if (g.bfsDepth >= 2) {
      DenseMap<StencilShift *, int64_t> memo;
      std::function<int64_t(StencilShift *)> longest =
          [&](StencilShift *sh) -> int64_t {
        auto it = memo.find(sh);
        if (it != memo.end())
          return it->second;
        int64_t best = 1;
        if (successors.count(sh))
          for (auto *ns : successors[sh])
            best = std::max(best, 1 + longest(ns));
        return memo[sh] = best;
      };
      for (auto *sh : g.allShifts)
        g.criticalPath = std::max(g.criticalPath, longest(sh));

      for (auto *sh : g.allShifts) {
        if (!g.firstOp || sh->slice->isBeforeInBlock(g.firstOp))
          g.firstOp = sh->slice;
        if (!g.lastOp || g.lastOp->isBeforeInBlock(sh->pad))
          g.lastOp = sh->pad;
      }
      groups.push_back(std::move(g));
    }
  }
  return groups;
}

} // namespace mlir::enzyme::stencil

#endif // ENZYMEXLA_STENCIL_ANALYSIS_H
