// StencilGhostCellWidening.cpp — Ghost cell widening for stencil computations.
//
// Replaces per-operator halo exchanges with a single wide ghost cell exchange,
// following the "overlapped tiling with redundant computation" approach
// (Meng & Skadron ICS'09, Mittermair et al. 2025, Bisbas et al. ASPLOS'24).
//
// == Formal Framework ==
//
// Each value v on stencil dimension d has a VALIDITY FOOTPRINT:
//   F(v) = extra elements beyond the original domain size on dim d
//
// A ghost cell exchange of width K sets: F(rootField) = 2K
//
// Operations transform footprints:
//   slice [s:e] on dim d:  F(result) = F(input)      (extra preserved)
//   pad low/high=[1]:      F(result) = F(input) - 2   (ghost consumed)
//   elementwise f(a,b):    F(result) = min(F(a), F(b)) (narrowest operand)
//   reshape NxM→1xNxM:     F tracks through dim change
//
// INVARIANT: F(v) >= 0 for all values in the widened computation.
// If F(v) = 0, the widened result has the same size as the original.
// If F(v) > 0, a narrowing slice extracts the valid interior.
//
// The required K = critical path length through the shift DAG,
// where each shift (slice→pad) consumes 2 from the footprint.
//
// == Correctness ==
//
// The widened computation is a faithful reproduction of the original on a
// wider tensor. Ghost cells provide boundary data that the original would
// obtain via per-operator collective-permutes. The interior F(v)/2 positions
// from each boundary contain identical values to the original SPMD computation,
// because the redundant computation in the ghost region propagates correct
// values inward (overlapped tiling property).
//
// References:
//   [1] Mittermair et al., "Automatic Distributed Parallelization and
//       Communication Optimization for Stencils", HAL 2025
//   [2] Meng & Skadron, "Performance Modeling and Automatic Ghost Zone
//       Optimization for Iterative Stencil Loops", ICS 2009
//   [3] Bisbas et al., "A Shared Compilation Stack for Distributed-Memory
//       Parallelism in Stencil DSLs", ASPLOS 2024

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "src/enzyme_ad/jax/Passes/Passes.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_STENCILGHOSTCELLWIDENINGPASS
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

using namespace mlir;

namespace mlir::enzyme {
namespace {

// ============================================================
// Helpers
// ============================================================

/// Returns true if the op can be safely widened (elementwise-like).
static bool isWidenSafe(Operation *op) {
  if (op->hasTrait<OpTrait::Elementwise>())
    return true;
  return isa<stablehlo::SelectOp, stablehlo::CompareOp, stablehlo::ConvertOp,
             stablehlo::ClampOp>(op);
}

// ============================================================
// Step 1: Stencil Shift Detection
// ============================================================

/// A stencil shift: slice(removes 1 element on dim d) → [elementwise] →
/// pad(adds 1 on dim d). In SPMD, each such pad becomes a collective-permute
/// (width-1 halo exchange).
struct StencilShift {
  int32_t dim;
  stablehlo::SliceOp slice;
  stablehlo::PadOp pad;
  Value rootField; // slice's operand (the field being shifted)
  bool isLow;      // true if pad low=[1], false if pad high=[1]
};

/// Find all stencil shifts in the IR.
static SmallVector<StencilShift> findAllShifts(Operation *root) {
  SmallVector<StencilShift> shifts;
  root->walk([&](stablehlo::PadOp pad) {
    auto lo = pad.getEdgePaddingLow();
    auto hi = pad.getEdgePaddingHigh();
    auto in = pad.getInteriorPadding();
    int64_t rank = lo.size();

    // Must pad exactly 1 on exactly 1 dim, no interior padding.
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

    // Trace backward through elementwise ops to find the slice.
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

    // Validate: slice removes exactly 1 element on dim d, identity elsewhere.
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

// ============================================================
// Step 2: Shift DAG Construction
// ============================================================

/// Build successor/predecessor edges between shifts.
/// Shift A → Shift B if A.pad's result feeds B.slice through elementwise ops.
static void buildShiftDAG(
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

// ============================================================
// Step 3: Stencil Group Discovery
// ============================================================

/// A group of connected shifts sharing a root field on the same dimension.
struct StencilGroup {
  int32_t dim;
  Value rootField;
  SmallVector<SmallVector<StencilShift *>> levels; // BFS levels
  SmallVector<StencilShift *> allShifts;
  int64_t bfsDepth = 0;
  int64_t criticalPath = 0; // = required ghost width K

  // IR bounds
  Operation *firstOp = nullptr;
  Operation *lastOp = nullptr;
};

/// Discover stencil groups via BFS from root shifts (no predecessors).
/// Compute critical path K = longest path through the shift DAG.
static SmallVector<StencilGroup> discoverGroups(
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

    // BFS: start with root shift + complementary shifts at the same slice
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

    // Critical path: longest path through the shift DAG (= required K).
    // Each shift on the path consumes 2 from the validity footprint.
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

      // Find IR bounds
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

// ============================================================
// Step 4: Transform — Widen Computation with Ghost Cells
// ============================================================

/// Widen a stencil group's computation region.
///
/// For a group with critical path K:
///   1. Insert pad(rootField, K, K) → ghost cell exchange (1
///   collective-permute)
///   2. Clone all ops in [firstOp, lastOp] with widened types
///   3. Chain pads → replaced by slices (consume ghost cells, F -= 2)
///   4. Non-chain pads → widened normally
///   5. At external use boundaries → insert narrowing slices (extract interior)
///   6. Erase originals
///
/// INVARIANT: For every widened value v, F(v) >= 0.
static void widenGroup(StencilGroup &g, OpBuilder &b) {
  int32_t dim = g.dim;
  int64_t K = g.criticalPath;
  Value rootField = g.rootField;
  auto rootTy = cast<RankedTensorType>(rootField.getType());
  int64_t rank = rootTy.getRank();

  // Collect chain pad ops
  DenseSet<Operation *> chainPadOps;
  for (auto *sh : g.allShifts)
    chainPadOps.insert(sh->pad.getOperation());

  // Collect region ops
  SmallVector<Operation *> regionOps;
  DenseSet<Operation *> regionSet;
  for (auto it = g.firstOp->getIterator();; ++it) {
    regionOps.push_back(&*it);
    regionSet.insert(&*it);
    if (&*it == g.lastOp)
      break;
  }

  // --- Ghost cell exchange: pad rootField by K on each side ---
  Location loc = g.firstOp->getLoc();
  b.setInsertionPoint(g.firstOp);
  auto zero = b.create<stablehlo::ConstantOp>(
      loc, b.getZeroAttr(RankedTensorType::get({}, rootTy.getElementType())));
  SmallVector<int64_t> pL(rank, 0), pH(rank, 0), pI(rank, 0);
  pL[dim] = K;
  pH[dim] = K;
  auto ws = llvm::to_vector(rootTy.getShape());
  ws[dim] += 2 * K;
  auto wRoot = b.create<stablehlo::PadOp>(loc, rootTy.clone(ws), rootField,
                                          zero, pL, pH, pI);

  // --- Validity footprint tracking ---
  // vm: original value → widened value
  // F:  original value → validity footprint (extra elements on stencil dim)
  // dm: original value → current stencil dim (tracks through reshapes)
  IRMapping vm;
  DenseMap<Value, int64_t> F;
  DenseMap<Value, int32_t> dm;
  DenseSet<Operation *> created;

  vm.map(rootField, wRoot.getResult());
  F[rootField] = 2 * K; // ghost exchange provides 2K extra
  dm[rootField] = dim;
  created.insert(zero);
  created.insert(wRoot);

  // --- Walk region, clone each op with widened types ---
  for (Operation *op : regionOps) {
    b.setInsertionPointAfter(op);

    // Check if any operand has a widened version
    bool needsWidening = false;
    int64_t minF = INT64_MAX;
    int32_t curDim = dim;
    for (Value v : op->getOperands()) {
      if (vm.contains(v)) {
        needsWidening = true;
        minF = std::min(minF, F.lookup(v));
        if (dm.count(v))
          curDim = dm.lookup(v);
      }
    }
    if (!needsWidening)
      continue;

    // Target type: from the narrowest widened operand
    RankedTensorType tgt = nullptr;
    for (Value v : op->getOperands())
      if (vm.contains(v) && F.lookup(v) == minF) {
        tgt = cast<RankedTensorType>(vm.lookup(v).getType());
        break;
      }
    if (!tgt)
      continue;

    // --- Trim: reduce a value's footprint to a target ---
    auto trim = [&](Value v, int64_t curF, int64_t tgtF) -> Value {
      if (curF <= tgtF)
        return v;
      int64_t t = curF - tgtF, tlo = t / 2, thi = t - tlo;
      auto vt = cast<RankedTensorType>(v.getType());
      SmallVector<int64_t> st(vt.getRank(), 0), li(vt.getRank()),
          str(vt.getRank(), 1);
      for (int64_t i = 0; i < vt.getRank(); ++i) {
        li[i] = vt.getDimSize(i);
        if (i == curDim) {
          st[i] = tlo;
          li[i] -= thi;
        }
      }
      auto s = llvm::to_vector(vt.getShape());
      s[curDim] -= t;
      auto sl = b.create<stablehlo::SliceOp>(loc, vt.clone(s), v, st, li, str);
      created.insert(sl);
      return sl.getResult();
    };

    // --- Build widened operands ---
    SmallVector<Value> wops;
    for (Value v : op->getOperands()) {
      if (vm.contains(v)) {
        // Widened operand: trim to minF
        wops.push_back(trim(vm.lookup(v), F.lookup(v), minF));
      } else {
        // Non-chain operand: pad up or trim down to match target
        auto vt = dyn_cast<RankedTensorType>(v.getType());
        if (vt && vt.getRank() == tgt.getRank()) {
          int64_t vD = vt.getDimSize(curDim), tD = tgt.getDimSize(curDim);
          if (vD < tD) {
            // Pad non-chain operand (typically a constant — local, no comm)
            int64_t n = tD - vD;
            SmallVector<int64_t> wl(vt.getRank(), 0), wh(vt.getRank(), 0),
                wi(vt.getRank(), 0);
            wl[curDim] = n / 2;
            wh[curDim] = n - n / 2;
            auto s = llvm::to_vector(vt.getShape());
            s[curDim] = tD;
            auto zz = b.create<stablehlo::ConstantOp>(
                loc,
                b.getZeroAttr(RankedTensorType::get({}, vt.getElementType())));
            auto p =
                b.create<stablehlo::PadOp>(loc, vt.clone(s), v, zz, wl, wh, wi);
            created.insert(zz);
            created.insert(p);
            wops.push_back(p);
          } else if (vD > tD) {
            wops.push_back(trim(v, vD - tD, 0));
          } else {
            wops.push_back(v);
          }
        } else {
          wops.push_back(v);
        }
      }
    }

    // --- Create widened op ---
    Value wr;
    int64_t newF = minF;

    if (auto sl = dyn_cast<stablehlo::SliceOp>(op)) {
      // Slice: extend limit on stencil dim. F preserved.
      auto st = llvm::to_vector(sl.getStartIndices());
      auto li = llvm::to_vector(sl.getLimitIndices());
      auto str = llvm::to_vector(sl.getStrides());
      li[curDim] += minF;
      auto s = llvm::to_vector(cast<RankedTensorType>(sl.getType()).getShape());
      s[curDim] += minF;
      wr = b.create<stablehlo::SliceOp>(
          loc, cast<RankedTensorType>(sl.getType()).clone(s), wops[0], st, li,
          str);
      // F(slice) = F(input) — footprint preserved through slice
      newF = minF;

    } else if (auto pd = dyn_cast<stablehlo::PadOp>(op)) {
      if (chainPadOps.contains(op)) {
        // Chain pad → replace with slice that consumes one ghost cell.
        // pad low=[1] (prepend zero) → drop last element
        // pad high=[1] (append zero) → drop first element
        // F(result) = F(input) - 2  (ghost consumed from both sides)
        auto inTy = cast<RankedTensorType>(wops[0].getType());
        int64_t S = inTy.getDimSize(curDim);
        SmallVector<int64_t> st(rank, 0), li(rank), str(rank, 1);
        for (int64_t i = 0; i < rank; ++i)
          li[i] = inTy.getDimSize(i);
        if (pd.getEdgePaddingHigh()[curDim] >= 1)
          st[curDim] = 1; // drop first
        else
          li[curDim] = S - 1; // drop last
        auto s = llvm::to_vector(inTy.getShape());
        s[curDim] = S - 1;
        wr = b.create<stablehlo::SliceOp>(
            loc, cast<RankedTensorType>(pd.getType()).clone(s), wops[0], st, li,
            str);
        newF = minF - 2;
        assert(newF >= 0 && "Ghost cells exhausted — K too small!");
      } else {
        // Non-chain pad: widen normally. F preserved.
        auto inTy = cast<RankedTensorType>(wops[0].getType());
        int64_t nd = inTy.getDimSize(curDim) + pd.getEdgePaddingLow()[curDim] +
                     pd.getEdgePaddingHigh()[curDim];
        auto s =
            llvm::to_vector(cast<RankedTensorType>(pd.getType()).getShape());
        s[curDim] = nd;
        wr = b.create<stablehlo::PadOp>(
            loc, cast<RankedTensorType>(pd.getType()).clone(s), wops[0],
            wops[1], pd.getEdgePaddingLow(), pd.getEdgePaddingHigh(),
            pd.getInteriorPadding());
      }

    } else if (isWidenSafe(op)) {
      // Elementwise: F(result) = min(F(operands)) = minF
      OperationState st(loc, op->getName());
      st.addOperands(wops);
      st.addTypes(tgt);
      st.addAttributes(op->getAttrs());
      wr = b.create(st)->getResult(0);

    } else if (op->getNumOperands() >= 1 && op->getNumResults() == 1 &&
               op->getResult(0).getType() == op->getOperand(0).getType()) {
      // Shape-preserving pass-through (e.g., enzymexla.rotate on non-stencil
      // dim)
      OperationState st(loc, op->getName());
      st.addOperands(wops);
      st.addTypes(cast<RankedTensorType>(wops[0].getType()));
      st.addAttributes(op->getAttrs());
      wr = b.create(st)->getResult(0);

    } else if (auto rs = dyn_cast<stablehlo::ReshapeOp>(op)) {
      // Reshape: track stencil dim through shape change
      auto ot = cast<RankedTensorType>(rs.getType());
      auto inTy = cast<RankedTensorType>(wops[0].getType());
      auto s = llvm::to_vector(ot.getShape());
      int32_t newDim = curDim;
      int64_t widenedSize = inTy.getDimSize(curDim);
      for (int64_t i = 0; i < (int64_t)s.size(); ++i)
        if (s[i] == widenedSize - minF) {
          s[i] = widenedSize;
          newDim = i;
          break;
        }
      wr = b.create<stablehlo::ReshapeOp>(loc, ot.clone(s), wops[0]);
      dm[op->getResult(0)] = newDim;
    }

    if (!wr)
      continue;

    F[op->getResult(0)] = newF;
    if (!dm.count(op->getResult(0)))
      dm[op->getResult(0)] = curDim;
    vm.map(op->getResult(0), wr);
    if (wr.getDefiningOp())
      created.insert(wr.getDefiningOp());
  }

  // --- Replace external uses with narrowed widened values ---
  int replaced = 0;
  for (Operation *op : regionOps) {
    Value orig = op->getResult(0);
    if (!vm.contains(orig))
      continue;
    Value wide = vm.lookup(orig);
    int64_t remF = F.lookup(orig);
    int32_t d = dm.count(orig) ? dm.lookup(orig) : dim;

    // Find external uses (outside region and not our created ops)
    SmallVector<OpOperand *> extUses;
    for (auto &use : orig.getUses())
      if (!regionSet.contains(use.getOwner()) &&
          !created.contains(use.getOwner()))
        extUses.push_back(&use);
    if (extUses.empty())
      continue;

    // Narrow: extract interior [F/2 : origSize + F/2]
    Value narrow;
    if (remF > 0) {
      b.setInsertionPointAfterValue(wide);
      int64_t nk = remF / 2;
      auto ot = cast<RankedTensorType>(orig.getType());
      SmallVector<int64_t> ns(ot.getRank(), 0), nl(ot.getRank()),
          nstr(ot.getRank(), 1);
      for (int64_t i = 0; i < ot.getRank(); ++i) {
        ns[i] = (i == d) ? nk : 0;
        nl[i] = (i == d) ? ot.getDimSize(i) + nk : ot.getDimSize(i);
      }
      auto sl = b.create<stablehlo::SliceOp>(loc, ot, wide, ns, nl, nstr);
      created.insert(sl);
      narrow = sl.getResult();
    } else if (orig.getType() == wide.getType()) {
      narrow = wide;
    } else {
      continue;
    }

    for (auto *use : extUses) {
      use->set(narrow);
      replaced++;
    }
  }

  // --- Erase dead originals ---
  for (auto it = regionOps.rbegin(); it != regionOps.rend(); ++it)
    if ((*it)->use_empty())
      (*it)->erase();

  llvm::errs() << "  widened " << regionOps.size() << " ops, " << replaced
               << " external uses replaced, K=" << K << "\n";
}

// ============================================================
// Pass Entry Point
// ============================================================

class StencilGhostCellWideningPass
    : public impl::StencilGhostCellWideningPassBase<
          StencilGhostCellWideningPass> {
public:
  using StencilGhostCellWideningPassBase::StencilGhostCellWideningPassBase;

  void runOnOperation() override {
    auto *root = getOperation();

    // Step 1: Find all stencil shifts
    auto shifts = findAllShifts(root);
    if (shifts.empty())
      return;

    // Step 2: Build shift DAG
    DenseMap<StencilShift *, SmallVector<StencilShift *>> successors,
        predecessors;
    DenseMap<Operation *, SmallVector<StencilShift *>> shiftsBySlice;
    buildShiftDAG(shifts, maxSearchDepth, successors, predecessors,
                  shiftsBySlice);

    // Step 3: Discover stencil groups
    auto groups =
        discoverGroups(shifts, successors, predecessors, shiftsBySlice);
    if (groups.empty())
      return;

    // Diagnostics
    for (size_t i = 0; i < groups.size(); ++i) {
      auto &g = groups[i];
      llvm::errs() << "StencilGhostCell group " << i << ": dim=" << g.dim
                   << " K=" << g.criticalPath
                   << " shifts=" << g.allShifts.size()
                   << " BFS_depth=" << g.bfsDepth
                   << " root=" << cast<RankedTensorType>(g.rootField.getType())
                   << "\n";
    }

    // Step 4: Transform each group
    OpBuilder b(root->getContext());
    for (size_t i = 0; i < groups.size(); ++i)
      widenGroup(groups[i], b);
  }
};

} // namespace
} // namespace mlir::enzyme
