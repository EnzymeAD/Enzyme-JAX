//===- FixedPointLoops.cpp - Loops marked as fixed points <-> the op ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// recognize-fixed-point-loops: marks convergence loops enzyme.fixed_point.
//
// raise-fixed-point-loops: a stablehlo.while carrying the marker becomes an
// enzymexla.fixed_point, which the Enzyme pass differentiates at its converged
// state and the StableHLO optimizations leave alone.
//
// lower-fixed-point-loops: the op becomes a marked stablehlo.while again.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "stablehlo/dialect/StablehloOps.h"

#include "src/enzyme_ad/jax/Dialect/Dialect.h"
#include "src/enzyme_ad/jax/Dialect/Ops.h"
#include "src/enzyme_ad/jax/Passes/Passes.h"

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_RAISEFIXEDPOINTLOOPSPASS
#define GEN_PASS_DEF_LOWERFIXEDPOINTLOOPSPASS
#define GEN_PASS_DEF_RECOGNIZEFIXEDPOINTLOOPSPASS
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

using namespace mlir;
using namespace mlir::enzyme;
using enzymexla::FixedPointOp;

//===----------------------------------------------------------------------===//
// Raising and lowering
//===----------------------------------------------------------------------===//

namespace {

// The loop's other attributes travel with it; the marker attributes are what
// the op expresses with its own.
static void copyAttrsExceptMarker(Operation *from, Operation *to) {
  for (NamedAttribute attr : from->getDiscardableAttrs()) {
    StringRef name = attr.getName().strref();
    if (name != FixedPointOp::kMarkerAttrName &&
        name != FixedPointOp::kMarkerReductionAttrName &&
        name != FixedPointOp::kMarkerMaxItersAttrName &&
        name != FixedPointOp::kMarkerMaskAttrName)
      to->setAttr(attr.getName(), attr.getValue());
  }
}

// stablehlo.while and enzymexla.fixed_point have the same shape: the (cond,
// body) regions move over, only their terminators change.
template <typename TerminatorTy>
static void moveRegions(RewriterBase &rewriter, Operation *from,
                        Operation *to) {
  for (auto [src, dst] :
       llvm::zip_equal(from->getRegions(), to->getRegions())) {
    dst.takeBody(src);
    Operation *term = dst.front().getTerminator();
    rewriter.setInsertionPoint(term);
    TerminatorTy::create(rewriter, term->getLoc(),
                         ValueRange(term->getOperands()));
    rewriter.eraseOp(term);
  }
}

static void raiseFixedPoint(IRRewriter &rewriter, stablehlo::WhileOp w) {
  FloatAttr reduction;
  IntegerAttr maxIters, mask;
  if (auto a =
          w->getAttrOfType<FloatAttr>(FixedPointOp::kMarkerReductionAttrName))
    reduction = rewriter.getF64FloatAttr(a.getValueAsDouble());
  if (auto a =
          w->getAttrOfType<IntegerAttr>(FixedPointOp::kMarkerMaxItersAttrName))
    maxIters = rewriter.getI64IntegerAttr(a.getInt());
  if (auto a = w->getAttrOfType<IntegerAttr>(FixedPointOp::kMarkerMaskAttrName))
    mask = rewriter.getI64IntegerAttr(a.getInt());
  rewriter.setInsertionPoint(w);
  auto fp = FixedPointOp::create(rewriter, w.getLoc(), w.getResultTypes(),
                                 w.getOperands(), reduction, maxIters, mask);
  copyAttrsExceptMarker(w, fp);
  moveRegions<enzymexla::YieldOp>(rewriter, w, fp);
  rewriter.replaceOp(w, fp.getResults());
}

static void lowerFixedPoint(IRRewriter &rewriter, FixedPointOp fp) {
  rewriter.setInsertionPoint(fp);
  auto w = stablehlo::WhileOp::create(rewriter, fp.getLoc(),
                                      fp.getResultTypes(), fp.getInits());
  copyAttrsExceptMarker(fp, w);
  w->setAttr(FixedPointOp::kMarkerAttrName, rewriter.getUnitAttr());
  if (auto a = fp.getReductionAttr())
    w->setAttr(FixedPointOp::kMarkerReductionAttrName, a);
  if (auto a = fp.getMaxItersAttr())
    w->setAttr(FixedPointOp::kMarkerMaxItersAttrName, a);
  if (auto a = fp.getMaskAttr())
    w->setAttr(FixedPointOp::kMarkerMaskAttrName, a);
  moveRegions<stablehlo::ReturnOp>(rewriter, fp, w);
  rewriter.replaceOp(fp, w->getResults());
}

struct RaiseFixedPointLoopsPass
    : public enzyme::impl::RaiseFixedPointLoopsPassBase<
          RaiseFixedPointLoopsPass> {
  void runOnOperation() override {
    // Inner loops first (post-order), so an outer loop's regions move with
    // their already-converted contents.
    SmallVector<stablehlo::WhileOp> loops;
    getOperation()->walk([&](stablehlo::WhileOp w) {
      if (FixedPointOp::isMarked(w))
        loops.push_back(w);
    });
    IRRewriter rewriter(&getContext());
    for (auto w : loops)
      raiseFixedPoint(rewriter, w);
  }
};

struct LowerFixedPointLoopsPass
    : public enzyme::impl::LowerFixedPointLoopsPassBase<
          LowerFixedPointLoopsPass> {
  void runOnOperation() override {
    SmallVector<FixedPointOp> ops;
    getOperation()->walk([&](FixedPointOp op) { ops.push_back(op); });
    IRRewriter rewriter(&getContext());
    for (auto op : ops)
      lowerFixedPoint(rewriter, op);
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Recognition
//===----------------------------------------------------------------------===//
//
// The convergence loops this is after have one signature whatever the dialect
// they are written in: the exit condition compares a float quantity against a
// loop-invariant tolerance, and that quantity is a *change* of the state (the
// step just taken, or the difference between successive iterates) or a
// residual of the equation being solved. The dangerous look-alikes -- a time
// loop stopping at t_end, a Krylov iteration -- are told apart by their
// updates rather than by their condition: they carry accumulators, values
// updated as `s + v` with `v` independent of `s`.
//
// Both loop shapes are seen through one view: the "cond" block computes the
// predicate from the carried values, the "body" block computes the next ones.
// For stablehlo.while and a plain scf.while these are the two regions. For a
// rotated scf.while (what a kernel's do-while lifts to) the before region does
// both and forwards the next values, and the after region is a pass-through;
// the view follows values across the two regions through the condition's
// forwarded arguments, so the rules below need not know which shape it is.

namespace {

struct LoopView {
  Operation *op = nullptr;
  Block *before = nullptr; // computes the predicate
  Block *after = nullptr;  // computes the next values
  Value pred;
  // What flows into before-arg i on the back edge: after's yield operand i,
  // or, when that is an after-arg, the value the condition forwards for it.
  SmallVector<Value> backedge;
  // The value in `after` that stands for before-arg i (the after-arg that
  // receives the forwarded before-arg), or null if the before region forwards
  // something else for it.
  SmallVector<Value> afterTwin;

  bool isInside(Operation *o) const { return op->isAncestor(o); }

  // A value computed inside the loop that is not a carried argument.
  Operation *insideDef(Value v) const {
    Operation *d = v.getDefiningOp();
    return d && isInside(d) ? d : nullptr;
  }

  bool isCarry(Value v) const {
    auto arg = dyn_cast<BlockArgument>(v);
    return arg && (arg.getOwner() == before || arg.getOwner() == after);
  }

  // Loop-invariant: defined outside the loop, or a constant.
  bool isInvariant(Value v) const {
    if (auto arg = dyn_cast<BlockArgument>(v))
      return !isInside(arg.getOwner()->getParentOp());
    Operation *d = v.getDefiningOp();
    if (!isInside(d))
      return true;
    return d->hasTrait<OpTrait::ConstantLike>();
  }
};

static std::optional<LoopView> viewOf(stablehlo::WhileOp w) {
  LoopView v;
  v.op = w;
  v.before = &w.getCond().front();
  v.after = &w.getBody().front();
  v.pred = v.before->getTerminator()->getOperand(0);
  Operation *yield = v.after->getTerminator();
  for (unsigned i = 0; i < w->getNumOperands(); ++i) {
    v.backedge.push_back(yield->getOperand(i));
    v.afterTwin.push_back(v.after->getArgument(i));
  }
  return v;
}

static std::optional<LoopView> viewOf(scf::WhileOp w) {
  LoopView v;
  v.op = w;
  v.before = w.getBeforeBody();
  v.after = w.getAfterBody();
  auto cond = cast<scf::ConditionOp>(v.before->getTerminator());
  v.pred = cond.getCondition();
  Operation *yield = v.after->getTerminator();
  if (cond.getArgs().size() != v.after->getNumArguments() ||
      yield->getNumOperands() != v.before->getNumArguments())
    return std::nullopt;
  for (unsigned i = 0; i < v.before->getNumArguments(); ++i) {
    Value y = yield->getOperand(i);
    if (auto a = dyn_cast<BlockArgument>(y); a && a.getOwner() == v.after)
      y = cond.getArgs()[a.getArgNumber()];
    v.backedge.push_back(y);
    Value twin = nullptr;
    for (auto [k, fwd] : llvm::enumerate(cond.getArgs()))
      if (fwd == v.before->getArgument(i))
        twin = v.after->getArgument(k);
    v.afterTwin.push_back(twin);
  }
  return v;
}

static bool isFloatLike(Type t) {
  return isa<FloatType>(getElementTypeOrSelf(t));
}

static bool isIntLike(Type t) {
  return isa<IntegerType>(getElementTypeOrSelf(t)) &&
         !getElementTypeOrSelf(t).isInteger(1);
}

static bool isAndOp(Operation *o) {
  return isa<LLVM::AndOp, arith::AndIOp, stablehlo::AndOp>(o);
}

static bool isSubOp(Operation *o) {
  return isa<LLVM::FSubOp, arith::SubFOp, stablehlo::SubtractOp>(o);
}

static bool isAddOp(Operation *o) {
  return isa<LLVM::FAddOp, arith::AddFOp, stablehlo::AddOp>(o);
}

// A float comparison in the predicate, oriented as (quantity, bound) where
// the loop goes on while quantity > bound (or >=). Returns false for anything
// else, including the opposite orientation, since a loop that goes on while
// something is *below* a bound is not converging on it.
static bool asGreaterThan(Operation *o, Value &quantity, Value &bound) {
  if (auto c = dyn_cast<LLVM::FCmpOp>(o)) {
    switch (c.getPredicate()) {
    case LLVM::FCmpPredicate::ogt:
    case LLVM::FCmpPredicate::oge:
    case LLVM::FCmpPredicate::ugt:
    case LLVM::FCmpPredicate::uge:
      quantity = c.getLhs();
      bound = c.getRhs();
      return true;
    case LLVM::FCmpPredicate::olt:
    case LLVM::FCmpPredicate::ole:
    case LLVM::FCmpPredicate::ult:
    case LLVM::FCmpPredicate::ule:
      quantity = c.getRhs();
      bound = c.getLhs();
      return true;
    default:
      return false;
    }
  }
  if (auto c = dyn_cast<arith::CmpFOp>(o)) {
    switch (c.getPredicate()) {
    case arith::CmpFPredicate::OGT:
    case arith::CmpFPredicate::OGE:
    case arith::CmpFPredicate::UGT:
    case arith::CmpFPredicate::UGE:
      quantity = c.getLhs();
      bound = c.getRhs();
      return true;
    case arith::CmpFPredicate::OLT:
    case arith::CmpFPredicate::OLE:
    case arith::CmpFPredicate::ULT:
    case arith::CmpFPredicate::ULE:
      quantity = c.getRhs();
      bound = c.getLhs();
      return true;
    default:
      return false;
    }
  }
  if (auto c = dyn_cast<stablehlo::CompareOp>(o)) {
    if (!isFloatLike(c.getLhs().getType()))
      return false;
    switch (c.getComparisonDirection()) {
    case stablehlo::ComparisonDirection::GT:
    case stablehlo::ComparisonDirection::GE:
      quantity = c.getLhs();
      bound = c.getRhs();
      return true;
    case stablehlo::ComparisonDirection::LT:
    case stablehlo::ComparisonDirection::LE:
      quantity = c.getRhs();
      bound = c.getLhs();
      return true;
    default:
      return false;
    }
  }
  return false;
}

// The values an op result is computed from. For an scf.if result that is the
// condition and the two yields of that result, not everything in the
// branches; for other region ops it is the operands plus whatever the regions
// read (reductions read only their block arguments).
static void producers(Operation *d, Value v, SmallVectorImpl<Value> &out) {
  if (auto ifOp = dyn_cast<scf::IfOp>(d)) {
    unsigned k = cast<OpResult>(v).getResultNumber();
    out.push_back(ifOp.getCondition());
    out.push_back(ifOp.thenYield().getOperand(k));
    out.push_back(ifOp.elseYield().getOperand(k));
    return;
  }
  out.append(d->getOperands().begin(), d->getOperands().end());
  for (Region &r : d->getRegions())
    for (Block &b : r)
      for (Operation &nested : b)
        out.append(nested.getOperands().begin(), nested.getOperands().end());
}

// Whether `v` depends on a carried value, walking the loop's own ops and
// following carried arguments across the back edge.
static bool dependsOnCarry(const LoopView &lv, Value v,
                           DenseSet<Value> &visited) {
  if (!visited.insert(v).second)
    return false;
  if (lv.isCarry(v))
    return true;
  if (Operation *d = lv.insideDef(v)) {
    SmallVector<Value> from;
    producers(d, v, from);
    for (Value operand : from)
      if (dependsOnCarry(lv, operand, visited))
        return true;
  }
  return false;
}

static bool dependsOnCarry(const LoopView &lv, Value v) {
  DenseSet<Value> visited;
  return dependsOnCarry(lv, v, visited);
}

// Whether `v` depends on the specific carried argument `arg` (and its twin in
// the other region), without crossing the back edge.
static bool dependsOnArg(const LoopView &lv, Value v, Value arg, Value twin,
                         DenseSet<Value> &visited) {
  if (!visited.insert(v).second)
    return false;
  if (v == arg || (twin && v == twin))
    return true;
  if (Operation *d = lv.insideDef(v)) {
    SmallVector<Value> from;
    producers(d, v, from);
    for (Value operand : from)
      if (dependsOnArg(lv, operand, arg, twin, visited))
        return true;
  }
  return false;
}

// Whether `v` is a change or residual of the state: it is derived from a
// subtraction one of whose operands depends on carried values. Any op is
// walked through (abs, square, sqrt, max, reductions, conversions, scaling,
// selects between branches), and a carried value is followed to what the
// loop feeds into it, once.
static bool isChangeOfState(const LoopView &lv, Value v,
                            DenseSet<Value> &visited) {
  if (!visited.insert(v).second)
    return false;
  if (auto arg = dyn_cast<BlockArgument>(v)) {
    if (arg.getOwner() == lv.before)
      return isChangeOfState(lv, lv.backedge[arg.getArgNumber()], visited);
    if (arg.getOwner() == lv.after) {
      // The after-arg stands for whatever the condition forwarded for it.
      for (auto [i, twin] : llvm::enumerate(lv.afterTwin))
        if (twin == v)
          return isChangeOfState(lv, lv.before->getArgument(i), visited);
      if (isa<scf::WhileOp>(lv.op)) {
        auto cond = cast<scf::ConditionOp>(lv.before->getTerminator());
        return isChangeOfState(lv, cond.getArgs()[arg.getArgNumber()], visited);
      }
    }
    return false;
  }
  Operation *d = lv.insideDef(v);
  if (!d)
    return false;
  if (isSubOp(d))
    for (Value operand : d->getOperands())
      if (dependsOnCarry(lv, operand))
        return true;
  SmallVector<Value> from;
  if (auto ifOp = dyn_cast<scf::IfOp>(d)) {
    // The result is one of the yields; the condition is not part of its value.
    unsigned k = cast<OpResult>(v).getResultNumber();
    from.push_back(ifOp.thenYield().getOperand(k));
    from.push_back(ifOp.elseYield().getOperand(k));
  } else {
    producers(d, v, from);
  }
  for (Value operand : from)
    if (isChangeOfState(lv, operand, visited))
      return true;
  return false;
}

static bool isConstantFalse(Value v) {
  llvm::APInt c;
  return matchPattern(v, m_ConstantInt(&c)) && c.isZero();
}

static bool isIntCast(Operation *o) {
  return isa<arith::ExtUIOp, arith::TruncIOp, arith::ExtSIOp, LLVM::ZExtOp,
             LLVM::TruncOp, LLVM::SExtOp>(o);
}

// Splits the predicate into conjuncts, looking through integer casts and
// through the shapes a short-circuit `a && b` lowers to: `select a, b, false`,
// or an scf.if on `a` whose other branch yields false (`!a && b` when the
// then-branch is the false one, as Julia's `while` with a cap lowers to).
static void conjuncts(const LoopView &lv, Value v, SmallVectorImpl<Value> &out,
                      unsigned depth = 0) {
  if (depth > 16) {
    out.push_back(v);
    return;
  }
  Operation *d = lv.insideDef(v);
  if (!d) {
    out.push_back(v);
    return;
  }
  if (isAndOp(d)) {
    for (Value operand : d->getOperands())
      conjuncts(lv, operand, out, depth + 1);
    return;
  }
  if (isIntCast(d)) {
    conjuncts(lv, d->getOperand(0), out, depth + 1);
    return;
  }
  if (isa<arith::SelectOp, LLVM::SelectOp, stablehlo::SelectOp>(d)) {
    Value c = d->getOperand(0), t = d->getOperand(1), f = d->getOperand(2);
    if (isConstantFalse(f)) {
      conjuncts(lv, c, out, depth + 1);
      conjuncts(lv, t, out, depth + 1);
      return;
    }
    if (isConstantFalse(t)) {
      out.push_back(c); // negated, but a conjunct nonetheless
      conjuncts(lv, f, out, depth + 1);
      return;
    }
  }
  if (auto ifOp = dyn_cast<scf::IfOp>(d)) {
    unsigned k = cast<OpResult>(v).getResultNumber();
    Value thenV = ifOp.thenYield().getOperand(k);
    Value elseV = ifOp.elseYield().getOperand(k);
    if (isConstantFalse(thenV)) {
      out.push_back(ifOp.getCondition());
      conjuncts(lv, elseV, out, depth + 1);
      return;
    }
    if (isConstantFalse(elseV)) {
      out.push_back(ifOp.getCondition());
      conjuncts(lv, thenV, out, depth + 1);
      return;
    }
  }
  out.push_back(v);
}

// The values a carried value may be fed on the back edge, looking through
// the scf.if that a capped loop's body sits under (one branch is the exit).
static void nextCandidates(const LoopView &lv, Value next,
                           SmallVectorImpl<Value> &out, unsigned depth = 0) {
  if (auto ifOp = dyn_cast_or_null<scf::IfOp>(lv.insideDef(next));
      ifOp && depth < 8) {
    unsigned k = cast<OpResult>(next).getResultNumber();
    nextCandidates(lv, ifOp.thenYield().getOperand(k), out, depth + 1);
    nextCandidates(lv, ifOp.elseYield().getOperand(k), out, depth + 1);
    return;
  }
  out.push_back(next);
}

static bool isConvergenceLoop(const LoopView &lv) {
  // 1. A convergence test among the conjuncts of the condition.
  SmallVector<Value> terms;
  conjuncts(lv, lv.pred, terms);
  bool found = false;
  for (Value term : terms) {
    Operation *cmp = lv.insideDef(term);
    Value quantity, bound;
    if (!cmp || !asGreaterThan(cmp, quantity, bound))
      continue;
    if (!dependsOnCarry(lv, quantity))
      continue;
    // The quantity must be a change of the state. The bound is not
    // inspected: a tolerance may be loop-invariant or relative to the state
    // (max(reltol * x, abstol)), and the state is itself built from the
    // step, so there is no structural property that separates it from the
    // quantity. What keeps non-convergence loops out is the accumulator rule
    // below, not the shape of the bound.
    DenseSet<Value> visitedQ;
    if (!isChangeOfState(lv, quantity, visitedQ))
      continue;
    found = true;
    break;
  }
  if (!found)
    return false;

  // 2. No accumulators among the float carries, and 3. no float computation
  //    depending on an integer counter.
  for (unsigned i = 0; i < lv.before->getNumArguments(); ++i) {
    Value arg = lv.before->getArgument(i);
    Value twin = lv.afterTwin[i];
    Value next = lv.backedge[i];
    if (isFloatLike(arg.getType())) {
      SmallVector<Value> candidates;
      nextCandidates(lv, next, candidates);
      for (Value cand : candidates) {
        Operation *d = lv.insideDef(cand);
        if (!d || !(isAddOp(d) || isSubOp(d)))
          continue;
        Value other = nullptr;
        if (d->getOperand(0) == arg || (twin && d->getOperand(0) == twin))
          other = d->getOperand(1);
        else if (isAddOp(d) && (d->getOperand(1) == arg ||
                                (twin && d->getOperand(1) == twin)))
          other = d->getOperand(0);
        if (!other)
          continue;
        DenseSet<Value> visited;
        if (!dependsOnArg(lv, other, arg, twin, visited))
          return false; // s + v with v independent of s: an accumulator
      }
    } else if (isIntLike(arg.getType())) {
      // A counter: anything float that depends on it makes the step
      // iteration-dependent.
      SmallVector<Value> work{arg};
      if (twin)
        work.push_back(twin);
      DenseSet<Value> seen;
      while (!work.empty()) {
        Value v = work.pop_back_val();
        if (!seen.insert(v).second)
          continue;
        for (Operation *user : v.getUsers()) {
          if (!lv.isInside(user))
            continue;
          // Deciding whether to take the step (the cap test feeding an if, a
          // select or the predicate) is not the step depending on the counter.
          if (auto ifOp = dyn_cast<scf::IfOp>(user);
              ifOp && ifOp.getCondition() == v)
            continue;
          if (isa<arith::SelectOp, LLVM::SelectOp, stablehlo::SelectOp>(user) &&
              user->getOperand(0) == v)
            continue;
          if (user->hasTrait<OpTrait::IsTerminator>())
            continue;
          for (Value res : user->getResults()) {
            if (isFloatLike(res.getType()))
              return false;
            work.push_back(res);
          }
        }
      }
    }
  }
  return true;
}

// `__enzyme_fixed_point([reduction[, max_iters]])` called in a loop's body
// marks that loop, with those settings when given as constants. The call
// itself goes away either way: it names an external function nothing defines.
static void applyFixedPointDirectives(Operation *root) {
  SmallVector<LLVM::CallOp> calls;
  root->walk([&](LLVM::CallOp call) {
    if (auto callee = call.getCallee();
        callee && callee->contains("__enzyme_fixed_point"))
      calls.push_back(call);
  });
  SetVector<Operation *> decls;
  for (LLVM::CallOp call : calls) {
    if (auto decl = SymbolTable::lookupNearestSymbolFrom(
            call, call.getCalleeAttr()))
      decls.insert(decl);
    auto loop = call->getParentOfType<scf::WhileOp>();
    if (!loop) {
      call.emitWarning("__enzyme_fixed_point is not inside a while loop");
    } else {
      Builder b(call);
      loop->setAttr(FixedPointOp::kMarkerAttrName, b.getUnitAttr());
      FloatAttr reduction;
      if (call.getNumOperands() > 0 &&
          matchPattern(call.getOperand(0), m_Constant(&reduction)))
        loop->setAttr(FixedPointOp::kMarkerReductionAttrName,
                      b.getF64FloatAttr(reduction.getValueAsDouble()));
      IntegerAttr maxIters;
      if (call.getNumOperands() > 1 &&
          matchPattern(call.getOperand(1), m_Constant(&maxIters)))
        loop->setAttr(FixedPointOp::kMarkerMaxItersAttrName,
                      b.getI64IntegerAttr(maxIters.getInt()));
    }
    call.erase();
  }
  for (Operation *decl : decls)
    if (root->isProperAncestor(decl) &&
        SymbolTable::symbolKnownUseEmpty(decl, decl->getParentOp()))
      decl->erase();
}

struct RecognizeFixedPointLoopsPass
    : public enzyme::impl::RecognizeFixedPointLoopsPassBase<
          RecognizeFixedPointLoopsPass> {
  using RecognizeFixedPointLoopsPassBase::RecognizeFixedPointLoopsPassBase;

  void runOnOperation() override {
    applyFixedPointDirectives(getOperation());
    if (!heuristic)
      return;
    getOperation()->walk([&](Operation *op) {
      if (op->hasAttr(FixedPointOp::kMarkerAttrName))
        return;
      std::optional<LoopView> lv;
      if (auto w = dyn_cast<stablehlo::WhileOp>(op))
        lv = viewOf(w);
      else if (auto w = dyn_cast<scf::WhileOp>(op))
        lv = viewOf(w);
      if (lv && isConvergenceLoop(*lv))
        op->setAttr(FixedPointOp::kMarkerAttrName,
                    UnitAttr::get(op->getContext()));
    });
  }
};

} // namespace
