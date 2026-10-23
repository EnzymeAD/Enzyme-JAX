//===- FixedPointLoops.cpp - Loops marked as fixed points <-> the op ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// apply-fixed-point-directives: a kernel loop that calls
// __enzyme_set_fixed_point (Reactant's `fixed_point_implicit`) is marked
// enzyme.fixed_point.
//
// raise-fixed-point-loops: a stablehlo.while carrying the marker becomes an
// enzymexla.fixed_point, which the Enzyme pass differentiates at its converged
// state and the StableHLO optimizations leave alone.
//
// lower-fixed-point-loops: the op becomes a marked stablehlo.while again.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "stablehlo/dialect/StablehloOps.h"

#include "src/enzyme_ad/jax/Dialect/Dialect.h"
#include "src/enzyme_ad/jax/Dialect/Ops.h"
#include "src/enzyme_ad/jax/Passes/Passes.h"

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_RAISEFIXEDPOINTLOOPSPASS
#define GEN_PASS_DEF_LOWERFIXEDPOINTLOOPSPASS
#define GEN_PASS_DEF_APPLYFIXEDPOINTDIRECTIVESPASS
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
// Directives
//===----------------------------------------------------------------------===//

namespace {

// `__enzyme_set_fixed_point(reduction, max_iters, control, state, bytes, ...)`
// called in a loop's body marks that loop, as for LLVM Enzyme (its
// include/enzyme/fixed_point.h). A negative reduction or max_iters selects the
// default, and so does a value that is not a constant. The state is the
// loop's carried values here, so the state arguments are not read. The call
// itself goes away either way: it names an external function nothing defines.
static void applyFixedPointDirectives(Operation *root) {
  SmallVector<LLVM::CallOp> calls;
  root->walk([&](LLVM::CallOp call) {
    if (auto callee = call.getCallee();
        callee && callee->contains("__enzyme_set_fixed_point"))
      calls.push_back(call);
  });
  SetVector<Operation *> decls;
  for (LLVM::CallOp call : calls) {
    if (auto decl = SymbolTable::lookupNearestSymbolFrom(
            call, call.getCalleeAttr()))
      decls.insert(decl);
    auto loop = call->getParentOfType<scf::WhileOp>();
    if (!loop) {
      call.emitWarning("__enzyme_set_fixed_point is not inside a while loop");
    } else {
      Builder b(call);
      loop->setAttr(FixedPointOp::kMarkerAttrName, b.getUnitAttr());
      FloatAttr reduction;
      if (call.getNumOperands() > 0 &&
          matchPattern(call.getOperand(0), m_Constant(&reduction)) &&
          reduction.getValueAsDouble() >= 0)
        loop->setAttr(FixedPointOp::kMarkerReductionAttrName,
                      b.getF64FloatAttr(reduction.getValueAsDouble()));
      IntegerAttr maxIters;
      if (call.getNumOperands() > 1 &&
          matchPattern(call.getOperand(1), m_Constant(&maxIters)) &&
          !maxIters.getValue().isNegative())
        loop->setAttr(FixedPointOp::kMarkerMaxItersAttrName,
                      b.getI64IntegerAttr(maxIters.getInt()));
      if (call.getNumOperands() > 2 &&
          !call.getOperand(2).getDefiningOp<LLVM::ZeroOp>())
        call.emitWarning("__enzyme_set_fixed_point: a control function is not "
                         "supported, the default convergence test is used");
    }
    call.erase();
  }
  for (Operation *decl : decls)
    if (root->isProperAncestor(decl) &&
        SymbolTable::symbolKnownUseEmpty(decl, decl->getParentOp()))
      decl->erase();
}

struct ApplyFixedPointDirectivesPass
    : public enzyme::impl::ApplyFixedPointDirectivesPassBase<
          ApplyFixedPointDirectivesPass> {
  void runOnOperation() override { applyFixedPointDirectives(getOperation()); }
};

} // namespace
