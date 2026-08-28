//===- AffineToStableHLORaising.cpp - Raise affine loops to stablehlo -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to raise affine loops functions to stablehlo.
//
//===----------------------------------------------------------------------===//

#include "src/enzyme_ad/jax/Passes/AffineUtils.h"
#include "src/enzyme_ad/jax/Passes/EnzymeHLOUnroll.h"
#include "src/enzyme_ad/jax/Passes/Passes.h"
#include "src/enzyme_ad/jax/Utils.h"

#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <numeric>

#include "Interfaces/AutoDiffTypeInterface.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"

#include "src/enzyme_ad/jax/Dialect/Ops.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "llvm/ADT/SmallSet.h"
#include <cassert>
#include <isl/ctx.h>
#include <isl/ilp.h>
#include <isl/map.h>
#include <isl/set.h>
#include <isl/space.h>
#include <isl/val.h>
#include <limits>
#include <optional>

namespace mlir {

static Block *getAllocaBlock(Operation *op) {
  Operation *currentOp = op;
  while (Operation *parentOp = currentOp->getParentOp()) {
    if (parentOp->mightHaveTrait<OpTrait::IsIsolatedFromAbove>() ||
        parentOp->mightHaveTrait<OpTrait::AutomaticAllocationScope>()) {
      return &currentOp->getParentRegion()->front();
    }
    currentOp = parentOp;
  }
  return nullptr;
}

namespace enzyme {
#define GEN_PASS_DEF_AFFINETOSTABLEHLORAISING
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

#define DEBUG_TYPE "raise-affine-to-stablehlo"

using namespace mlir;
using namespace mlir::enzyme;

bool isXLACompatiblePrimitive(Type ty) {
  if (isa<FloatType>(ty))
    return true;
  if (isa<IntegerType>(ty))
    return true;
  if (isa<ComplexType>(ty))
    return true;
  return false;
}

Type makeIndexToI64(Type ty) {
  if (isa<IndexType>(ty))
    return IntegerType::get(ty.getContext(), 64);

  if (auto tenTy = dyn_cast<RankedTensorType>(ty))
    return RankedTensorType::get(tenTy.getShape(),
                                 makeIndexToI64(tenTy.getElementType()));

  return ty;
}

mlir::Location rewriteLocation(mlir::Location loc, bool strip_llvm_debuginfo) {
  if (auto csl = dyn_cast<CallSiteLoc>(loc)) {
    auto callee = rewriteLocation(csl.getCallee(), strip_llvm_debuginfo);
    auto caller = rewriteLocation(csl.getCaller(), strip_llvm_debuginfo);
    if (isa<UnknownLoc>(callee)) {
      return caller;
    }
    return CallSiteLoc::get(callee, caller);
  }
  if (auto si = dyn_cast<FusedLocWith<LLVM::DISubprogramAttr>>(loc)) {
    if (strip_llvm_debuginfo)
      return UnknownLoc::get(si.getContext());
    auto attr = si.getMetadata();
    auto fl = FileLineColLoc::get(attr.getContext(), attr.getFile().getName(),
                                  attr.getLine(), 0);
    return Location(NameLoc::get(attr.getName(), fl));
  }
  return loc;
}

// This represents the values taken from an induction variable with the
// following syntax: [lb:ub:step]. ub is non-inclusive.
struct InductionVariableRange {
  int64_t lb;
  int64_t ub;
  int64_t step;

  int64_t getNumIters() {
    if (ub <= lb) {
      if (step < 0)
        return (lb - ub + (-step) - 1) / (-step);
      return 0;
    }
    return (ub - lb + step - 1) / step;
  }
};

static unsigned getIVPos(affine::AffineValueMap map, AffineExpr expr) {
  assert(!expr.isSymbolicOrConstant());
  unsigned pos;
  expr.walk([&pos](AffineExpr expr) {
    if (auto dimExpr = dyn_cast<AffineDimExpr>(expr)) {
      pos = dimExpr.getPosition();
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return pos;
}

// Assumes a single IV per Expr. (i) -> (i * 3 + 2)
static Value getIVForExpr(affine::AffineValueMap map, AffineExpr expr) {
  unsigned pos = getIVPos(map, expr);
  return map.getOperand(pos);
}

// has single (or zero) iv per dim.
// iv are present only at one dim.
static bool needsGeneralScatterGather(affine::AffineValueMap accessValueMap) {
  bool repeatingIV = false;
  auto map = accessValueMap.getAffineMap();
  auto sz = map.getNumDims();
  SmallVector<bool> ivseen(sz, false);
  for (auto E : map.getResults()) {
    if (E.isSymbolicOrConstant())
      continue;
    bool moreThanOneIV = false;
    for (int iv = 0; iv < sz; ++iv) {
      if (!E.isFunctionOfDim(iv))
        continue;
      if (ivseen[iv]) {
        repeatingIV = true;
        break;
      }
      if (moreThanOneIV) {
        return true;
      }
      moreThanOneIV = true;
      ivseen[iv] = true;
    }
  }
  return repeatingIV;
}

static std::optional<int64_t> getConstant(AffineMap map) {
  if (map.isSingleConstant())
    return map.getSingleConstantResult();
  return std::nullopt;
}

static std::optional<InductionVariableRange> getIVRange(Value iv) {
  assert(affine::isAffineInductionVar(iv));

  if (auto owner = affine::getAffineParallelInductionVarOwner(iv)) {

    auto ivPos = cast<BlockArgument>(iv).getArgNumber();
    auto lb = getConstant(owner.getLowerBoundMap(ivPos));
    auto ub = getConstant(owner.getUpperBoundMap(ivPos));
    auto step = owner.getSteps()[ivPos];
    if (!lb || !ub)
      return std::nullopt;
    return InductionVariableRange{*lb, *ub, step};
  }
  if (auto owner = affine::getForInductionVarOwner(iv)) {
    auto lb = getConstant(owner.getLowerBoundMap());
    auto ub = getConstant(owner.getUpperBoundMap());
    auto step = owner.getStep();
    if (!lb || !ub)
      return std::nullopt;
    return InductionVariableRange{*lb, *ub, step.getSExtValue()};
  }
  llvm_unreachable("Not affine iv");
}

static std::optional<InductionVariableRange>
computeExprRange(affine::AffineValueMap map, AffineExpr expr) {
  InductionVariableRange range;

  if (auto dimExpr = dyn_cast<AffineDimExpr>(expr)) {
    Value iv = map.getOperand(dimExpr.getPosition());
    auto range_ = getIVRange(iv);

    if (!range_.has_value())
      return std::nullopt;

    range = *range_;
  } else if (auto binExpr = dyn_cast<AffineBinaryOpExpr>(expr)) {
    auto rhs = binExpr.getRHS();
    auto lhs = binExpr.getLHS();

    auto rhsConst = dyn_cast<AffineConstantExpr>(rhs);
    auto constantSide = rhsConst ? rhsConst : dyn_cast<AffineConstantExpr>(lhs);
    auto dynSide = rhsConst ? lhs : rhs;

    auto rangeDyn = computeExprRange(map, dynSide);

    if (!rangeDyn.has_value() || !constantSide)
      return std::nullopt;

    auto const_ = constantSide.getValue();

    auto kind = expr.getKind();
    switch (kind) {
    case AffineExprKind::Add:
      range.lb = rangeDyn->lb + const_;
      range.ub = rangeDyn->ub + const_;
      range.step = rangeDyn->step;
      break;
    case AffineExprKind::Mul:
      range.lb = rangeDyn->lb * const_;
      range.ub = rangeDyn->ub * const_;
      range.step = rangeDyn->step * const_;
      break;
    default:
      // unsupported
      return std::nullopt;
    }
  } else {
    return std::nullopt;
  }

  return std::optional<InductionVariableRange>{range};
}

static void
emitIVToStableHLO(OpBuilder &builder, Value iv, InductionVariableRange range,
                  IRMapping &mapping,
                  llvm::DenseMap<Value, affine::AffineValueMap> &maps,
                  bool strip_llvm_debuginfo) {
  auto ET = builder.getI64Type();
  auto Ty = RankedTensorType::get({range.getNumIters()}, ET);
  Value iota =
      stablehlo::IotaOp::create(
          builder, rewriteLocation(iv.getLoc(), strip_llvm_debuginfo), Ty, 0)
          .getResult();
  iota = stablehlo::AddOp::create(
      builder, rewriteLocation(iv.getLoc(), strip_llvm_debuginfo), Ty, iota,
      stablehlo::ConstantOp::create(
          builder, rewriteLocation(iv.getLoc(), strip_llvm_debuginfo), Ty,
          SplatElementsAttr::get(
              Ty, ArrayRef<Attribute>(IntegerAttr::get(ET, range.lb)))));
  iota = stablehlo::MulOp::create(
      builder, rewriteLocation(iv.getLoc(), strip_llvm_debuginfo), Ty, iota,
      stablehlo::ConstantOp::create(
          builder, rewriteLocation(iv.getLoc(), strip_llvm_debuginfo), Ty,
          SplatElementsAttr::get(
              Ty, ArrayRef<Attribute>(IntegerAttr::get(ET, range.step)))));
  mapping.map(iv, iota);

  // contiguous with respect to itself: (d0) -> (d0)
  affine::AffineValueMap accessMap(
      AffineMap::getMultiDimIdentityMap(1, iv.getContext()), {iv});
  maps[iota] = accessMap;
}

// The name is parallel context but a more accurate description would be
// LockStepContext
struct ParallelContext {
  struct Options {
    bool enableLockstepFor = true;
    bool dump_failed_lockstep = false;
    bool preferWhileRaising = true;
    bool strip_llvm_debuginfo = false;
  } options;

  explicit ParallelContext(Options &options) : options(options) {}

  SmallVector<InductionVariableRange, 8> ranges;
  SmallVector<Value, 8> ivs;
  Value mask = nullptr;

  bool isParallelIV(Value iv) { return llvm::is_contained(ivs, iv); }

  RankedTensorType getTensorType(Type elTy) {
    SmallVector<int64_t> shape = llvm::map_to_vector(
        ranges, [&](auto range) { return range.getNumIters(); });
    assert(isXLACompatiblePrimitive(elTy) &&
           "unsupported element type for XLA");
    return RankedTensorType::get(shape, elTy);
  }

  struct Broadcast {
    Value v;
    affine::AffineValueMap avm;
  };

  std::optional<Broadcast> getBroadcast(OpBuilder &b,
                                        affine::AffineValueMap avm, Value v) {
    auto CTT = dyn_cast<RankedTensorType>(v.getType());
    if (!CTT)
      return std::nullopt;
    auto TT = getTensorType(CTT.getElementType());
    assert(TT.getElementType() == CTT.getElementType());
    if (CTT.getRank() > TT.getRank())
      return std::nullopt;

    SmallVector<int64_t> broadcastDims(CTT.getRank(), -1);

    for (auto [i, E] : llvm::enumerate(avm.getAffineMap().getResults())) {
      if (E.isSymbolicOrConstant())
        return std::nullopt;

      Value iv = nullptr;
      for (auto [j, I] : llvm::enumerate(avm.getOperands())) {
        if (E.isFunctionOfDim(j)) {
          if (!iv)
            iv = I;
          else
            return std::nullopt;
        }
      }

      int64_t pos = 0;

      for (auto I : ivs) {
        if (iv == I) {
          broadcastDims[i] = pos;
          break;
        }
        pos++;
      }

      if (pos == ivs.size())
        return std::nullopt;

      broadcastDims[i] = pos;
    }

    for (auto bdim : broadcastDims)
      if (bdim == -1)
        return std::nullopt;

    Value br = stablehlo::BroadcastInDimOp::create(b, v.getLoc(), TT, v,
                                                   broadcastDims);

    affine::AffineValueMap TMap(
        AffineMap::getMultiDimIdentityMap(TT.getRank(), b.getContext()), ivs);
    return Broadcast{.v = br, .avm = TMap};
  }

  std::optional<ParallelContext> add(affine::AffineForOp forOp) {
    ParallelContext newPc = *this;
    auto iv = forOp.getInductionVar();
    auto ivr = getIVRange(iv);
    if (!ivr)
      return std::nullopt;
    newPc.ranges.push_back(*ivr);
    newPc.ivs.push_back(iv);
    return newPc;
  }

  std::optional<ParallelContext> add(affine::AffineParallelOp parallelOp) {
    ParallelContext newPc = *this;
    for (auto iv : parallelOp.getIVs()) {
      auto ivr = getIVRange(iv);
      if (!ivr)
        return std::nullopt;
      newPc.ranges.push_back(*ivr);
      newPc.ivs.push_back(iv);
    }
    return newPc;
  }

  static std::optional<ParallelContext> get(affine::AffineParallelOp parallelOp,
                                            Options &options) {
    ParallelContext pc(options);
    return pc.add(parallelOp);
  }
  static ParallelContext getEmpty(Options &options) {
    return ParallelContext(options);
  }
};

// Given an affine map for a load/store operation, compute the startIndices,
// limitIndices and strides corresponding in the memref based on the loop
// induction variables.
//
// (i) -> (0, i, 10) will give [0:1:1, begin:end:step, 10:11:1]
// (i) -> (2 * i, i + 2, 10) will give [begin*2:end*2:2*step,
// begin+2:end+2:step, 10:11:1]
//
// with begin:end:step corresponding to the range of the iv i.
static LogicalResult affineMapToSlice(affine::AffineValueMap accessValueMap,
                                      SmallVectorImpl<int64_t> &strides,
                                      SmallVectorImpl<int64_t> &reverseDims,
                                      ParallelContext pc) {
  auto rank = accessValueMap.getNumResults();

  strides.reserve(rank);

  for (unsigned i = 0; i < rank; i++) {
    auto expr = accessValueMap.getResult(i);

    if (auto constExpr = dyn_cast<AffineConstantExpr>(expr)) {
      strides.push_back(1);
      continue;
    }
    if (expr.walk([](AffineExpr e) {
              if (isa<AffineSymbolExpr>(e))
                return WalkResult::interrupt();
              return WalkResult::advance();
            })
            .wasInterrupted()) {
      return failure();
    }

    Value iv = getIVForExpr(accessValueMap, expr);
    if (affine::isAffineForInductionVar(iv) && !pc.isParallelIV(iv)) {
      strides.push_back(1);
      continue;
    }

    auto range = computeExprRange(accessValueMap, expr);

    if (!range.has_value())
      return failure();

    if (range->step < 0) {
      // 0:-1:-180 -> -179:1:1
      strides.push_back(-range->step);
      reverseDims.push_back(i);
    } else {
      strides.push_back(range->step);
    }
  }

  return success();
}

static SmallVector<int64_t>
affineMapShape(affine::AffineValueMap accessValueMap, ParallelContext pc) {
  AffineMap map = accessValueMap.getAffineMap();

  SmallVector<int64_t> shape;
  shape.reserve(map.getNumResults());

  for (auto E : map.getResults()) {
    if (E.isSymbolicOrConstant()) {
      shape.push_back(1);
      continue;
    }

    Value iv = getIVForExpr(accessValueMap, E);
    if (!iv)
      return {};
    if (affine::isAffineForInductionVar(iv) && !pc.isParallelIV(iv)) {
      shape.push_back(1);
      continue;
    }
    if (!affine::isAffineInductionVar(iv))
      return {};

    auto range = getIVRange(iv);
    if (!range.has_value())
      return {};

    shape.push_back(range->getNumIters());
  }

  return shape;
}

static affine::AffineValueMap
alignMemoryAccess(Value &a, affine::AffineValueMap src, Value *bs,
                  ArrayRef<affine::AffineValueMap> dsts, OpBuilder &builder,
                  ParallelContext pc, bool *ok = nullptr) {
  // NOTE a default-constructed AffineValueMap holds a null context, so its
  // getAffineMap() cannot even be probed; validity is reported through `ok`
  // and inputs must be maps the caller actually recorded.
  auto fail = [&]() {
    if (ok)
      *ok = false;
    return affine::AffineValueMap();
  };
  if (ok)
    *ok = true;
  if (!a)
    return fail();
  for (unsigned qi = 0; qi < dsts.size(); ++qi)
    if (!bs[qi])
      return fail();
  // -> tensor<10x1xf32> loaded from (i) -> (i, 0)
  // -> to tensor<1x10xf32> written as (i) -> (0, i)

  // affineMapShape bails to an empty vector on accesses it cannot size
  // (an IV with no static range); that must reject the alignment, not read
  // past the end below.
  SmallVector<int64_t> shapeA = affineMapShape(src, pc);
  if (shapeA.size() != cast<RankedTensorType>(a.getType()).getShape().size())
    return fail();
  SmallVector<SmallVector<int64_t>> shapeBs;
  for (size_t i = 0; i < dsts.size(); i++) {
    shapeBs.push_back(affineMapShape(dsts[i], pc));
    if (shapeBs[i].size() !=
        cast<RankedTensorType>(bs[i].getType()).getShape().size())
      return fail();
  }

  SmallVector<int64_t> outputShape;

  SmallVector<int64_t> broadcastDimensionsA(shapeA.size(), -1);
  SmallVector<SmallVector<int64_t>> broadcastDimensionsBs;
  for (auto shapeB : shapeBs)
    broadcastDimensionsBs.emplace_back(shapeB.size(), -1);

  SmallVector<AffineExpr> exprs;
  SmallVector<Value> mapOperands;

  SetVector<Value> ivs;

  bool needsBroadcastA = false;
  SmallVector<bool> needsBroadcastBs(shapeBs.size(), false);

  for (auto [i, EA] : llvm::enumerate(src.getAffineMap().getResults())) {
    broadcastDimensionsA[i] = outputShape.size();

    Value ivA = getIVForExpr(src, EA);

    for (auto [dst, broadcastDimensionsB] :
         llvm::zip(dsts, broadcastDimensionsBs)) {

      for (unsigned j = 0, e = dst.getNumResults(); j < e; ++j) {
        auto EB = dst.getAffineMap().getResult(j);
        if (getIVForExpr(dst, EB) == ivA) {
          broadcastDimensionsB[j] = outputShape.size();
        }
      }
    }

    outputShape.push_back(shapeA[i]);

    // A result with no induction variable -- a constant or symbolic access
    // -- is a unit dimension: nothing iterates it, so nothing maps to it.
    if (ivA) {
      exprs.push_back(
          mlir::getAffineDimExpr(mapOperands.size(), ivA.getContext()));
      mapOperands.push_back(ivA);
    } else {
      exprs.push_back(
          mlir::getAffineConstantExpr(0, src.getAffineMap().getContext()));
    }
  }

  for (auto &&[dst, broadcastDimensionsB, shapeB] :
       llvm::zip(dsts, broadcastDimensionsBs, shapeBs)) {
    for (auto [i, EB] : llvm::enumerate(dst.getAffineMap().getResults())) {
      if (broadcastDimensionsB[i] != -1)
        continue; // dim already set in A

      needsBroadcastA = true;

      Value ivB = getIVForExpr(dst, EB);

      for (auto &&[dst2, broadcastDimensionsB2] :
           llvm::zip(dsts, broadcastDimensionsBs)) {
        for (unsigned j = 0, e = dst2.getNumResults(); j < e; ++j) {
          auto EB2 = dst2.getAffineMap().getResult(j);
          if (getIVForExpr(dst2, EB2) == ivB) {
            broadcastDimensionsB2[j] = outputShape.size();
          }
        }
      }

      outputShape.push_back(shapeB[i]);

      if (ivB) {
        exprs.push_back(
            mlir::getAffineDimExpr(mapOperands.size(), ivB.getContext()));
        mapOperands.push_back(ivB);
      } else {
        exprs.push_back(
            mlir::getAffineConstantExpr(0, src.getAffineMap().getContext()));
      }
    }
  }

  auto TA = cast<RankedTensorType>(a.getType());

  if (needsBroadcastA) {
    a = stablehlo::BroadcastInDimOp::create(
            builder,
            rewriteLocation(a.getLoc(), pc.options.strip_llvm_debuginfo),
            TA.clone(outputShape), a, broadcastDimensionsA)
            .getResult();
  }

  for (size_t i = 0; i < dsts.size(); i++) {
    auto TB = cast<RankedTensorType>(bs[i].getType());

    bool needsBroadcast = false;
    if (TB.getShape().size() == outputShape.size()) {
      for (auto bdim : llvm::enumerate(broadcastDimensionsBs[i])) {
        if (bdim.index() != bdim.value()) {
          needsBroadcast = true;
          break;
        }
      }
    } else
      needsBroadcast = true;

    if (needsBroadcast)
      bs[i] =
          stablehlo::BroadcastInDimOp::create(
              builder,
              rewriteLocation(bs[i].getLoc(), pc.options.strip_llvm_debuginfo),
              TB.clone(outputShape), bs[i], broadcastDimensionsBs[i])
              .getResult();
  }

  // One result per output dimension: an identity dim for each operand, a
  // constant for each unit dimension nothing iterates.
  affine::AffineValueMap outputMap(
      AffineMap::get(mapOperands.size(), 0, exprs, a.getContext()),
      mapOperands);

  return outputMap;
}

static affine::AffineValueMap
alignMemoryAccess(Value &a, affine::AffineValueMap src, Value &b,
                  affine::AffineValueMap dst, OpBuilder &builder,
                  ParallelContext pc, bool *ok = nullptr) {
  Value bs[] = {b};
  affine::AffineValueMap dsts[] = {dst};
  auto res = alignMemoryAccess(a, src, bs, dsts, builder, pc, ok);
  b = bs[0];
  return res;
}

// like affine::expandAffineExpr but with stablehlo ops and returning
// the corresponding AffineValueMap for the produced value.
static std::tuple<Value, affine::AffineValueMap>
expandAffineExpr(OpBuilder &builder, Location loc, AffineExpr expr,
                 ValueRange operands, IRMapping &mapping,
                 llvm::DenseMap<Value, affine::AffineValueMap> &maps,
                 unsigned numDims, ParallelContext pc) {
  if (auto constExpr = dyn_cast<AffineConstantExpr>(expr)) {
    auto ET = builder.getI64Type();
    auto TT = RankedTensorType::get({}, ET);
    Value res = stablehlo::ConstantOp::create(
        builder, loc, TT,
        SplatElementsAttr::get(TT, ArrayRef<Attribute>(IntegerAttr::get(
                                       ET, constExpr.getValue()))));
    return {res, affine::AffineValueMap(AffineMap::get(expr.getContext()), {})};
  }

  if (auto binExpr = dyn_cast<AffineBinaryOpExpr>(expr)) {
    AffineExpr lhsExpr = binExpr.getLHS(), rhsExpr = binExpr.getRHS();
    auto [lhs, lhsMap] = expandAffineExpr(builder, loc, lhsExpr, operands,
                                          mapping, maps, numDims, pc);
    auto [rhs, rhsMap] = expandAffineExpr(builder, loc, rhsExpr, operands,
                                          mapping, maps, numDims, pc);

    bool alignOk = true;
    affine::AffineValueMap outputMap =
        alignMemoryAccess(lhs, lhsMap, rhs, rhsMap, builder, pc, &alignOk);
    if (!alignOk)
      return {nullptr, affine::AffineValueMap()};

    auto makeI64Constant = [loc, &builder](ShapedType ty,
                                           int64_t cst) -> Value {
      return stablehlo::ConstantOp::create(
                 builder, loc, ty,

                 SplatElementsAttr::get(
                     ty, ArrayRef<Attribute>(
                             IntegerAttr::get(ty.getElementType(), cst))))
          .getResult();
    };

    Value result;
    switch (expr.getKind()) {
    case AffineExprKind::Add:
      result = stablehlo::AddOp::create(builder, loc, lhs, rhs);
      break;
    case AffineExprKind::Mul:
      result = stablehlo::MulOp::create(builder, loc, lhs, rhs);
      break;
    case AffineExprKind::Mod:
      // a mod b =
      //     let remainder = srem a, b;
      //         negative = a < 0 in
      //     select negative, remainder + b, remainder.
      {
        Value remainder = stablehlo::RemOp::create(builder, loc, lhs, rhs);
        Value negative = stablehlo::CompareOp::create(
            builder, loc, lhs,
            makeI64Constant(cast<ShapedType>(lhs.getType()), 0),
            stablehlo::ComparisonDirection::LT);
        result = stablehlo::SelectOp::create(
            builder, loc, negative,
            stablehlo::AddOp::create(builder, loc, remainder, rhs), remainder);
      };
      break;
    case AffineExprKind::FloorDiv:
      // a floordiv b =
      //     let negative = a < 0 in
      //     let absolute = negative ? -a - 1 : a in
      //     let quotient = absolute / b in
      //         negative ? -quotient - 1 : quotient
      {
        Value negative = stablehlo::CompareOp::create(
            builder, loc, lhs,
            makeI64Constant(cast<ShapedType>(lhs.getType()), 0),
            stablehlo::ComparisonDirection::LT);
        Value one = makeI64Constant(cast<ShapedType>(lhs.getType()), 1);
        Value absolute = stablehlo::SelectOp::create(
            builder, loc, negative,
            stablehlo::SubtractOp::create(
                builder, loc, stablehlo::NegOp::create(builder, loc, lhs), one),
            lhs);
        Value quotient = stablehlo::DivOp::create(builder, loc, absolute, rhs);
        result = stablehlo::SelectOp::create(
            builder, loc, negative,
            stablehlo::SubtractOp::create(
                builder, loc, stablehlo::NegOp::create(builder, loc, quotient),
                one),
            quotient);
      };
      break;
    case AffineExprKind::CeilDiv:
      // a ceildiv b =
      //     let negative = a <= 0 in
      //     let absolute = negative ? -a : a - 1 in
      //     let quotient = absolute / b in
      //         negative ? -quotient : quotient + 1
      {
        Value negative = stablehlo::CompareOp::create(
            builder, loc, lhs,
            makeI64Constant(cast<ShapedType>(lhs.getType()), 0),
            stablehlo::ComparisonDirection::LE);
        Value one = makeI64Constant(cast<ShapedType>(lhs.getType()), 1);
        Value negLhs = stablehlo::NegOp::create(builder, loc, lhs);
        Value lhsPlusOne = stablehlo::AddOp::create(builder, loc, lhs, one);
        Value absolute = stablehlo::SelectOp::create(builder, loc, negative,
                                                     negLhs, lhsPlusOne);
        Value quotient = stablehlo::DivOp::create(builder, loc, absolute, rhs);
        Value negQuotient = stablehlo::NegOp::create(builder, loc, quotient);
        Value quotientPlusOne =
            stablehlo::AddOp::create(builder, loc, quotient, one);
        result = stablehlo::SelectOp::create(builder, loc, negative,
                                             negQuotient, quotientPlusOne);
      };
      break;
    default:
      llvm_unreachable("unsupported expansion of expr");
    }
    return {result, outputMap};
  }

  if (auto symExpr = dyn_cast<AffineSymbolExpr>(expr)) {
    Value sym = operands[symExpr.getPosition() + numDims];
    Value mapped = mapping.lookupOrNull(sym);
    if (mapped && maps.count(mapped))
      return {mapped, maps.lookup(mapped)};
    return {mapped,
            affine::AffineValueMap(AffineMap::get(sym.getContext()), {})};
  }

  if (auto dimExpr = dyn_cast<AffineDimExpr>(expr)) {
    Value dim = operands[dimExpr.getPosition()];
    Value mapped = mapping.lookupOrNull(dim);

    if (!pc.isParallelIV(dim)) {
      // A masked loop's induction variable maps to a lane tensor carrying
      // its own access map; a uniform while counter maps to a scalar.
      if (mapped && maps.count(mapped))
        return {mapped, maps.lookup(mapped)};
      return {mapped,
              affine::AffineValueMap(AffineMap::get(dim.getContext()), {})};
    }

    return {mapped, affine::AffineValueMap(
                        AffineMap::getMultiDimIdentityMap(1, expr.getContext()),
                        {dim})};
  }

  llvm_unreachable("unreachable");
}

// A for-loop induction variable whose raised value is a lane tensor (a
// masked while over per-lane bounds) cannot use the slice path: its start
// index is not a scalar.
static bool usesLaneTensorIV(affine::AffineValueMap &avm, IRMapping &mapping,
                             ParallelContext &pc) {
  for (Value opnd : avm.getOperands()) {
    if (!affine::isAffineForInductionVar(opnd) || pc.isParallelIV(opnd))
      continue;
    Value mapped = mapping.lookupOrNull(opnd);
    if (!mapped)
      continue;
    auto tt = dyn_cast<RankedTensorType>(mapped.getType());
    if (tt && tt.getRank() > 0)
      return true;
  }
  return false;
}

/// scope is an operation _in_ the scope we are interested in
bool isSafeToSpeculativelyExecuteAtScope(Operation *scope, Operation *op) {
  if (mlir::isPure(op))
    return true;

  MemRefType ty = nullptr;
  if (auto read = dyn_cast<affine::AffineReadOpInterface>(op))
    ty = read.getMemRefType();
  if (!ty)
    return false;

  // 0-dim memrefs are always legal to speculate (assuming non-null)
  if (ty.getShape().size() == 0)
    return true;

  IslAnalysis ia;

  isl_set *array = ia.getMemrefShape(ty);
  if (!array)
    return false;

  isl_map *accessMap = ia.getAccessMap(op);
  if (!accessMap) {
    isl_set_free(array);
    return false;
  }

  isl_set *domain = ia.getDomain(scope);
  if (!domain) {
    isl_set_free(array);
    isl_map_free(accessMap);
    return false;
  }
  isl_set *accessed = isl_set_apply(domain, accessMap);
  isl_bool inBounds = isl_set_is_subset(accessed, array);
  isl_set_free(array);
  isl_set_free(accessed);
  if (inBounds == isl_bool_error)
    return false;
  return inBounds;
}

static LogicalResult
tryRaisingOpToStableHLO(Operation *op, IRMapping &mapping, OpBuilder &builder,
                        llvm::DenseMap<Value, affine::AffineValueMap> &maps,
                        ParallelContext pc);

// The buffers a loop must carry are the arguments of the block being raised
// from: the kernel function's entry, or the gpu_wrapper region's block when
// raising a wrapper in place.
static Block *getRaisedEntryBlock(Operation *op) {
  while (op->getParentOp() &&
         !isa<func::FuncOp, enzymexla::GPUWrapperOp>(op->getParentOp()))
    op = op->getParentOp();
  return &op->getParentRegion()->front();
}

static LogicalResult tryRaisingForOpToStableHLOWhile(
    affine::AffineForOp forOp, IRMapping &parentMapping, OpBuilder &builder,
    llvm::DenseMap<Value, affine::AffineValueMap> &maps, ParallelContext pc,
    stablehlo::WhileOp *createdWhileOp = nullptr,
    SmallVectorImpl<Value> *carriedBuffers = nullptr);

static LogicalResult
emitIfAsSelect(Operation *ifOp, Value cond, affine::AffineValueMap map,
               OpBuilder &builder, IRMapping &mapping,
               DenseMap<Value, affine::AffineValueMap> &maps,
               ParallelContext pc) {
  Block *thenBlock = &ifOp->getRegion(0).front();
  Block *elseBlock =
      ifOp->getRegion(1).empty() ? nullptr : &ifOp->getRegion(1).front();

  auto getMaskedCond = [&](Value cond, Value mask) {
    if (mask) {
      // merge mask and current cond
      affine::AffineValueMap maskMap = maps.lookup(mask),
                             condMap = maps.lookup(cond);
      bool alignOk = true;
      affine::AffineValueMap newMaskMap = alignMemoryAccess(
          mask, maskMap, cond, condMap, builder, pc, &alignOk);
      if (!alignOk)
        return Value();

      mask = stablehlo::AndOp::create(
          builder,
          rewriteLocation(ifOp->getLoc(), pc.options.strip_llvm_debuginfo),
          mask, cond);
      maps[mask] = newMaskMap;
    } else {
      mask = cond;
    }
    return mask;
  };

  Value mask = getMaskedCond(cond, pc.mask);
  if (pc.mask && !mask)
    return failure();

  ParallelContext thenPc(pc.options);
  thenPc.ranges = pc.ranges;
  thenPc.ivs = pc.ivs;
  thenPc.mask = mask;

  for (auto &innerOp : thenBlock->without_terminator()) {
    if (tryRaisingOpToStableHLO(&innerOp, mapping, builder, maps, thenPc)
            .failed())
      return failure();
  }

  ParallelContext elsePc(pc.options);
  elsePc.ranges = pc.ranges;
  elsePc.ivs = pc.ivs;

  if (elseBlock) {
    Value elseCond = stablehlo::NotOp::create(
        builder,
        rewriteLocation(ifOp->getLoc(), pc.options.strip_llvm_debuginfo), cond);
    maps[elseCond] = maps.lookup(cond);

    Value elseMask = getMaskedCond(elseCond, pc.mask);
    if (pc.mask && !elseMask)
      return failure();
    assert(maps.contains(elseMask));
    elsePc.mask = elseMask;

    for (auto &innerOp : elseBlock->without_terminator()) {
      if (tryRaisingOpToStableHLO(&innerOp, mapping, builder, maps, elsePc)
              .failed())
        return failure();
    }
  }

  Operation *thenTerm = thenBlock->getTerminator();
  Operation *elseTerm = elseBlock ? elseBlock->getTerminator() : nullptr;

  if (ifOp->getNumResults() > 0) {
    for (auto [thenVal, elseVal, res] :
         llvm::zip_equal(thenTerm->getOperands(), elseTerm->getOperands(),
                         ifOp->getResults())) {
      Value a = cond;
      if (isa<MemRefType, LLVM::LLVMPointerType>(res.getType()) &&
          res.use_empty())
        continue;
      if (isa<MemRefType, LLVM::LLVMPointerType>(res.getType())) {
        // A branch choosing between whole buffers raises as a select of the
        // whole tensors when the choice is uniform and nothing writes through
        // it; a write would have to fan back out into both source buffers.
        Value thenBuf = mapping.lookupOrNull(thenVal);
        Value elseBuf = mapping.lookupOrNull(elseVal);
        auto condTy = dyn_cast<RankedTensorType>(cond.getType());
        // The select captures the buffers as of this point, so nothing may
        // write to them (through the select or directly) or later reads
        // through the select would miss the write.
        auto loadOnly = [&](Value buf) {
          return llvm::all_of(buf.getUsers(), [&](Operation *user) {
            return isa<affine::AffineLoadOp, memref::LoadOp,
                       affine::AffineVectorLoadOp>(user) ||
                   user == ifOp || user == thenTerm || user == elseTerm;
          });
        };
        bool readOnly = loadOnly(res) && loadOnly(thenVal) && loadOnly(elseVal);
        if (!thenBuf || !elseBuf || thenBuf.getType() != elseBuf.getType() ||
            !condTy || condTy.getRank() != 0 || !readOnly) {
          if (getenv("DEBUG_BUFSEL")) {
            llvm::errs() << "BUFSEL fail: thenBuf=" << (bool)thenBuf
                         << " elseBuf=" << (bool)elseBuf << " tyEq="
                         << (thenBuf && elseBuf &&
                             thenBuf.getType() == elseBuf.getType())
                         << " condTy=" << (bool)condTy << " condRank="
                         << (condTy ? condTy.getRank() : -1)
                         << " readOnly=" << readOnly << "\n"
                         << *ifOp << "\n";
          }
          return ifOp->emitError(
              "cannot raise a branch choosing between buffers");
        }
        if (!maps.count(thenBuf))
          return ifOp->emitError(
              "cannot raise a branch choosing between buffers");
        auto sel = stablehlo::SelectOp::create(
            builder,
            rewriteLocation(ifOp->getLoc(), pc.options.strip_llvm_debuginfo),
            cond, thenBuf, elseBuf);
        mapping.map(res, sel.getResult());
        maps[sel.getResult()] = maps.lookup(thenBuf);
        continue;
      }
      Value b = mapping.lookupOrNull(thenVal);
      Value c = mapping.lookupOrNull(elseVal);
      if (!b || !c)
        return ifOp->emitError(
            "cannot raise branch result without an access map");
      auto itB = maps.find(b), itC = maps.find(c);
      if (itB == maps.end() || itC == maps.end())
        return ifOp->emitError(
            "cannot raise branch result without an access map");

      auto mapA = map;
      Value dsts[] = {b, c};
      affine::AffineValueMap submaps[] = {itB->second, itC->second};
      bool alignOk = true;
      auto outputMap =
          alignMemoryAccess(a, mapA, dsts, submaps, builder, pc, &alignOk);
      if (!alignOk)
        return ifOp->emitError(
            "cannot raise branch result without an access map");
      b = dsts[0];
      c = dsts[1];
      assert(b.getType() == c.getType());

      auto newOp = stablehlo::SelectOp::create(
          builder,
          rewriteLocation(ifOp->getLoc(), pc.options.strip_llvm_debuginfo), a,
          b, c);
      mapping.map(res, newOp.getResult());
      maps[newOp.getResult()] = outputMap;
    }
  }

  return success();
}

// Builds a `[gridShape..., numColumns]` index tensor from per-dimension index
// columns, deduplicating induction variables: when the same IV indexes more
// than one column, the columns share a single grid axis instead of forming a
// cartesian product. `ivs` is filled with the distinct IVs (one per grid axis,
// in first-appearance order) and `gridShape` with the extent of each axis.
// The grid axes are kept multi-dimensional (used as implicit batch dims with
// index_vector_dim = rank - 1) rather than flattened to a single dimension,
// whose extent could overflow for large grids.
static Value buildGatherScatterIndices(
    Location loc, ValueRange indexColumns, OpBuilder &builder,
    llvm::DenseMap<Value, affine::AffineValueMap> &maps,
    SmallVectorImpl<Value> &ivs, SmallVectorImpl<int64_t> &gridShape) {
  Value indices = nullptr;

  for (auto raisedIdx : indexColumns) {
    auto Ty = cast<RankedTensorType>(raisedIdx.getType());

    SmallVector<int64_t> dimsToBroadcast;

    if (!maps.count(raisedIdx))
      return nullptr;
    auto map = maps.lookup(raisedIdx);

    for (auto [i, E] : llvm::enumerate(map.getAffineMap().getResults())) {
      auto iv = getIVForExpr(map, E);

      unsigned ivPos = 0;
      for (unsigned e = ivs.size(); ivPos < e; ++ivPos) {
        if (ivs[ivPos] == iv) {
          break;
        }
      }

      if (ivPos == ivs.size()) {
        gridShape.push_back(Ty.getShape()[i]);
        dimsToBroadcast.push_back(ivs.size());
        ivs.push_back(iv);
      } else {
        // this dim is already present
        dimsToBroadcast.push_back((int64_t)ivPos);
      }
    }

    if (indices) {
      auto indicesTy = cast<RankedTensorType>(indices.getType());

      SmallVector<int64_t> raisedIdxShape(
          indicesTy.getShape().drop_back().begin(),
          indicesTy.getShape().drop_back().end());

      for (auto [dim, bdim] : llvm::enumerate(dimsToBroadcast)) {
        if (bdim >= raisedIdxShape.size()) {
          raisedIdxShape.push_back(Ty.getShape()[dim]);
        }
      }

      raisedIdxShape.push_back(1);

      raisedIdx = stablehlo::BroadcastInDimOp::create(
          builder, loc, Ty.clone(raisedIdxShape), raisedIdx, dimsToBroadcast);

      SmallVector<int64_t> shape(indicesTy.getShape().drop_back().begin(),
                                 indicesTy.getShape().drop_back().end());

      for (auto [dim, bdim] : llvm::enumerate(dimsToBroadcast)) {
        if (bdim >= shape.size()) {
          shape.push_back(Ty.getShape()[dim]);
        }
      }

      shape.push_back(indicesTy.getShape()[indicesTy.getShape().size() - 1]);
      SmallVector<int64_t> bDims;
      for (int i = 0, e = indicesTy.getShape().size() - 1; i < e; ++i)
        bDims.push_back(i);
      bDims.push_back(shape.size() - 1);

      indices = stablehlo::BroadcastInDimOp::create(
          builder, loc, Ty.clone(shape), indices, bDims);

      indicesTy = cast<RankedTensorType>(indices.getType());
      SmallVector<int64_t> newIndicesShape(
          indicesTy.getShape().drop_back().begin(),
          indicesTy.getShape().drop_back().end());
      newIndicesShape.push_back(
          indicesTy.getShape()[indicesTy.getShape().size() - 1] + 1);

      indices = stablehlo::ConcatenateOp::create(
          builder, loc, Ty.clone(newIndicesShape),
          ValueRange{indices, raisedIdx}, (int64_t)newIndicesShape.size() - 1);
    } else {

      auto S = cast<RankedTensorType>(raisedIdx.getType()).getShape();
      SmallVector<int64_t> shape(S.begin(), S.end());
      shape.push_back(1);

      indices = stablehlo::ReshapeOp::create(builder, loc, Ty.clone(shape),
                                             raisedIdx);
    }
  }

  return indices;
}

static Value
emitLoadAsGather(Location loc, Value mappedMemref, ValueRange lIndices,
                 OpBuilder &builder,
                 llvm::DenseMap<Value, affine::AffineValueMap> &maps) {
  SmallVector<int64_t> sliceSizes(lIndices.size(), 1);
  SmallVector<int64_t> startIndexMap;
  for (int64_t i = 0, e = lIndices.size(); i < e; ++i)
    startIndexMap.push_back(i);

  SmallVector<int64_t> outputShape;
  SmallVector<Value> ivs;

  Value indices =
      buildGatherScatterIndices(loc, lIndices, builder, maps, ivs, outputShape);
  if (!indices)
    return nullptr;

  // The grid axes of `indices` act as implicit batch dimensions, so the
  // gather result directly has shape `outputShape`.
  Value res =
      stablehlo::GatherOp::create(builder, loc, mappedMemref, indices,
                                  stablehlo::GatherDimensionNumbersAttr::get(
                                      loc.getContext(),
                                      /*offsetDims*/ {},
                                      /*collapsedSliceDims*/ startIndexMap,
                                      /*operandBatchingDims*/ {},
                                      /*startIndicesBatchingDims*/ {},
                                      /*startIndexMap*/ startIndexMap,
                                      /*indexVectorDim*/
                                      (int64_t)outputShape.size()),
                                  sliceSizes);

  affine::AffineValueMap outputMap(
      AffineMap::getMultiDimIdentityMap(ivs.size(), loc.getContext()), ivs);

  maps[res] = outputMap;

  return res;
}

static Value
emitStoreAsScatter(Location loc, Value update, Value input, ValueRange sIndices,
                   OpBuilder &builder,
                   llvm::DenseMap<Value, affine::AffineValueMap> &maps,
                   const ParallelContext &pc) {
  if (!maps.count(update))
    return nullptr;
  affine::AffineValueMap updateValueMap = maps.lookup(update);

  auto UTy = cast<RankedTensorType>(update.getType());

  // Build the scatter indices the same way the gather path does: one grid axis
  // per distinct induction variable, so an IV that indexes several memref
  // dimensions reuses a single axis instead of forming a cartesian product.
  SmallVector<Value> ivs;
  SmallVector<int64_t> gridShape;
  Value indices =
      buildGatherScatterIndices(loc, sIndices, builder, maps, ivs, gridShape);
  if (!indices)
    return nullptr;

  SmallVector<int64_t> scatterDimsToOperandDims;
  for (int64_t i = 0, e = sIndices.size(); i < e; ++i)
    scatterDimsToOperandDims.push_back(i);

  // Map each dimension of the stored value to the grid axis of the induction
  // variable that indexes it. Grid axes not targeted by any update dimension
  // are broadcast across (the stored value is constant along them).
  SmallVector<int64_t> broadcastDims(UTy.getShape().size(), -1);
  for (auto [updateIdx, E] :
       llvm::enumerate(updateValueMap.getAffineMap().getResults())) {
    Value updateIV = getIVForExpr(updateValueMap, E);
    for (auto [k, iv] : llvm::enumerate(ivs)) {
      if (iv == updateIV) {
        broadcastDims[updateIdx] = (int64_t)k;
        break;
      }
    }
  }

  // A value varying along an axis the store indices do not cover is a racy
  // write. With no mask along the axis, lane 0 is a sound refinement; with a
  // mask covering the update's space, pick any admitted lane's value.
  {
    SmallVector<int64_t> raceDims;
    for (auto [updateIdx, dim] : llvm::enumerate(broadcastDims))
      if (dim == -1)
        raceDims.push_back((int64_t)updateIdx);
    if (!raceDims.empty()) {
      bool maskVaries = false;
      if (pc.mask) {
        affine::AffineValueMap mm = maps.lookup(pc.mask);
        for (int64_t ri : raceDims) {
          Value riv = getIVForExpr(updateValueMap,
                                   updateValueMap.getAffineMap().getResult(ri));
          for (auto E : mm.getAffineMap().getResults())
            if (getIVForExpr(mm, E) == riv)
              maskVaries = true;
        }
      }
      Value newUpdate;
      if (!maskVaries) {
        SmallVector<int64_t> starts(UTy.getRank(), 0);
        SmallVector<int64_t> limits(UTy.getShape().begin(),
                                    UTy.getShape().end());
        SmallVector<int64_t> ones(UTy.getRank(), 1);
        for (int64_t ri : raceDims)
          limits[ri] = 1;
        newUpdate = stablehlo::SliceOp::create(builder, loc, update, starts,
                                               limits, ones);
        SmallVector<int64_t> keptShape;
        SmallVector<AffineExpr> keptExprs;
        SmallVector<int64_t> keptBroadcastDims;
        for (auto [updateIdx, dim] : llvm::enumerate(broadcastDims)) {
          if (llvm::is_contained(raceDims, (int64_t)updateIdx))
            continue;
          keptShape.push_back(UTy.getShape()[updateIdx]);
          keptExprs.push_back(
              updateValueMap.getAffineMap().getResult(updateIdx));
          keptBroadcastDims.push_back(dim);
        }
        update = stablehlo::ReshapeOp::create(
            builder, loc,
            RankedTensorType::get(keptShape, UTy.getElementType()), newUpdate);
        updateValueMap = affine::AffineValueMap(
            AffineMap::get(updateValueMap.getAffineMap().getNumDims(),
                           updateValueMap.getAffineMap().getNumSymbols(),
                           keptExprs, loc.getContext()),
            updateValueMap.getOperands());
        updateValueMap.composeSimplifyAndCanonicalize();
        broadcastDims.assign(keptBroadcastDims.begin(),
                             keptBroadcastDims.end());
      } else {
        // The mask can span axes the update lacks: work in their union.
        Value mAligned = pc.mask;
        Value updAligned = update;
        affine::AffineValueMap mm = maps.lookup(pc.mask);
        bool alignOk = true;
        affine::AffineValueMap unionMap = alignMemoryAccess(
            mAligned, mm, updAligned, updateValueMap, builder, pc, &alignOk);
        if (!alignOk)
          return nullptr;
        auto AT = cast<RankedTensorType>(updAligned.getType());
        if (mAligned.getType() !=
            RankedTensorType::get(AT.getShape(), builder.getI1Type()))
          return nullptr;
        // Reduce every union axis the scatter grid does not carry; a unit
        // axis (no IV) reduces harmlessly too.
        SmallVector<int64_t> unionRace;
        SmallVector<int64_t> keptShape;
        SmallVector<AffineExpr> keptExprs;
        SmallVector<int64_t> keptBroadcastDims;
        for (auto [i, E] :
             llvm::enumerate(unionMap.getAffineMap().getResults())) {
          Value uiv = getIVForExpr(unionMap, E);
          int64_t gridAxis = -1;
          for (auto [k, giv] : llvm::enumerate(ivs))
            if (giv == uiv)
              gridAxis = (int64_t)k;
          if (uiv && gridAxis != -1) {
            keptShape.push_back(AT.getShape()[i]);
            keptExprs.push_back(E);
            keptBroadcastDims.push_back(gridAxis);
          } else {
            unionRace.push_back((int64_t)i);
          }
        }
        if (unionRace.empty())
          return nullptr;
        auto elemTy = RankedTensorType::get({}, AT.getElementType());
        auto boolTy = RankedTensorType::get({}, builder.getI1Type());
        Value zeroInit = stablehlo::ConstantOp::create(
            builder, loc, elemTy,
            SplatElementsAttr::get(elemTy,
                                   builder.getZeroAttr(AT.getElementType())));
        Value falseInit = stablehlo::ConstantOp::create(
            builder, loc, boolTy,
            SplatElementsAttr::get(boolTy, builder.getBoolAttr(false)));
        auto reduce = stablehlo::ReduceOp::create(
            builder, loc, ValueRange{updAligned, mAligned},
            ValueRange{zeroInit, falseInit},
            builder.getDenseI64ArrayAttr(unionRace));
        {
          Block *rb = new Block();
          reduce.getBody().push_back(rb);
          Value av = rb->addArgument(elemTy, loc);
          Value am = rb->addArgument(boolTy, loc);
          Value bv = rb->addArgument(elemTy, loc);
          Value bm = rb->addArgument(boolTy, loc);
          OpBuilder::InsertionGuard guard(builder);
          builder.setInsertionPointToStart(rb);
          Value v = stablehlo::SelectOp::create(builder, loc, am, av, bv);
          Value m = stablehlo::OrOp::create(builder, loc, am, bm);
          stablehlo::ReturnOp::create(builder, loc, ValueRange{v, m});
        }
        update = stablehlo::ReshapeOp::create(
            builder, loc, RankedTensorType::get(keptShape, AT.getElementType()),
            reduce.getResult(0));
        updateValueMap = affine::AffineValueMap(
            AffineMap::get(unionMap.getAffineMap().getNumDims(),
                           unionMap.getAffineMap().getNumSymbols(), keptExprs,
                           loc.getContext()),
            unionMap.getOperands());
        updateValueMap.composeSimplifyAndCanonicalize();
        broadcastDims.assign(keptBroadcastDims.begin(),
                             keptBroadcastDims.end());
      }
      maps[update] = updateValueMap;
      UTy = cast<RankedTensorType>(update.getType());
    }
  }
  if (llvm::any_of(broadcastDims, [](int64_t dim) { return dim == -1; })) {
    return nullptr;
  }

  // Align update to the store indices grid; the grid axes act as implicit
  // batch dimensions of the scatter.
  update = stablehlo::BroadcastInDimOp::create(
      builder, loc, UTy.clone(gridShape), update, broadcastDims);

  if (pc.mask) {
    // Broadcast the mask from its IV-space to the update's grid shape. A
    // mask axis over an IV the store does not index or-reduces away first:
    // the update is already known invariant along it (a variant update
    // bailed above), so the store happens when any lane's mask is set.
    Value mask = pc.mask;
    affine::AffineValueMap maskMap = maps.lookup(mask);
    SmallVector<int64_t> maskBroadcastDims;
    SmallVector<int64_t> maskReduceDims;
    for (auto [mi, E] : llvm::enumerate(maskMap.getAffineMap().getResults())) {
      Value maskIV = getIVForExpr(maskMap, E);
      bool onGrid = false;
      for (auto [k, iv] : llvm::enumerate(ivs)) {
        if (iv == maskIV) {
          maskBroadcastDims.push_back((int64_t)k);
          onGrid = true;
          break;
        }
      }
      if (!onGrid)
        maskReduceDims.push_back((int64_t)mi);
    }
    if (!maskReduceDims.empty()) {
      auto boolTy = RankedTensorType::get({}, builder.getI1Type());
      Value initFalse = stablehlo::ConstantOp::create(
          builder, loc, boolTy,
          SplatElementsAttr::get(boolTy, builder.getBoolAttr(false)));
      auto reduce = stablehlo::ReduceOp::create(
          builder, loc, ValueRange{mask}, ValueRange{initFalse},
          builder.getDenseI64ArrayAttr(maskReduceDims));
      Block *body = new Block();
      reduce.getBody().push_back(body);
      body->addArgument(boolTy, loc);
      body->addArgument(boolTy, loc);
      {
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(body);
        Value ored = stablehlo::OrOp::create(builder, loc, body->getArgument(0),
                                             body->getArgument(1));
        stablehlo::ReturnOp::create(builder, loc, ored);
      }
      mask = reduce.getResult(0);
    }
    auto gridTy = cast<RankedTensorType>(update.getType());
    auto maskGridTy =
        RankedTensorType::get(gridTy.getShape(), builder.getI1Type());
    Value broadcastedMask = stablehlo::BroadcastInDimOp::create(
        builder, loc, maskGridTy, mask, maskBroadcastDims);

    // A masked-out lane must not write at all: its index expression is
    // unconstrained and can collide with a live lane's slot, and scatter
    // applies duplicate indices in unspecified order. Send dead lanes out
    // of bounds — the scatter drops those updates.
    auto idxTy = cast<RankedTensorType>(indices.getType());
    Value minusOne = stablehlo::ConstantOp::create(
        builder, loc, idxTy,
        SplatElementsAttr::get(
            idxTy, builder.getIntegerAttr(idxTy.getElementType(), -1)));
    auto idxMaskTy =
        RankedTensorType::get(idxTy.getShape(), builder.getI1Type());
    SmallVector<int64_t> idxMaskDims;
    for (int64_t k = 0, e = (int64_t)gridShape.size(); k < e; ++k)
      idxMaskDims.push_back(k);
    Value idxMask = stablehlo::BroadcastInDimOp::create(
        builder, loc, idxMaskTy, broadcastedMask, idxMaskDims);
    indices = stablehlo::SelectOp::create(builder, loc, idxMask, indices,
                                          minusOne);
  }

  auto Ty = cast<RankedTensorType>(input.getType());
  stablehlo::ScatterOp scatter = stablehlo::ScatterOp::create(
      builder, loc, llvm::ArrayRef<Type>{Ty}, ValueRange{input}, indices,
      ValueRange{update},
      stablehlo::ScatterDimensionNumbersAttr::get(
          loc.getContext(),
          /*updateWindowDims*/ {},
          /*insertedWindowDims*/ scatterDimsToOperandDims,
          /*inputBatchingDims*/ {},
          /*scatterIndicesBatchingDims*/ {},
          /*scatterDimsToOperandDims*/ scatterDimsToOperandDims,
          /*indexVectorDim*/ (int64_t)gridShape.size()),
      /*indicesAreSorted*/ false,
      // Masked-out lanes all share the out-of-bounds index, so uniqueness
      // can only be claimed without a mask.
      /*uniqueIndices*/ !pc.mask);
  Value res = scatter.getResult(0);

  Block *updateBody = new Block();
  scatter.getUpdateComputation().push_back(updateBody);

  auto unrankedTy = RankedTensorType::get(
      {}, cast<RankedTensorType>(update.getType()).getElementType());
  updateBody->addArgument(unrankedTy, loc);
  Value updateInBody = updateBody->addArgument(unrankedTy, loc);

  {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(updateBody);
    stablehlo::ReturnOp::create(builder, loc, updateInBody);
  }

  return res;
}

static LogicalResult tryRaisingForOpToStableHLOUnroll(
    affine::AffineForOp forOp, IRMapping &mapping, OpBuilder &builder,
    llvm::DenseMap<Value, affine::AffineValueMap> &maps, ParallelContext pc) {

  stablehlo::WhileOp whileOp;
  SmallVector<Value> carriedBuffers;
  if (tryRaisingForOpToStableHLOWhile(forOp, mapping, builder, maps, pc,
                                      &whileOp, &carriedBuffers)
          .failed()) {
    return failure();
  }

  unsigned numIterArgs = forOp.getNumRegionIterArgs();
  SmallVector<affine::AffineValueMap> resMaps;
  resMaps.reserve(forOp.getNumResults());
  for (unsigned i = 0, e = forOp.getNumResults(); i < e; ++i)
    resMaps.push_back(maps.lookup(whileOp.getResult(1 + i)));

  IRRewriter rewriter(builder);
  rewriter.setInsertionPoint(whileOp);
  SmallVector<Value> results;
  if (failed(unrollWhileOp(whileOp, rewriter, /*maxNumIterations=*/-1,
                           /*maxOperationThreshold=*/-1, &results))) {
    return failure();
  }

  for (auto [i, forRes] : llvm::enumerate(forOp.getResults())) {
    mapping.map(forRes, results[1 + i]);
    maps[results[1 + i]] = resMaps[i];
  }

  for (auto [i, memref] : llvm::enumerate(carriedBuffers))
    mapping.map(memref, results[1 + numIterArgs + i]);
  return success();
}

// compares the two AffineValueMap and return whether they are aligned
// under their respective dynamic dimensions. that is:
//
//  - they depend on the same induction variables
//  - they have the same number of results
//  - each result depending on the same induction variable as the corresponding
//    result in the other map (up to a constant offset).
//  - up to a permutation can be applied.
//
//  it returns the permutation to apply to a in order to align to b.
//
static std::optional<SmallVector<int64_t>>
memoryEquivalentPermutation(const affine::AffineValueMap &aIn,
                            const affine::AffineValueMap &bIn) {
  // Dims that appear in no result do not affect the laid-out memory (an
  // eliminated unit axis leaves its dim behind); canonicalize them away so
  // equivalent maps compare equal.
  auto canonicalize = [](const affine::AffineValueMap &m) {
    AffineMap map = m.getAffineMap();
    SmallVector<Value> ops(m.getOperands().begin(), m.getOperands().end());
    affine::canonicalizeMapAndOperands(&map, &ops);
    return affine::AffineValueMap(map, ops);
  };
  affine::AffineValueMap a = canonicalize(aIn), b = canonicalize(bIn);
  SmallVector<int64_t> perm(a.getNumResults(), -1);

  auto amap = a.getAffineMap(), bmap = b.getAffineMap();

  if (amap.getNumDims() != bmap.getNumDims() ||
      amap.getNumSymbols() != bmap.getNumSymbols() ||
      amap.getNumResults() != bmap.getNumResults())
    return std::nullopt;

  for (auto EA : amap.getResults()) {
    if (EA.isSymbolicOrConstant())
      return std::nullopt;

    auto apos = getIVPos(a, EA);
    auto aiv = a.getOperand(apos);

    AffineExpr EB = nullptr;
    unsigned bpos;
    for (auto EBB : bmap.getResults()) {
      bpos = getIVPos(b, EBB);

      if (b.getOperand(bpos) == aiv) {
        EB = EBB;
        break;
      }
    }

    if (!EB)
      return std::nullopt;

    perm[bpos] = apos;
    EB = EB.replace(mlir::getAffineDimExpr(bpos, aiv.getContext()),
                    mlir::getAffineDimExpr(apos, aiv.getContext()));
    AffineExpr E = EA - EB;
    if (!E.isSymbolicOrConstant())
      return std::nullopt;
  }

  if (!perm.empty() && llvm::any_of(perm, [](int64_t d) { return d == -1; }))
    return std::nullopt;

  return {perm};
}

// A yielded value carrying fewer lane axes than the loop-carried arg
// stores the same number into every missing lane: broadcast it up to the
// carried layout before matching.
static bool broadcastYieldToCarried(
    Value &yielded, Value carried, OpBuilder &builder,
    llvm::DenseMap<Value, affine::AffineValueMap> &maps, ParallelContext &pc) {
  if (yielded.getType() == carried.getType())
    return true;
  bool ok = true;
  Value a = carried;
  Value b = yielded;
  auto outMap = alignMemoryAccess(a, maps.lookup(carried), b,
                                  maps.lookup(yielded), builder, pc, &ok);
  if (!ok || a.getType() != carried.getType())
    return false;
  maps[b] = outMap;
  yielded = b;
  return true;
}

static LogicalResult tryRaisingForOpToStableHLOWhile(
    affine::AffineForOp forOp, IRMapping &parentMapping, OpBuilder &builder,
    llvm::DenseMap<Value, affine::AffineValueMap> &maps, ParallelContext pc,
    stablehlo::WhileOp *createdWhileOp,
    SmallVectorImpl<Value> *carriedBuffers) {
  IRMapping mapping = parentMapping;

  Value iv = forOp.getInductionVar();

  auto ET = builder.getI64Type();
  auto TT = RankedTensorType::get({}, ET);
  auto wloc = rewriteLocation(forOp.getLoc(), pc.options.strip_llvm_debuginfo);

  auto makeConst = [&](int64_t v) -> Value {
    return stablehlo::ConstantOp::create(
        builder, wloc, TT,
        SplatElementsAttr::get(TT,
                               ArrayRef<Attribute>(IntegerAttr::get(ET, v))));
  };

  Value lb, ub;
  if (forOp.hasConstantBounds()) {
    lb = makeConst(forOp.getConstantLowerBound());
    ub = makeConst(forOp.getConstantUpperBound());
  } else {
    // A while iterates however many times the bounds say at runtime, so the
    // bounds only need to be evaluated as scalars: a lower bound is the max
    // of its results, an upper bound the min.
    auto evalBound = [&](AffineMap map, ValueRange operands,
                         bool isUpper) -> Value {
      Value acc;
      for (AffineExpr expr : map.getResults()) {
        auto [val, avm] = expandAffineExpr(builder, wloc, expr, operands,
                                           mapping, maps, map.getNumDims(), pc);
        if (!val)
          return nullptr;
        auto vt = dyn_cast<RankedTensorType>(val.getType());
        if (!vt || vt.getRank() != 0)
          return nullptr;
        if (vt.getElementType() != ET)
          val = stablehlo::ConvertOp::create(builder, wloc, TT, val);
        acc = !acc ? val
                   : (isUpper ? (Value)stablehlo::MinOp::create(builder, wloc,
                                                                acc, val)
                              : (Value)stablehlo::MaxOp::create(builder, wloc,
                                                                acc, val));
      }
      return acc;
    };
    lb = evalBound(forOp.getLowerBoundMap(), forOp.getLowerBoundOperands(),
                   /*isUpper=*/false);
    ub = evalBound(forOp.getUpperBoundMap(), forOp.getUpperBoundOperands(),
                   /*isUpper=*/true);
    if (!lb || !ub)
      return failure();
  }
  Value step = makeConst(forOp.getStepAsInt());

  Block *entryBlock = getRaisedEntryBlock(forOp);

  Block *cond = new Block(), *body = new Block();
  Value ivInCond = cond->addArgument(
      TT, rewriteLocation(iv.getLoc(), pc.options.strip_llvm_debuginfo));
  Value ivInBody = body->addArgument(
      TT, rewriteLocation(iv.getLoc(), pc.options.strip_llvm_debuginfo));

  SmallVector<Value> inits;
  inits.push_back(lb);

  for (auto [init, iterArg] :
       llvm::zip(forOp.getInits(), forOp.getRegionIterArgs())) {
    auto TT = pc.getTensorType(init.getType());
    cond->addArgument(
        TT, rewriteLocation(iterArg.getLoc(), pc.options.strip_llvm_debuginfo));
    Value iterArgInBody = body->addArgument(
        TT, rewriteLocation(iterArg.getLoc(), pc.options.strip_llvm_debuginfo));
    auto tensorInit = mapping.lookupOrNull(init);
    if (!tensorInit || !maps.count(tensorInit))
      return failure();
    auto broadcastInit =
        pc.getBroadcast(builder, maps.lookup(tensorInit), tensorInit);
    if (!broadcastInit)
      return failure();
    inits.push_back(broadcastInit->v);
    mapping.map(iterArg, iterArgInBody);
    maps[iterArgInBody] = broadcastInit->avm;
  }

  // Every buffer written in the body must be loop-carried, or its mapping
  // after the loop would point into the body. That is the entry block's
  // arguments plus any other outside-defined buffer the body touches
  // (e.g. a raised memref.alloca).
  SmallVector<Value> buffers(entryBlock->getArguments().begin(),
                             entryBlock->getArguments().end());
  {
    llvm::SmallPtrSet<Value, 8> seen(buffers.begin(), buffers.end());
    forOp.getBody()->walk([&](Operation *innerOp) {
      for (Value v : innerOp->getOperands())
        if (isa<MemRefType>(v.getType()) && mapping.contains(v) &&
            !forOp->isAncestor(v.getParentRegion()->getParentOp()) &&
            seen.insert(v).second)
          buffers.push_back(v);
    });
  }

  if (carriedBuffers)
    carriedBuffers->assign(buffers.begin(), buffers.end());

  for (auto memref : buffers) {
    Value mappedMemref = mapping.lookup(memref);
    inits.push_back(mappedMemref);

    cond->addArgument(mappedMemref.getType(),
                      rewriteLocation(mappedMemref.getLoc(),
                                      pc.options.strip_llvm_debuginfo));
    Value memrefInBody =
        body->addArgument(mappedMemref.getType(),
                          rewriteLocation(mappedMemref.getLoc(),
                                          pc.options.strip_llvm_debuginfo));
    mapping.map(memref, memrefInBody);
  }

  auto whileOp = stablehlo::WhileOp::create(
      builder, rewriteLocation(forOp.getLoc(), pc.options.strip_llvm_debuginfo),
      inits);

  whileOp->getRegion(0).push_back(cond);
  whileOp->getRegion(1).push_back(body);

  // A loop peeled off a parallel axis stays a parallel axis: iterations are
  // independent, which downstream passes may use without reanalyzing.
  if (forOp->hasAttr("enzymexla.parallel"))
    whileOp->setAttr("enzymexla.parallel", builder.getUnitAttr());

  if (createdWhileOp)
    *createdWhileOp = whileOp;

  {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(cond);
    Value cond = stablehlo::CompareOp::create(
        builder,
        rewriteLocation(forOp.getLoc(), pc.options.strip_llvm_debuginfo),
        ivInCond, ub, stablehlo::ComparisonDirection::LT);
    stablehlo::ReturnOp::create(
        builder,
        rewriteLocation(forOp.getLoc(), pc.options.strip_llvm_debuginfo), cond);
  }
  {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(body);

    mapping.map(iv, ivInBody);
    maps[ivInBody] =
        affine::AffineValueMap(AffineMap::get(forOp->getContext()), {});

    for (auto &innerOp : forOp.getBody()->without_terminator()) {
      if (tryRaisingOpToStableHLO(&innerOp, mapping, builder, maps, pc)
              .failed())
        return failure();
    }

    Value newIvInBody = stablehlo::AddOp::create(
        builder,
        rewriteLocation(forOp.getLoc(), pc.options.strip_llvm_debuginfo),
        ivInBody, step);

    SmallVector<Value> loopCarried = {newIvInBody};
    for (auto [iterArg, yieldedIterArgs] :
         llvm::zip(forOp.getRegionIterArgs(),
                   forOp.getBody()->getTerminator()->getOperands())) {

      Value raisedYieldedIterArg = mapping.lookupOrNull(yieldedIterArgs);
      Value raisedIterArg = mapping.lookupOrNull(iterArg);
      if (!raisedYieldedIterArg || !raisedIterArg)
        return failure();

      if (!maps.count(raisedYieldedIterArg) || !maps.count(raisedIterArg))
        return failure();
      if (!broadcastYieldToCarried(raisedYieldedIterArg, raisedIterArg,
                                   builder, maps, pc))
        return failure();
      auto perm = memoryEquivalentPermutation(maps.lookup(raisedYieldedIterArg),
                                              maps.lookup(raisedIterArg));

      if (!perm.has_value()) {
        // Leave the abandoned while in place: raised values in `maps` may
        // reference its body, and the failed function is discarded whole.
        return failure();
      }

      if (!std::is_sorted(perm->begin(), perm->end()))
        raisedYieldedIterArg = stablehlo::TransposeOp::create(
            builder, raisedYieldedIterArg.getLoc(), raisedYieldedIterArg,
            *perm);

      loopCarried.push_back(raisedYieldedIterArg);
    }

    for (auto memref : buffers)
      loopCarried.push_back(mapping.lookup(memref));
    stablehlo::ReturnOp::create(
        builder,
        rewriteLocation(forOp.getLoc(), pc.options.strip_llvm_debuginfo),
        loopCarried);
  }

  for (auto [i, memref] : llvm::enumerate(buffers))
    mapping.map(memref,
                whileOp.getResult(i + 1 + forOp.getNumRegionIterArgs()));
  for (auto [forRes, forIterArg, whileRes] :
       llvm::zip(forOp.getResults(), forOp.getRegionIterArgs(),
                 llvm::drop_begin(whileOp.getResults()))) {
    mapping.map(forRes, whileRes);
    maps[whileRes] = maps.lookup(mapping.lookup(forIterArg));
  }

  parentMapping = mapping;
  return success();
}

// An scf.for whose bounds and step raise as uniform scalars iterates as a
// stablehlo.while, exactly like a runtime-extent affine loop: the induction
// variable is a rank-0 tensor of the loop's integer type, iter args are
// loop-carried (broadcast like the affine path), and every buffer the body
// writes is threaded through the loop.
static LogicalResult tryRaisingSCFForOpToStableHLOWhile(
    scf::ForOp forOp, IRMapping &parentMapping, OpBuilder &builder,
    llvm::DenseMap<Value, affine::AffineValueMap> &maps, ParallelContext pc) {
  IRMapping mapping = parentMapping;

  Value iv = forOp.getInductionVar();
  auto wloc = rewriteLocation(forOp.getLoc(), pc.options.strip_llvm_debuginfo);

  Value lb = mapping.lookupOrNull(forOp.getLowerBound());
  Value ub = mapping.lookupOrNull(forOp.getUpperBound());
  Value step = mapping.lookupOrNull(forOp.getStep());
  if (!lb || !ub || !step)
    return failure();
  auto scalarTy = dyn_cast<RankedTensorType>(lb.getType());
  if (!scalarTy || scalarTy.getRank() != 0)
    return failure();
  auto toCounterType = [&](Value v) -> Value {
    auto vt = dyn_cast<RankedTensorType>(v.getType());
    if (!vt || vt.getRank() != 0)
      return nullptr;
    if (vt != scalarTy)
      v = stablehlo::ConvertOp::create(builder, wloc, scalarTy, v);
    return v;
  };
  ub = toCounterType(ub);
  step = toCounterType(step);
  if (!ub || !step)
    return failure();

  Block *entryBlock = getRaisedEntryBlock(forOp);

  Block *cond = new Block(), *body = new Block();
  Value ivInCond = cond->addArgument(
      scalarTy, rewriteLocation(iv.getLoc(), pc.options.strip_llvm_debuginfo));
  Value ivInBody = body->addArgument(
      scalarTy, rewriteLocation(iv.getLoc(), pc.options.strip_llvm_debuginfo));

  SmallVector<Value> inits;
  inits.push_back(lb);

  for (auto [init, iterArg] :
       llvm::zip(forOp.getInits(), forOp.getRegionIterArgs())) {
    auto TT = pc.getTensorType(init.getType());
    cond->addArgument(
        TT, rewriteLocation(iterArg.getLoc(), pc.options.strip_llvm_debuginfo));
    Value iterArgInBody = body->addArgument(
        TT, rewriteLocation(iterArg.getLoc(), pc.options.strip_llvm_debuginfo));
    auto tensorInit = mapping.lookupOrNull(init);
    if (!tensorInit || !maps.count(tensorInit))
      return failure();
    auto broadcastInit =
        pc.getBroadcast(builder, maps.lookup(tensorInit), tensorInit);
    if (!broadcastInit)
      return failure();
    inits.push_back(broadcastInit->v);
    mapping.map(iterArg, iterArgInBody);
    maps[iterArgInBody] = broadcastInit->avm;
  }

  SmallVector<Value> buffers(entryBlock->getArguments().begin(),
                             entryBlock->getArguments().end());
  {
    llvm::SmallPtrSet<Value, 8> seen(buffers.begin(), buffers.end());
    forOp.getBody()->walk([&](Operation *innerOp) {
      for (Value v : innerOp->getOperands())
        if (isa<MemRefType>(v.getType()) && mapping.contains(v) &&
            !forOp->isAncestor(v.getParentRegion()->getParentOp()) &&
            seen.insert(v).second)
          buffers.push_back(v);
    });
  }

  for (auto memref : buffers) {
    Value mappedMemref = mapping.lookup(memref);
    inits.push_back(mappedMemref);
    cond->addArgument(mappedMemref.getType(),
                      rewriteLocation(mappedMemref.getLoc(),
                                      pc.options.strip_llvm_debuginfo));
    Value memrefInBody =
        body->addArgument(mappedMemref.getType(),
                          rewriteLocation(mappedMemref.getLoc(),
                                          pc.options.strip_llvm_debuginfo));
    mapping.map(memref, memrefInBody);
  }

  auto whileOp = stablehlo::WhileOp::create(builder, wloc, inits);
  whileOp->getRegion(0).push_back(cond);
  whileOp->getRegion(1).push_back(body);

  {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(cond);
    Value condVal = stablehlo::CompareOp::create(
        builder, wloc, ivInCond, ub, stablehlo::ComparisonDirection::LT);
    stablehlo::ReturnOp::create(builder, wloc, condVal);
  }
  {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(body);

    mapping.map(iv, ivInBody);
    maps[ivInBody] =
        affine::AffineValueMap(AffineMap::get(forOp->getContext()), {});

    for (auto &innerOp : forOp.getBody()->without_terminator()) {
      if (tryRaisingOpToStableHLO(&innerOp, mapping, builder, maps, pc)
              .failed())
        return failure();
    }

    Value newIvInBody = stablehlo::AddOp::create(builder, wloc, ivInBody, step);

    SmallVector<Value> loopCarried = {newIvInBody};
    for (auto [iterArg, yielded] :
         llvm::zip(forOp.getRegionIterArgs(),
                   forOp.getBody()->getTerminator()->getOperands())) {
      Value raisedYielded = mapping.lookupOrNull(yielded);
      Value raisedIterArg = mapping.lookup(iterArg);
      if (!raisedYielded)
        return failure();
      if (!maps.count(raisedYielded) || !maps.count(raisedIterArg))
        return failure();
      if (!broadcastYieldToCarried(raisedYielded, raisedIterArg, builder,
                                   maps, pc))
        return failure();
      auto perm = memoryEquivalentPermutation(maps.lookup(raisedYielded),
                                              maps.lookup(raisedIterArg));
      if (!perm.has_value()) {
        // Leave the abandoned while in place: raised values in `maps` may
        // reference its body, and the failed function is discarded whole.
        return failure();
      }
      if (!std::is_sorted(perm->begin(), perm->end()))
        raisedYielded = stablehlo::TransposeOp::create(
            builder, raisedYielded.getLoc(), raisedYielded, *perm);
      loopCarried.push_back(raisedYielded);
    }

    for (auto memref : buffers)
      loopCarried.push_back(mapping.lookup(memref));
    stablehlo::ReturnOp::create(builder, wloc, loopCarried);
  }

  for (auto [i, memref] : llvm::enumerate(buffers))
    mapping.map(memref,
                whileOp.getResult(i + 1 + forOp.getNumRegionIterArgs()));
  for (auto [forRes, forIterArg, whileRes] :
       llvm::zip(forOp.getResults(), forOp.getRegionIterArgs(),
                 llvm::drop_begin(whileOp.getResults()))) {
    mapping.map(forRes, whileRes);
    maps[whileRes] = maps.lookup(mapping.lookup(forIterArg));
  }

  parentMapping = mapping;
  return success();
}

// An scf.for whose bounds vary per lane (a thread-stride remainder loop with
// tid-dependent extent) iterates a scalar trip counter to the maximum lane
// trip count, with a per-lane active mask lb + k*step < ub applied to the
// body: masked stores skip finished lanes and iter args keep their value.
static LogicalResult raiseLoopToMaskedWhile(
    Operation *loopOp, Value iv, Block *loopBody, ValueRange loopInits,
    ValueRange regionIterArgs, Value lb, Value ub, Value step,
    IRMapping &parentMapping, IRMapping &mapping, OpBuilder &builder,
    llvm::DenseMap<Value, affine::AffineValueMap> &maps, ParallelContext pc);

static LogicalResult tryRaisingSCFForOpToMaskedWhile(
    scf::ForOp forOp, IRMapping &parentMapping, OpBuilder &builder,
    llvm::DenseMap<Value, affine::AffineValueMap> &maps, ParallelContext pc) {
  IRMapping mapping = parentMapping;

  // scf.for requires a positive step, so the trip-count math below is sound
  // for dynamic steps too (a thread-stride loop steps by the block size).

  Value lb = mapping.lookupOrNull(forOp.getLowerBound());
  Value ub = mapping.lookupOrNull(forOp.getUpperBound());
  Value step = mapping.lookupOrNull(forOp.getStep());
  if (!lb || !ub || !step)
    return failure();
  for (Value v : {lb, ub, step})
    if (!isa<RankedTensorType>(v.getType()) ||
        !cast<RankedTensorType>(v.getType()).getElementType().isInteger())
      return failure();
  return raiseLoopToMaskedWhile(forOp, forOp.getInductionVar(), forOp.getBody(),
                                forOp.getInits(), forOp.getRegionIterArgs(), lb,
                                ub, step, parentMapping, mapping, builder, maps,
                                pc);
}

static LogicalResult raiseLoopToMaskedWhile(
    Operation *loopOp, Value iv, Block *loopBody, ValueRange loopInits,
    ValueRange regionIterArgs, Value lb, Value ub, Value step,
    IRMapping &parentMapping, IRMapping &mapping, OpBuilder &builder,
    llvm::DenseMap<Value, affine::AffineValueMap> &maps, ParallelContext pc) {
  auto wloc =
      rewriteLocation(loopOp->getLoc(), pc.options.strip_llvm_debuginfo);

  // One common lane space for the bounds, referenced from whichever bound
  // already carries the richest lane shape (lb may be uniform while ub is
  // per-lane).
  if (!maps.count(lb) || !maps.count(ub) || !maps.count(step))
    return failure();
  Value bnds[3] = {lb, ub, step};
  unsigned refIdx = 0;
  int64_t bestRank = -1;
  for (unsigned i = 0; i < 3; ++i) {
    int64_t r = cast<RankedTensorType>(bnds[i].getType()).getRank();
    if (r > bestRank) {
      bestRank = r;
      refIdx = i;
    }
  }
  Value vals[] = {bnds[(refIdx + 1) % 3], bnds[(refIdx + 2) % 3]};
  affine::AffineValueMap dsts[] = {maps.lookup(vals[0]), maps.lookup(vals[1])};
  bool laneOk = true;
  affine::AffineValueMap laneMap = alignMemoryAccess(
      bnds[refIdx], maps.lookup(bnds[refIdx]), vals, dsts, builder, pc,
      &laneOk);
  if (!laneOk)
    return failure();
  bnds[(refIdx + 1) % 3] = vals[0];
  bnds[(refIdx + 2) % 3] = vals[1];
  lb = bnds[0];
  ub = bnds[1];
  step = bnds[2];
  auto laneTy = cast<RankedTensorType>(bnds[refIdx].getType());
  if (laneTy.getRank() == 0)
    return failure(); // the uniform path handles this
  auto ET = laneTy.getElementType();

  auto splat = [&](int64_t v) -> Value {
    return stablehlo::ConstantOp::create(
        builder, wloc, laneTy,
        SplatElementsAttr::get(laneTy, IntegerAttr::get(ET, v)));
  };

  // Per-lane trip count cdiv(max(ub - lb, 0), step), then its maximum.
  Value diff = stablehlo::SubtractOp::create(builder, wloc, ub, lb);
  diff = stablehlo::MaxOp::create(builder, wloc, diff, splat(0));
  Value stepM1 = stablehlo::SubtractOp::create(builder, wloc, step, splat(1));
  Value iters = stablehlo::DivOp::create(
      builder, wloc, stablehlo::AddOp::create(builder, wloc, diff, stepM1),
      step);
  auto scalarTy = RankedTensorType::get({}, ET);
  Value initMin = stablehlo::ConstantOp::create(
      builder, wloc, scalarTy,
      SplatElementsAttr::get(scalarTy, IntegerAttr::get(ET, 0)));
  SmallVector<int64_t> allDims(laneTy.getRank());
  std::iota(allDims.begin(), allDims.end(), 0);
  auto maxReduce = stablehlo::ReduceOp::create(
      builder, wloc, ValueRange{iters}, ValueRange{initMin},
      builder.getDenseI64ArrayAttr(allDims));
  {
    Block *rbody = new Block();
    maxReduce.getBody().push_back(rbody);
    rbody->addArgument(scalarTy, wloc);
    rbody->addArgument(scalarTy, wloc);
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(rbody);
    Value m = stablehlo::MaxOp::create(builder, wloc, rbody->getArgument(0),
                                       rbody->getArgument(1));
    stablehlo::ReturnOp::create(builder, wloc, m);
  }
  Value kMax = maxReduce.getResult(0);

  Block *entryBlock = getRaisedEntryBlock(loopOp);

  Block *cond = new Block(), *body = new Block();
  Value kInCond = cond->addArgument(scalarTy, wloc);
  Value kInBody = body->addArgument(scalarTy, wloc);

  SmallVector<Value> inits;
  Value kZero = stablehlo::ConstantOp::create(
      builder, wloc, scalarTy,
      SplatElementsAttr::get(scalarTy, IntegerAttr::get(ET, 0)));
  inits.push_back(kZero);

  SmallVector<affine::AffineValueMap> iterArgMaps;
  for (auto [init, iterArg] : llvm::zip(loopInits, regionIterArgs)) {
    auto TT = pc.getTensorType(init.getType());
    cond->addArgument(
        TT, rewriteLocation(iterArg.getLoc(), pc.options.strip_llvm_debuginfo));
    Value iterArgInBody = body->addArgument(
        TT, rewriteLocation(iterArg.getLoc(), pc.options.strip_llvm_debuginfo));
    auto tensorInit = mapping.lookupOrNull(init);
    if (!tensorInit || !maps.count(tensorInit))
      return failure();
    auto broadcastInit =
        pc.getBroadcast(builder, maps.lookup(tensorInit), tensorInit);
    if (!broadcastInit)
      return failure();
    inits.push_back(broadcastInit->v);
    mapping.map(iterArg, iterArgInBody);
    maps[iterArgInBody] = broadcastInit->avm;
    iterArgMaps.push_back(broadcastInit->avm);
  }

  SmallVector<Value> buffers(entryBlock->getArguments().begin(),
                             entryBlock->getArguments().end());
  {
    llvm::SmallPtrSet<Value, 8> seen(buffers.begin(), buffers.end());
    loopBody->walk([&](Operation *innerOp) {
      for (Value v : innerOp->getOperands())
        if (isa<MemRefType>(v.getType()) && mapping.contains(v) &&
            !loopOp->isAncestor(v.getParentRegion()->getParentOp()) &&
            seen.insert(v).second)
          buffers.push_back(v);
    });
  }
  for (auto memref : buffers) {
    Value mappedMemref = mapping.lookup(memref);
    inits.push_back(mappedMemref);
    cond->addArgument(mappedMemref.getType(),
                      rewriteLocation(mappedMemref.getLoc(),
                                      pc.options.strip_llvm_debuginfo));
    Value memrefInBody =
        body->addArgument(mappedMemref.getType(),
                          rewriteLocation(mappedMemref.getLoc(),
                                          pc.options.strip_llvm_debuginfo));
    mapping.map(memref, memrefInBody);
  }

  auto whileOp = stablehlo::WhileOp::create(builder, wloc, inits);
  whileOp->getRegion(0).push_back(cond);
  whileOp->getRegion(1).push_back(body);

  {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(cond);
    Value c = stablehlo::CompareOp::create(builder, wloc, kInCond, kMax,
                                           stablehlo::ComparisonDirection::LT);
    stablehlo::ReturnOp::create(builder, wloc, c);
  }
  {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(body);

    Value kLane = stablehlo::BroadcastInDimOp::create(
        builder, wloc, laneTy, kInBody, ArrayRef<int64_t>{});
    Value ivT = stablehlo::AddOp::create(
        builder, wloc, lb,
        stablehlo::MulOp::create(builder, wloc, kLane, step));
    mapping.map(iv, ivT);
    maps[ivT] = laneMap;

    Value active = stablehlo::CompareOp::create(
        builder, wloc, ivT, ub, stablehlo::ComparisonDirection::LT);
    maps[active] = laneMap;

    ParallelContext bodyPc(pc.options);
    bodyPc.ranges = pc.ranges;
    bodyPc.ivs = pc.ivs;
    Value mask = active;
    if (pc.mask) {
      Value pm = pc.mask;
      affine::AffineValueMap pmMap = maps.lookup(pm);
      bool mergeOk = true;
      affine::AffineValueMap merged =
          alignMemoryAccess(pm, pmMap, mask, laneMap, builder, pc, &mergeOk);
      if (!mergeOk)
        return failure();
      mask = stablehlo::AndOp::create(builder, wloc, pm, mask);
      maps[mask] = merged;
    }
    bodyPc.mask = mask;

    for (auto &innerOp : loopBody->without_terminator()) {
      if (tryRaisingOpToStableHLO(&innerOp, mapping, builder, maps, bodyPc)
              .failed())
        return failure();
    }

    Value kNext = stablehlo::AddOp::create(
        builder, wloc, kInBody,
        stablehlo::ConstantOp::create(
            builder, wloc, scalarTy,
            SplatElementsAttr::get(scalarTy, IntegerAttr::get(ET, 1))));

    SmallVector<Value> loopCarried = {kNext};
    for (auto [i, pair] : llvm::enumerate(llvm::zip(
             regionIterArgs, loopBody->getTerminator()->getOperands()))) {
      auto [iterArg, yielded] = pair;
      Value raisedYielded = mapping.lookupOrNull(yielded);
      Value raisedIterArg = mapping.lookup(iterArg);
      if (!raisedYielded)
        return failure();
      // Finished lanes keep their value.
      Value m = mask;
      if (!maps.count(raisedYielded) || !maps.count(raisedIterArg))
        return failure();
      affine::AffineValueMap mMap = maps.lookup(m);
      Value selVals[] = {raisedYielded, raisedIterArg};
      affine::AffineValueMap selMaps[] = {maps.lookup(raisedYielded),
                                          maps.lookup(raisedIterArg)};
      bool selOk = true;
      affine::AffineValueMap outMap =
          alignMemoryAccess(m, mMap, selVals, selMaps, builder, pc, &selOk);
      if (!selOk)
        return failure();
      Value sel =
          stablehlo::SelectOp::create(builder, wloc, m, selVals[0], selVals[1]);
      // The aligned select may order the axes differently than the carried
      // value; permute it back into the iter arg's layout.
      auto perm =
          memoryEquivalentPermutation(outMap, maps.lookup(raisedIterArg));
      if (perm.has_value() && !std::is_sorted(perm->begin(), perm->end()))
        sel = stablehlo::TransposeOp::create(builder, wloc, sel, *perm);
      if (sel.getType() != raisedIterArg.getType())
        return failure();
      loopCarried.push_back(sel);
    }

    for (auto memref : buffers)
      loopCarried.push_back(mapping.lookup(memref));
    stablehlo::ReturnOp::create(builder, wloc, loopCarried);
  }

  for (auto [i, memref] : llvm::enumerate(buffers))
    mapping.map(memref, whileOp.getResult(i + 1 + regionIterArgs.size()));
  for (auto [i, loopRes] : llvm::enumerate(loopOp->getResults())) {
    Value whileRes = whileOp.getResult(1 + i);
    mapping.map(loopRes, whileRes);
    maps[whileRes] = iterArgMaps[i];
  }

  parentMapping = mapping;
  return success();
}

// An affine.for whose bounds only evaluate per lane takes the same masked
// path: bounds expand elementwise (max over lower results, min over upper),
// and the loop iterates to the maximum lane trip count.
static LogicalResult tryRaisingAffineForOpToMaskedWhile(
    affine::AffineForOp forOp, IRMapping &parentMapping, OpBuilder &builder,
    llvm::DenseMap<Value, affine::AffineValueMap> &maps, ParallelContext pc) {
  if (forOp.getStepAsInt() <= 0)
    return failure();
  IRMapping mapping = parentMapping;
  auto wloc = rewriteLocation(forOp.getLoc(), pc.options.strip_llvm_debuginfo);

  auto evalBound = [&](AffineMap map, ValueRange operands,
                       bool isUpper) -> Value {
    Value acc;
    affine::AffineValueMap accMap;
    for (AffineExpr expr : map.getResults()) {
      auto [val, avm] = expandAffineExpr(builder, wloc, expr, operands, mapping,
                                         maps, map.getNumDims(), pc);
      if (!val)
        return nullptr;
      if (!maps.count(val))
        maps[val] = avm;
      else
        avm = maps.lookup(val);
      if (!acc) {
        acc = val;
        accMap = avm;
        continue;
      }
      bool alignOk = true;
      affine::AffineValueMap merged =
          alignMemoryAccess(acc, accMap, val, avm, builder, pc, &alignOk);
      if (!alignOk)
        return nullptr;
      acc = isUpper ? (Value)stablehlo::MinOp::create(builder, wloc, acc, val)
                    : (Value)stablehlo::MaxOp::create(builder, wloc, acc, val);
      accMap = merged;
    }
    if (acc)
      maps[acc] = accMap;
    return acc;
  };
  Value lb = evalBound(forOp.getLowerBoundMap(), forOp.getLowerBoundOperands(),
                       /*isUpper=*/false);
  Value ub = evalBound(forOp.getUpperBoundMap(), forOp.getUpperBoundOperands(),
                       /*isUpper=*/true);
  if (!lb || !ub)
    return failure();
  auto ET = builder.getI64Type();
  auto TT = RankedTensorType::get({}, ET);
  Value step = stablehlo::ConstantOp::create(
      builder, wloc, TT,
      SplatElementsAttr::get(TT, IntegerAttr::get(ET, forOp.getStepAsInt())));
  maps[step] = affine::AffineValueMap(AffineMap::get(forOp->getContext()), {});
  return raiseLoopToMaskedWhile(forOp, forOp.getInductionVar(), forOp.getBody(),
                                forOp.getInits(), forOp.getRegionIterArgs(), lb,
                                ub, step, parentMapping, mapping, builder, maps,
                                pc);
}

// A general scf.while raises by rotation: its before region runs once
// (peeled), producing the loop condition and carried values; the
// stablehlo.while then carries (cond, args, buffers) and its body runs the
// do region followed by the before region again. Only a uniform (rank-0)
// condition is supported.
static LogicalResult tryRaisingSCFWhileOpToStableHLO(
    scf::WhileOp whileOp, IRMapping &parentMapping, OpBuilder &builder,
    llvm::DenseMap<Value, affine::AffineValueMap> &maps, ParallelContext pc) {
  IRMapping mapping = parentMapping;
  auto wloc =
      rewriteLocation(whileOp.getLoc(), pc.options.strip_llvm_debuginfo);
  Block *before = whileOp.getBeforeBody();
  Block *after = whileOp.getAfterBody();
  auto condOp = cast<scf::ConditionOp>(before->getTerminator());
  auto yieldOp = cast<scf::YieldOp>(after->getTerminator());

  for (auto [arg, init] :
       llvm::zip(before->getArguments(), whileOp.getInits())) {
    Value m = mapping.lookupOrNull(init);
    if (!m || !maps.count(m))
      return failure();
    mapping.map(arg, m);
    maps[m] = maps.lookup(m);
  }
  for (auto &op : before->without_terminator())
    if (tryRaisingOpToStableHLO(&op, mapping, builder, maps, pc).failed())
      return failure();
  Value cond0 = mapping.lookupOrNull(condOp.getCondition());
  if (!cond0)
    return failure();
  auto condTy = dyn_cast<RankedTensorType>(cond0.getType());
  if (!condTy || condTy.getRank() != 0)
    return failure();

  SmallVector<Value> carriedInit{cond0};
  SmallVector<affine::AffineValueMap> argMaps;
  for (Value a : condOp.getArgs()) {
    Value m = mapping.lookupOrNull(a);
    if (!m || !maps.count(m))
      return failure();
    carriedInit.push_back(m);
    argMaps.push_back(maps.lookup(m));
  }

  Block *entryBlock = getRaisedEntryBlock(whileOp);
  SmallVector<Value> buffers(entryBlock->getArguments().begin(),
                             entryBlock->getArguments().end());
  {
    llvm::SmallPtrSet<Value, 8> seen(buffers.begin(), buffers.end());
    auto collect = [&](Block *b) {
      b->walk([&](Operation *innerOp) {
        for (Value v : innerOp->getOperands())
          if (isa<MemRefType>(v.getType()) && mapping.contains(v) &&
              !whileOp->isAncestor(v.getParentRegion()->getParentOp()) &&
              seen.insert(v).second)
            buffers.push_back(v);
      });
    };
    collect(before);
    collect(after);
  }
  for (auto memref : buffers)
    carriedInit.push_back(mapping.lookup(memref));

  Block *cond = new Block(), *body = new Block();
  for (Value v : carriedInit) {
    cond->addArgument(v.getType(), wloc);
    body->addArgument(v.getType(), wloc);
  }

  unsigned nargs = condOp.getArgs().size();
  for (auto [i, arg] : llvm::enumerate(after->getArguments())) {
    Value bodyArg = body->getArgument(1 + i);
    mapping.map(arg, bodyArg);
    maps[bodyArg] = argMaps[i];
  }
  for (auto [i, memref] : llvm::enumerate(buffers))
    mapping.map(memref, body->getArgument(1 + nargs + i));

  auto newWhile = stablehlo::WhileOp::create(builder, wloc, carriedInit);
  newWhile->getRegion(0).push_back(cond);
  newWhile->getRegion(1).push_back(body);

  {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(cond);
    stablehlo::ReturnOp::create(builder, wloc, cond->getArgument(0));
  }
  {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(body);
    for (auto &op : after->without_terminator())
      if (tryRaisingOpToStableHLO(&op, mapping, builder, maps, pc).failed())
        return failure();
    // The do region yields the next before-region arguments.
    for (auto [arg, y] :
         llvm::zip(before->getArguments(), yieldOp.getOperands())) {
      Value m = mapping.lookupOrNull(y);
      if (!m)
        return failure();
      mapping.map(arg, m);
    }
    for (auto &op : before->without_terminator())
      if (tryRaisingOpToStableHLO(&op, mapping, builder, maps, pc).failed())
        return failure();
    Value nextCond = mapping.lookupOrNull(condOp.getCondition());
    if (!nextCond || nextCond.getType() != cond0.getType())
      return failure();
    SmallVector<Value> carried{nextCond};
    for (auto [i, a] : llvm::enumerate(condOp.getArgs())) {
      Value m = mapping.lookupOrNull(a);
      if (!m || m.getType() != carriedInit[1 + i].getType())
        return failure();
      carried.push_back(m);
    }
    for (auto memref : buffers)
      carried.push_back(mapping.lookup(memref));
    stablehlo::ReturnOp::create(builder, wloc, carried);
  }

  for (auto [i, res] : llvm::enumerate(whileOp.getResults())) {
    Value whileRes = newWhile.getResult(1 + i);
    mapping.map(res, whileRes);
    maps[whileRes] = argMaps[i];
  }
  for (auto [i, memref] : llvm::enumerate(buffers))
    mapping.map(memref, newWhile.getResult(1 + nargs + i));

  parentMapping = mapping;
  return success();
}

template <class T> static SmallVector<BlockArgument, 6> getIVs(T op);
template <> SmallVector<BlockArgument, 6> getIVs(affine::AffineParallelOp op) {
  return {op.getIVs().begin(), op.getIVs().end()};
}
template <> SmallVector<BlockArgument, 6> getIVs(affine::AffineForOp op) {
  return {op.getInductionVar()};
}

template <class T>
static LogicalResult tryRaisingParallelOpToStableHLO(
    T parallelOp, IRMapping &parentMapping, OpBuilder &builder,
    llvm::DenseMap<Value, affine::AffineValueMap> &maps, ParallelContext pc) {
  IRMapping mapping = parentMapping;

  for (auto iv : getIVs(parallelOp)) {
    auto range = getIVRange(iv);
    if (!range.has_value()) {
      return parallelOp.getOperation()->emitError(
          "parallel loop has non-constant bounds, which is not currently "
          "supported");
    }
    emitIVToStableHLO(builder, iv, *range, mapping, maps,
                      pc.options.strip_llvm_debuginfo);
  }

  auto newPc = pc.add(parallelOp);
  if (!newPc)
    return failure();

  SmallVector<Value> iter_inputs;
  SmallVector<BlockArgument> iters;
  if (auto forOp = dyn_cast<affine::AffineForOp>(parallelOp.getOperation())) {
    for (auto &&[inp, arg] :
         llvm::zip_equal(forOp.getInits(), forOp.getRegionIterArgs())) {
      iters.push_back(arg);
      iter_inputs.push_back(inp);
    }
  }

  for (auto &innerOp : parallelOp.getBody()->without_terminator()) {
    ssize_t reduced_idx = -1;
    ssize_t op_idx = -1;
    for (auto &&[j, operand] : llvm::enumerate(innerOp.getOperands())) {
      for (auto &&[i, iter] : llvm::enumerate(iters)) {
        if (iter == operand) {
          reduced_idx = i;
          op_idx = j;
          break;
        }
      }
    }

    if (reduced_idx != -1) {
      Value reduced_val = innerOp.getOperand(1 - op_idx);
      Value init_val = iter_inputs[reduced_idx];

      Value reduce_broadcasted = mapping.lookup(reduced_val);
      if (!maps.count(reduce_broadcasted))
        return failure();
      auto reduce_map = maps.lookup(reduce_broadcasted);

      auto forOp = cast<affine::AffineForOp>(
          iters[reduced_idx].getOwner()->getParentOp());

      Value idx_broadcasted = mapping.lookup(forOp.getInductionVar());
      if (!maps.count(idx_broadcasted))
        return failure();
      auto idx_map = maps.lookup(idx_broadcasted);

      Value dsts[] = {idx_broadcasted, mapping.lookup(init_val)};
      affine::AffineValueMap submaps[] = {idx_map, maps.lookup(dsts[1])};

      bool alignOk = true;
      auto outputMap = alignMemoryAccess(reduce_broadcasted, reduce_map, dsts,
                                         submaps, builder, *newPc, &alignOk);
      if (!alignOk)
        return failure();

      ssize_t idx_to_reduce = -1;
      for (auto &&[i, expr] :
           llvm::enumerate(outputMap.getAffineMap().getResults())) {
        auto dim = cast<AffineDimExpr>(expr);
        if (outputMap.getOperands()[dim.getPosition()] ==
            forOp.getInductionVar()) {
          assert(idx_to_reduce == -1);
          idx_to_reduce = i;
        }
      }
      assert(idx_to_reduce != -1);

      auto unrankedTensorType = RankedTensorType::get(
          {}, cast<RankedTensorType>(reduce_broadcasted.getType())
                  .getElementType());
      Value init_values[1] = {stablehlo::ConstantOp::create(
          builder,
          rewriteLocation(innerOp.getLoc(), pc.options.strip_llvm_debuginfo),
          builder.getZeroAttr(unrankedTensorType))};

      auto shape =
          cast<RankedTensorType>(reduce_broadcasted.getType()).getShape();
      SmallVector<int64_t> win_dim(shape.size(), 1);
      win_dim[idx_to_reduce] = shape[idx_to_reduce];

      SmallVector<int64_t> win_strides(shape.size(), 1);
      SmallVector<int64_t> win_dialations(shape.size(), 1);
      SmallVector<int64_t> base_dialations(shape.size(), 1);
      SmallVector<int64_t> padding_dialations(2 * shape.size(), 0);
      padding_dialations[2 * idx_to_reduce] = shape[idx_to_reduce] - 1;

      int64_t padding_shape[2] = {(int64_t)shape.size(), 2};

      Value operands[1] = {reduce_broadcasted};
      Type restys[1] = {reduce_broadcasted.getType()};
      auto redwin = stablehlo::ReduceWindowOp::create(
          builder,
          rewriteLocation(innerOp.getLoc(), pc.options.strip_llvm_debuginfo),
          restys, operands, init_values, builder.getDenseI64ArrayAttr(win_dim),
          builder.getDenseI64ArrayAttr(win_strides),
          builder.getDenseI64ArrayAttr(base_dialations),
          builder.getDenseI64ArrayAttr(win_dialations),
          DenseIntElementsAttr::get(
              RankedTensorType::get(padding_shape, builder.getIntegerType(64)),
              padding_dialations));

      auto block = new Block();
      redwin.getBody().push_back(block);

      auto a = block->addArgument(
          unrankedTensorType,
          rewriteLocation(innerOp.getLoc(), pc.options.strip_llvm_debuginfo));
      auto b = block->addArgument(
          unrankedTensorType,
          rewriteLocation(innerOp.getLoc(), pc.options.strip_llvm_debuginfo));

      {
        OpBuilder builder(block, block->end());
        auto addOp = stablehlo::AddOp::create(
            builder,
            rewriteLocation(innerOp.getLoc(), pc.options.strip_llvm_debuginfo),
            a, b);
        stablehlo::ReturnOp::create(
            builder,
            rewriteLocation(innerOp.getLoc(), pc.options.strip_llvm_debuginfo),
            addOp.getResult());
      }

      Value result = redwin->getResult(0);
      if (isa<arith::AddIOp, arith::AddFOp>(&innerOp)) {
        result = stablehlo::AddOp::create(
            builder,
            rewriteLocation(innerOp.getLoc(), pc.options.strip_llvm_debuginfo),
            result, dsts[1]);
      } else if (isa<arith::SubIOp, arith::SubFOp>(&innerOp)) {
        result = stablehlo::SubtractOp::create(
            builder,
            rewriteLocation(innerOp.getLoc(), pc.options.strip_llvm_debuginfo),
            dsts[1], result);
      } else {
        llvm_unreachable("unhandled reduction");
      }

      mapping.map(innerOp.getResult(0), result);
      maps[result] = outputMap;

      continue;
    }

    if (tryRaisingOpToStableHLO(&innerOp, mapping, builder, maps, *newPc)
            .failed())
      return failure();
  }

  auto yld = parallelOp.getBody()->getTerminator();
  for (auto &&[res, yval] :
       llvm::zip_equal(parallelOp.getResults(), yld->getOperands())) {
    auto val = mapping.lookupOrNull(yval);
    if (!val)
      return failure();
    auto outputMap = maps[val];

    if (auto forOp = dyn_cast<affine::AffineForOp>(parallelOp.getOperation())) {

      ssize_t idx_to_reduce = -1;
      for (auto &&[i, expr] :
           llvm::enumerate(outputMap.getAffineMap().getResults())) {
        auto dim = cast<AffineDimExpr>(expr);
        if (outputMap.getOperands()[dim.getPosition()] ==
            forOp.getInductionVar()) {
          assert(idx_to_reduce == -1);
          idx_to_reduce = i;
        }
      }

      if (idx_to_reduce == -1) {
        mapping.map(res, val);
      } else {

        SmallVector<int64_t> startIndices;
        SmallVector<int64_t> limitIndices;
        SmallVector<int64_t> strides;

        SmallVector<AffineExpr> exprs;
        for (auto &&[i, expr0] :
             llvm::enumerate(outputMap.getAffineMap().getResults())) {
          auto expr = expr0;
          if (i == idx_to_reduce) {
            auto range = computeExprRange(outputMap, expr);
            expr = builder.getAffineConstantExpr(range->ub - 1);
          } else {
            exprs.push_back(expr);
          }
          strides.push_back(1);
          if (auto constOp = dyn_cast<AffineConstantExpr>(expr)) {
            startIndices.push_back(constOp.getValue());
            limitIndices.push_back(constOp.getValue() + 1);
            continue;
          }
          auto range = computeExprRange(outputMap, expr);
          startIndices.push_back(range->step < 0 ? range->ub - range->step
                                                 : range->lb);
          limitIndices.push_back(range->step < 0 ? range->lb - range->step
                                                 : range->ub);
        }
        SmallVector<Value> vals;
        for (auto v : outputMap.getOperands()) {
          if (v == forOp.getInductionVar()) {
            v = arith::ConstantIndexOp::create(
                builder,
                rewriteLocation(res.getLoc(), pc.options.strip_llvm_debuginfo),
                0);
          }
          vals.push_back(v);
        }

        auto newVal = stablehlo::SliceOp::create(
            builder,
            rewriteLocation(res.getLoc(), pc.options.strip_llvm_debuginfo), val,
            startIndices, limitIndices, strides);

        SmallVector<int64_t> newShape;
        for (auto &&[i, sz] : llvm::enumerate(newVal.getType().getShape())) {
          if (i != idx_to_reduce) {
            newShape.push_back(sz);
          }
        }
        auto newVal2 = stablehlo::ReshapeOp::create(
            builder,
            rewriteLocation(res.getLoc(), pc.options.strip_llvm_debuginfo),
            RankedTensorType::get(newShape, newVal.getType().getElementType()),
            newVal);
        mapping.map(res, newVal2);
        maps[newVal2] = affine::AffineValueMap(
            AffineMap::get(outputMap.getAffineMap().getNumDims(),
                           outputMap.getAffineMap().getNumSymbols(), exprs,
                           res.getContext()),
            vals);
      }
    } else if (auto pforOp = dyn_cast<affine::AffineParallelOp>(
                   parallelOp.getOperation())) {

      SmallVector<int64_t> dims_to_reduce;
      for (auto &&[i, expr] :
           llvm::enumerate(outputMap.getAffineMap().getResults())) {
        auto dim = cast<AffineDimExpr>(expr);
        auto operand =
            dyn_cast<BlockArgument>(outputMap.getOperands()[dim.getPosition()]);
        if (!operand)
          continue;
        if (operand.getOwner()->getParentOp() == pforOp)
          dims_to_reduce.push_back(operand.getArgNumber());
      }

      SmallVector<Value> dsts;
      SmallVector<affine::AffineValueMap> submaps;
      for (auto idx : dims_to_reduce) {
        auto dst = mapping.lookup(pforOp.getIVs()[idx]);
        dsts.push_back(dst);
        if (!maps.count(dst))
          return failure();
        submaps.push_back(maps.lookup(dst));
      }
      bool align2Ok = true;
      auto outputMap2 = alignMemoryAccess(val, outputMap, dsts.data(), submaps,
                                          builder, *newPc, &align2Ok);
      if (!align2Ok)
        return failure();

      SmallVector<int64_t> idxs_to_reduce;
      SmallVector<int64_t> redshape;
      SmallVector<AffineExpr> newExprs;
      for (auto &&[i, expr] :
           llvm::enumerate(outputMap.getAffineMap().getResults())) {
        auto dim = cast<AffineDimExpr>(expr);
        auto operand =
            dyn_cast<BlockArgument>(outputMap.getOperands()[dim.getPosition()]);
        if (!operand)
          continue;
        if (operand.getOwner()->getParentOp() == pforOp)
          idxs_to_reduce.push_back(i);
        else {
          redshape.push_back(
              cast<RankedTensorType>(val.getType()).getShape()[i]);
          newExprs.push_back(expr);
        }
      }

      ArrayRef<Attribute> reductions = pforOp.getReductions().getValue();
      auto intAttr =
          llvm::dyn_cast<IntegerAttr>(reductions[res.getResultNumber()]);
      if (!intAttr || !arith::symbolizeAtomicRMWKind(intAttr.getInt()))
        return failure();
      auto kind = arith::symbolizeAtomicRMWKind(intAttr.getInt()).value();

      Value inits[1] = {nullptr};
      Value inputs[] = {val};
      Type types[] = {RankedTensorType::get(redshape, res.getType())};

      mlir::Type ET = cast<RankedTensorType>(val.getType()).getElementType();
      auto unrankedTensorType = RankedTensorType::get({}, ET);

      std::string innerRedName;

      switch (kind) {
      case arith::AtomicRMWKind::addf:
      case arith::AtomicRMWKind::addi:
        inits[0] = stablehlo::ConstantOp::create(
            builder,
            rewriteLocation(res.getLoc(), pc.options.strip_llvm_debuginfo),
            builder.getZeroAttr(unrankedTensorType));
        innerRedName = "stablehlo.add";
        break;
      case arith::AtomicRMWKind::mulf:
      case arith::AtomicRMWKind::muli:
        inits[0] = stablehlo::ConstantOp::create(
            builder,
            rewriteLocation(res.getLoc(), pc.options.strip_llvm_debuginfo),
            builder.getOneAttr(unrankedTensorType));
        innerRedName = "stablehlo.multiply";
        break;
      case arith::AtomicRMWKind::ori:
        inits[0] = stablehlo::ConstantOp::create(
            builder,
            rewriteLocation(res.getLoc(), pc.options.strip_llvm_debuginfo),
            builder.getZeroAttr(unrankedTensorType));
        innerRedName = "stablehlo.or";
        break;
      case arith::AtomicRMWKind::xori:
        inits[0] = stablehlo::ConstantOp::create(
            builder,
            rewriteLocation(res.getLoc(), pc.options.strip_llvm_debuginfo),
            builder.getZeroAttr(unrankedTensorType));
        innerRedName = "stablehlo.xor";
        break;
      case arith::AtomicRMWKind::andi:
        inits[0] = stablehlo::ConstantOp::create(
            builder,
            rewriteLocation(res.getLoc(), pc.options.strip_llvm_debuginfo),
            SplatElementsAttr::get(
                unrankedTensorType,
                ArrayRef<Attribute>(IntegerAttr::get(
                    ET, APInt::getAllOnes(ET.getIntOrFloatBitWidth())))));
        innerRedName = "stablehlo.and";
        break;
      case arith::AtomicRMWKind::maximumf:
        inits[0] = stablehlo::ConstantOp::create(
            builder,
            rewriteLocation(res.getLoc(), pc.options.strip_llvm_debuginfo),
            SplatElementsAttr::get(
                unrankedTensorType,
                ArrayRef<Attribute>(FloatAttr::get(
                    ET, -std::numeric_limits<double>::infinity()))));
        innerRedName = "stablehlo.maximum";
        break;
      case arith::AtomicRMWKind::maxnumf:
        inits[0] = stablehlo::ConstantOp::create(
            builder,
            rewriteLocation(res.getLoc(), pc.options.strip_llvm_debuginfo),
            SplatElementsAttr::get(
                unrankedTensorType,
                ArrayRef<Attribute>(FloatAttr::get(
                    ET, -std::numeric_limits<double>::infinity()))));
        innerRedName = "arith.maxnumf";
        break;
      case arith::AtomicRMWKind::minimumf:
        inits[0] = stablehlo::ConstantOp::create(
            builder,
            rewriteLocation(res.getLoc(), pc.options.strip_llvm_debuginfo),
            SplatElementsAttr::get(
                unrankedTensorType,
                ArrayRef<Attribute>(FloatAttr::get(
                    ET, std::numeric_limits<double>::infinity()))));
        innerRedName = "stablehlo.minimum";
        break;
      case arith::AtomicRMWKind::minnumf:
        inits[0] = stablehlo::ConstantOp::create(
            builder,
            rewriteLocation(res.getLoc(), pc.options.strip_llvm_debuginfo),
            SplatElementsAttr::get(
                unrankedTensorType,
                ArrayRef<Attribute>(FloatAttr::get(
                    ET, std::numeric_limits<double>::infinity()))));
        innerRedName = "arith.minnumf";
        break;
      default:
        parallelOp->emitError()
            << "unsupported parallel reduction kind \"" << kind << "\"";
        return failure();
      }

      auto red = stablehlo::ReduceOp::create(
          builder,
          rewriteLocation(val.getLoc(), pc.options.strip_llvm_debuginfo), types,
          inputs, inits, builder.getDenseI64ArrayAttr(idxs_to_reduce));

      auto block = new Block();
      red.getBody().push_back(block);

      auto a = block->addArgument(
          unrankedTensorType,
          rewriteLocation(res.getLoc(), pc.options.strip_llvm_debuginfo));
      auto b = block->addArgument(
          unrankedTensorType,
          rewriteLocation(res.getLoc(), pc.options.strip_llvm_debuginfo));

      {
        OpBuilder builder(block, block->end());
        auto innerRedOp = builder.create(
            rewriteLocation(res.getLoc(), pc.options.strip_llvm_debuginfo),
            StringAttr::get(res.getContext(), innerRedName), ValueRange{a, b},
            TypeRange{unrankedTensorType});
        stablehlo::ReturnOp::create(
            builder,
            rewriteLocation(res.getLoc(), pc.options.strip_llvm_debuginfo),
            innerRedOp->getResult(0));
      }

      SmallVector<Value> vals;
      for (auto v : outputMap.getOperands()) {
        auto operand = dyn_cast<BlockArgument>(v);
        if (operand && operand.getOwner()->getParentOp() == pforOp) {
          v = arith::ConstantIndexOp::create(
              builder,
              rewriteLocation(res.getLoc(), pc.options.strip_llvm_debuginfo),
              0);
        }
        vals.push_back(v);
      }
      affine::AffineValueMap avm(
          AffineMap::get(outputMap.getAffineMap().getNumDims(),
                         outputMap.getAffineMap().getNumSymbols(), newExprs,
                         res.getContext()),
          vals);

      avm.composeSimplifyAndCanonicalize();
      mapping.map(res, red->getResult(0));
      maps[red->getResult(0)] = avm;

    } else {
      llvm_unreachable("unknown input operand");
    }
  }

  parentMapping = mapping;
  return success();
}

bool isLoopLockStepExecutable(
    affine::AffineForOp forOp,
    SmallVectorImpl<affine::LoopReduction> *parallelReductions);
static bool isLockStepExecutable(affine::AffineForOp forOp,
                                 ParallelContext &pc) {
  SmallVector<mlir::affine::LoopReduction> red;
  if (isLoopLockStepExecutable(forOp, &red)) {

    llvm::SmallSet<Operation *, 1> reductions;
    for (auto &&[i, arg] : llvm::enumerate(forOp.getRegionIterArgs())) {
      if (!arg.hasOneUse()) {
        if (pc.options.dump_failed_lockstep)
          llvm::errs() << "multiple use iter arg: " << arg << "\n";
        return false;
      }
      Operation *user = nullptr;
      for (auto user2 : arg.getUsers()) {
        user = user2;
        break;
      }
      assert(user);
      if (user->getParentOp() != forOp) {
        if (pc.options.dump_failed_lockstep)
          llvm::errs() << "user not directly in for: " << *user << "\n";
        return false;
      }
      if (isa<arith::AddIOp, arith::AddFOp>(user)) {
      } else if (auto sub = dyn_cast<arith::SubIOp>(user)) {
        if (sub.getRhs() == arg) {
          if (pc.options.dump_failed_lockstep)
            llvm::errs() << "wrong side sub: " << *user << "\n";
          return false;
        }
      } else if (auto sub = dyn_cast<arith::SubFOp>(user)) {
        if (sub.getRhs() == arg) {
          if (pc.options.dump_failed_lockstep)
            llvm::errs() << "wrong side sub: " << *user << "\n";
          return false;
        }
      } else {
        if (pc.options.dump_failed_lockstep)
          llvm::errs() << "unknown reduction user: " << *user << "\n";
        return false;
      }
      if (reductions.contains(user)) {
        if (pc.options.dump_failed_lockstep)
          llvm::errs() << "multi reduction user: " << *user << "\n";
        return false;
      }
      reductions.insert(user);

      bool hadYield = false;
      for (auto &user2 : user->getResult(0).getUses()) {
        if (auto yld = dyn_cast<affine::AffineYieldOp>(user2.getOwner())) {
          if (user2.getOperandNumber() != i) {
            if (pc.options.dump_failed_lockstep)
              llvm::errs() << "non linked yield: " << *yld << "\n";
            return false;
          }
          hadYield = true;
          continue;
        }
      }
      if (!hadYield) {
        if (pc.options.dump_failed_lockstep)
          llvm::errs() << "no yield user: " << *user << "\n";
        return false;
      }
    }

    return true;
  } else {
    if (pc.options.dump_failed_lockstep) {
      llvm::errs() << "inner failed lockstep executable\n";
    }
    return false;
  }
}

static LogicalResult tryRaisingLockStepForOpToStableHLO(
    affine::AffineForOp forOp, IRMapping &mapping, OpBuilder &builder,
    llvm::DenseMap<Value, affine::AffineValueMap> &maps, ParallelContext pc) {
  LLVM_DEBUG(llvm::dbgs() << "Trying to lock step execute for " << *forOp
                          << "\n");
  if (isLockStepExecutable(forOp, pc)) {
    LLVM_DEBUG(llvm::dbgs() << "Legal\n");
    return tryRaisingParallelOpToStableHLO(forOp, mapping, builder, maps, pc);
  }
  LLVM_DEBUG(llvm::dbgs() << "Illegal\n");
  if (pc.options.dump_failed_lockstep) {
    llvm::errs() << " failed lockstep of for raise: " << *forOp << "\n";
  }
  return failure();
}

static LogicalResult
tryRaisingOpToStableHLO(Operation *op, IRMapping &mapping, OpBuilder &builder,
                        llvm::DenseMap<Value, affine::AffineValueMap> &maps,
                        ParallelContext pc) {

  // Affine load inside a loop becomes a slice
  if (auto loadOp = dyn_cast<affine::AffineLoadOp>(op)) {
    affine::MemRefAccess access(loadOp);

    affine::AffineValueMap accessValueMap;
    access.getAccessMap(&accessValueMap);
    // See tryRaisingForOpToStableHLOUnroll
    accessValueMap.composeSimplifyAndCanonicalize();

    auto inputTen = mapping.lookupOrNull(access.memref);
    if (!inputTen)
      return failure();

    SmallVector<int64_t> outputShape = affineMapShape(accessValueMap, pc);

    SmallVector<int64_t> strides;
    SmallVector<int64_t> reverseDims;

    bool dynIndices = llvm::any_of(accessValueMap.getOperands(), [](Value iv) {
      return affine::isAffineForInductionVar(iv);
    });

    bool emitAsGather =
        affineMapToSlice(accessValueMap, strides, reverseDims, pc).failed() ||
        (dynIndices &&
         llvm::any_of(strides, [](int64_t stride) { return stride != 1; })) ||
        needsGeneralScatterGather(accessValueMap) ||
        usesLaneTensorIV(accessValueMap, mapping, pc);

    if (emitAsGather) {
      SmallVector<Value> lIndices;
      for (auto E : accessValueMap.getAffineMap().getResults()) {
        auto [idx, idxMap] = expandAffineExpr(
            builder,
            rewriteLocation(op->getLoc(), pc.options.strip_llvm_debuginfo), E,
            accessValueMap.getOperands(), mapping, maps,
            accessValueMap.getAffineMap().getNumDims(), pc);
        if (!idx)
          return failure();
        maps[idx] = idxMap;
        lIndices.push_back(idx);
      }

      Value res = emitLoadAsGather(
          rewriteLocation(op->getLoc(), pc.options.strip_llvm_debuginfo),
          inputTen, lIndices, builder, maps);
      if (!res) {
        return op->emitError("failed to raise load (indices of rank > 1)")
               << *op;
      }
      mapping.map(loadOp.getResult(), res);
      return success();
    }

    auto T = RankedTensorType::get(
        outputShape,
        cast<RankedTensorType>(inputTen.getType()).getElementType());

    Value newVal;

    if (dynIndices) {
      SmallVector<Value> startIndices;

      for (auto E : accessValueMap.getAffineMap().getResults()) {
        AffineExpr exprToEmit = E;
        if (!E.isSymbolicOrConstant()) {
          auto iv = getIVForExpr(accessValueMap, E);
          if (pc.isParallelIV(iv)) {
            auto r = computeExprRange(accessValueMap, E);
            auto lb = r->step < 0 ? r->ub - r->step : r->lb;
            exprToEmit = mlir::getAffineConstantExpr(lb, iv.getContext());
          }
        }

        auto [startIndex, _] = expandAffineExpr(
            builder,
            rewriteLocation(op->getLoc(), pc.options.strip_llvm_debuginfo),
            exprToEmit, accessValueMap.getOperands(), mapping, maps,
            accessValueMap.getAffineMap().getNumDims(), pc);
        if (!startIndex)
          return failure();

        startIndices.push_back(startIndex);
      }

      newVal = stablehlo::DynamicSliceOp::create(
          builder,
          rewriteLocation(op->getLoc(), pc.options.strip_llvm_debuginfo), T,
          inputTen, startIndices, outputShape);
    } else {
      bool needSlice = false;
      bool needPad = false;

      SmallVector<int64_t> startIndices;
      SmallVector<int64_t> limitIndices;

      SmallVector<int64_t> padLow;
      SmallVector<int64_t> padHigh;

      bool hasDynamicEdgePadding = llvm::any_of(
          cast<RankedTensorType>(inputTen.getType()).getShape(),
          [](int64_t s) { return s == mlir::ShapedType::kDynamic; });
      SmallVector<Value> dynPadLow;
      SmallVector<Value> dynPadHigh;
      SmallVector<Value> dynPaddingInterior;
      auto i64Ty = builder.getI64Type();
      auto ti64Ty = RankedTensorType::get({1}, i64Ty);

      for (auto [i, T] : llvm::enumerate(llvm::zip_equal(
               accessValueMap.getAffineMap().getResults(), strides,
               cast<RankedTensorType>(inputTen.getType()).getShape()))) {
        auto E = std::get<0>(T);
        auto stride = std::get<1>(T);
        auto sz = std::get<2>(T);

        int64_t start, limit;
        if (auto constOp = dyn_cast<AffineConstantExpr>(E)) {
          start = constOp.getValue();
          limit = constOp.getValue() + 1;
          stride = 1;
        } else {
          auto range = computeExprRange(accessValueMap, E);
          if (!range.has_value())
            return failure();
          start = range->step < 0 ? range->ub - range->step : range->lb;
          limit = range->step < 0 ? range->lb - range->step : range->ub;
        }

        needSlice |= sz != (limit - start) / stride;

        int64_t pLow = 0;
        int64_t pHigh = 0;

        if (start < 0) {
          pLow = -start;
          start = 0;
          limit += pLow;
        }

        if (hasDynamicEdgePadding) {
          needPad = true;
          needSlice = true;

          auto pLowVal = stablehlo::ConstantOp::create(
              builder,
              rewriteLocation(op->getLoc(), pc.options.strip_llvm_debuginfo),
              ti64Ty, cast<ElementsAttr>(builder.getI64TensorAttr({pLow})));
          dynPadLow.push_back(pLowVal);

          auto szVal = stablehlo::GetDimensionSizeOp::create(
              builder,
              rewriteLocation(op->getLoc(), pc.options.strip_llvm_debuginfo),
              inputTen, i);

          auto szVal64 = stablehlo::ConvertOp::create(
              builder,
              rewriteLocation(op->getLoc(), pc.options.strip_llvm_debuginfo),
              RankedTensorType::get({}, builder.getI64Type()), szVal);

          auto szVal1D_Cast = stablehlo::ReshapeOp::create(
              builder,
              rewriteLocation(op->getLoc(), pc.options.strip_llvm_debuginfo),
              ti64Ty, szVal64);

          auto limitVal = stablehlo::ConstantOp::create(
              builder,
              rewriteLocation(op->getLoc(), pc.options.strip_llvm_debuginfo),
              ti64Ty, cast<ElementsAttr>(builder.getI64TensorAttr({limit})));

          auto sumVal = stablehlo::AddOp::create(
              builder,
              rewriteLocation(op->getLoc(), pc.options.strip_llvm_debuginfo),
              szVal1D_Cast, pLowVal);
          auto diffVal = stablehlo::SubtractOp::create(
              builder,
              rewriteLocation(op->getLoc(), pc.options.strip_llvm_debuginfo),
              limitVal, sumVal);

          auto zeroVal = stablehlo::ConstantOp::create(
              builder,
              rewriteLocation(op->getLoc(), pc.options.strip_llvm_debuginfo),
              ti64Ty, cast<ElementsAttr>(builder.getI64TensorAttr({0})));
          auto pHighVal = stablehlo::MaxOp::create(
              builder,
              rewriteLocation(op->getLoc(), pc.options.strip_llvm_debuginfo),
              diffVal, zeroVal);

          dynPadHigh.push_back(pHighVal);
        } else {
          if (limit > sz + pLow) {
            pHigh = limit - (sz + pLow);
          }

          if (pLow != 0 || pHigh != 0) {
            needPad = true;
            needSlice = true;
          }
        }

        padLow.push_back(pLow);
        padHigh.push_back(pHigh);

        startIndices.push_back(start);
        limitIndices.push_back(limit);
      }

      if (needPad) {
        auto elemType =
            cast<RankedTensorType>(inputTen.getType()).getElementType();
        auto tensorType = RankedTensorType::get({}, elemType);
        auto padVal = stablehlo::ConstantOp::create(
            builder,
            rewriteLocation(op->getLoc(), pc.options.strip_llvm_debuginfo),
            tensorType, cast<ElementsAttr>(builder.getZeroAttr(tensorType)));

        if (hasDynamicEdgePadding) {
          auto edgePaddingLow = stablehlo::ConcatenateOp::create(
              builder,
              rewriteLocation(op->getLoc(), pc.options.strip_llvm_debuginfo),
              dynPadLow, 0);
          auto edgePaddingHigh = stablehlo::ConcatenateOp::create(
              builder,
              rewriteLocation(op->getLoc(), pc.options.strip_llvm_debuginfo),
              dynPadHigh, 0);

          auto interiorPadding0 = stablehlo::ConstantOp::create(
              builder,
              rewriteLocation(op->getLoc(), pc.options.strip_llvm_debuginfo),
              ti64Ty, cast<ElementsAttr>(builder.getI64TensorAttr({0})));

          auto interiorPadding = stablehlo::ConcatenateOp::create(
              builder,
              rewriteLocation(op->getLoc(), pc.options.strip_llvm_debuginfo),
              SmallVector<Value>(dynPadLow.size(), interiorPadding0), 0);

          SmallVector<int64_t> paddedShape(
              cast<RankedTensorType>(inputTen.getType()).getShape().size(),
              ShapedType::kDynamic);

          inputTen = stablehlo::DynamicPadOp::create(
              builder,
              rewriteLocation(op->getLoc(), pc.options.strip_llvm_debuginfo),
              cast<RankedTensorType>(inputTen.getType()).clone(paddedShape),
              inputTen, padVal, edgePaddingLow, edgePaddingHigh,
              interiorPadding);
        } else {
          SmallVector<int64_t> paddedShape;
          SmallVector<int64_t> interior(
              cast<RankedTensorType>(inputTen.getType()).getShape().size(), 0);
          for (auto [sz, low, high] :
               llvm::zip(cast<RankedTensorType>(inputTen.getType()).getShape(),
                         padLow, padHigh)) {
            paddedShape.push_back(sz + low + high);
          }

          inputTen = stablehlo::PadOp::create(
              builder,
              rewriteLocation(op->getLoc(), pc.options.strip_llvm_debuginfo),
              cast<RankedTensorType>(inputTen.getType()).clone(paddedShape),
              inputTen, padVal, padLow, padHigh, interior);
        }
      }

      if (needSlice) {
        newVal = stablehlo::SliceOp::create(
            builder,
            rewriteLocation(op->getLoc(), pc.options.strip_llvm_debuginfo), T,
            inputTen, startIndices, limitIndices, strides);
      } else {
        newVal = inputTen;
      }
    }

    if (reverseDims.size())
      newVal = stablehlo::ReverseOp::create(
          builder,
          rewriteLocation(inputTen.getLoc(), pc.options.strip_llvm_debuginfo),
          newVal, reverseDims);

    SmallVector<AffineExpr> dynExprs;
    SmallVector<int64_t> dynShape;

    AffineMap affineMap = accessValueMap.getAffineMap();
    for (auto [S, E] : llvm::zip_equal(outputShape, affineMap.getResults())) {
      if (!E.isSymbolicOrConstant() &&
          pc.isParallelIV(getIVForExpr(accessValueMap, E))) {
        dynExprs.push_back(E);
        dynShape.push_back(S);
      }
    }

    auto val = loadOp.getResult();

    newVal =
        stablehlo::ReshapeOp::create(
            builder,
            rewriteLocation(newVal.getLoc(), pc.options.strip_llvm_debuginfo),
            cast<RankedTensorType>(newVal.getType()).clone(dynShape), newVal)
            .getResult();
    mapping.map(val, newVal);

    affine::AffineValueMap dynAffineValueMap(
        AffineMap::get(affineMap.getNumDims(), affineMap.getNumSymbols(),
                       dynExprs, newVal.getContext()),
        accessValueMap.getOperands());
    maps[newVal] = dynAffineValueMap;

    return success();
  }

  // Affine store inside a loop becomes a dynamic_update_slice
  if (auto storeOp = dyn_cast<affine::AffineStoreOp>(op)) {
    auto operand = mapping.lookupOrNull(storeOp.getMemref());
    auto update = mapping.lookupOrNull(storeOp.getValue());
    if (!operand || !update)
      return failure();

    affine::MemRefAccess access(storeOp);

    affine::AffineValueMap accessValueMap;
    access.getAccessMap(&accessValueMap);
    // See tryRaisingForOpToStableHLOUnroll
    accessValueMap.composeSimplifyAndCanonicalize();

    SmallVector<int64_t> strides;
    SmallVector<int64_t> reverseDims;

    bool emitAsScatter =
        affineMapToSlice(accessValueMap, strides, reverseDims, pc).failed() ||
        llvm::any_of(strides, [](int64_t stride) { return stride != 1; }) ||
        needsGeneralScatterGather(accessValueMap) ||
        usesLaneTensorIV(accessValueMap, mapping, pc);

    if (emitAsScatter) {
      SmallVector<Value> sIndices;
      for (auto E : accessValueMap.getAffineMap().getResults()) {
        auto [expandedIndex, indexMap] = expandAffineExpr(
            builder,
            rewriteLocation(op->getLoc(), pc.options.strip_llvm_debuginfo), E,
            accessValueMap.getOperands(), mapping, maps,
            accessValueMap.getAffineMap().getNumDims(), pc);
        if (!expandedIndex)
          return failure();
        maps[expandedIndex] = indexMap;
        sIndices.push_back(expandedIndex);
      }

      Value res = emitStoreAsScatter(
          rewriteLocation(op->getLoc(), pc.options.strip_llvm_debuginfo),
          update, operand, sIndices, builder, maps, pc);
      if (!res) {
        auto err = op->emitError("affine.store (scatter) is dependent on "
                                 "less dims than stored value: ")
                   << *op;
        for (auto iv : accessValueMap.getOperands()) {
          printAsOperand(err, iv, OpPrintingFlags());
          err << ", ";
        }
        err << "\n";
        err << accessValueMap.getAffineMap();

        return err;
      }
      mapping.map(storeOp.getMemref(), res);
      return success();
    }

    auto rank = access.getRank();
    SmallVector<Value> startIndicesValues;
    startIndicesValues.reserve(rank);

    auto Ty = builder.getI64Type();
    auto unrankedTensorType = RankedTensorType::get({}, Ty);

    if (!maps.contains(update))
      return failure();
    affine::AffineValueMap updateValueMap = maps.lookup(update);

    // for each dim in update, where it will
    // be located in broadcastedupdate
    SmallVector<int64_t> broadcastDims(
        cast<RankedTensorType>(update.getType()).getShape().size(), -1);
    SmallVector<int64_t> updateShape;

    bool needPad = false;
    SmallVector<int64_t> padLow;
    SmallVector<int64_t> padHigh;

    bool hasDynamicEdgePadding =
        llvm::any_of(cast<RankedTensorType>(operand.getType()).getShape(),
                     [](int64_t s) { return s == mlir::ShapedType::kDynamic; });
    SmallVector<Value> dynPadLow;
    SmallVector<Value> dynPadHigh;
    SmallVector<Value> dynNegPadLow;
    SmallVector<Value> dynNegPadHigh;
    SmallVector<Value> dynPaddingInterior;
    auto i64Ty = builder.getI64Type();
    auto ti64Ty = RankedTensorType::get({1}, i64Ty);

    for (auto [i, T] : llvm::enumerate(llvm::zip_equal(
             accessValueMap.getAffineMap().getResults(), strides,
             cast<RankedTensorType>(operand.getType()).getShape()))) {
      auto E = std::get<0>(T);
      [[maybe_unused]] auto stride = std::get<1>(T);
      auto sz = std::get<2>(T);

      int64_t start = 0, limit = 0;
      bool hasRange = true;
      if (auto constOp = dyn_cast<AffineConstantExpr>(E)) {
        start = constOp.getValue();
        limit = constOp.getValue() + 1;
      } else if (!E.isSymbolicOrConstant()) {
        auto range = computeExprRange(accessValueMap, E);
        if (range.has_value()) {
          start = range->step < 0 ? range->ub - range->step : range->lb;
          limit = range->step < 0 ? range->lb - range->step : range->ub;
        } else {
          // A while-raised loop's IV has no static range, but its store is a
          // single dynamically-indexed element along this dim: no padding
          // analysis needed. Only a batched (parallel-IV) dim needs the range.
          Value iv = getIVForExpr(accessValueMap, E);
          if (!iv || pc.isParallelIV(iv))
            return failure();
          hasRange = false;
        }
      } else {
        hasRange = false;
      }

      int64_t pLow = 0;
      int64_t pHigh = 0;

      if (hasRange) {
        if (start < 0) {
          pLow = -start;
          start = 0;
          limit += pLow;
        }
        if (sz == ShapedType::kDynamic) {
          needPad = true;

          auto pLowVal = stablehlo::ConstantOp::create(
              builder,
              rewriteLocation(op->getLoc(), pc.options.strip_llvm_debuginfo),
              ti64Ty, cast<ElementsAttr>(builder.getI64TensorAttr({pLow})));
          dynPadLow.push_back(pLowVal);

          auto szVal = stablehlo::GetDimensionSizeOp::create(
              builder,
              rewriteLocation(op->getLoc(), pc.options.strip_llvm_debuginfo),
              operand, i);

          auto szVal64 = stablehlo::ConvertOp::create(
              builder,
              rewriteLocation(op->getLoc(), pc.options.strip_llvm_debuginfo),
              RankedTensorType::get({}, builder.getI64Type()), szVal);

          auto szVal1D_Cast = stablehlo::ReshapeOp::create(
              builder,
              rewriteLocation(op->getLoc(), pc.options.strip_llvm_debuginfo),
              ti64Ty, szVal64);

          auto limitVal = stablehlo::ConstantOp::create(
              builder,
              rewriteLocation(op->getLoc(), pc.options.strip_llvm_debuginfo),
              ti64Ty, cast<ElementsAttr>(builder.getI64TensorAttr({limit})));

          // pHigh = max(0, limit - (sz + pLow))
          auto sumVal = stablehlo::AddOp::create(
              builder,
              rewriteLocation(op->getLoc(), pc.options.strip_llvm_debuginfo),
              szVal1D_Cast, pLowVal);
          auto diffVal = stablehlo::SubtractOp::create(
              builder,
              rewriteLocation(op->getLoc(), pc.options.strip_llvm_debuginfo),
              limitVal, sumVal);

          auto zeroVal = stablehlo::ConstantOp::create(
              builder,
              rewriteLocation(op->getLoc(), pc.options.strip_llvm_debuginfo),
              ti64Ty, cast<ElementsAttr>(builder.getI64TensorAttr({0})));
          auto pHighVal = stablehlo::MaxOp::create(
              builder,
              rewriteLocation(op->getLoc(), pc.options.strip_llvm_debuginfo),
              diffVal, zeroVal);

          dynPadHigh.push_back(pHighVal);

          auto negOneVal = stablehlo::ConstantOp::create(
              builder,
              rewriteLocation(op->getLoc(), pc.options.strip_llvm_debuginfo),
              ti64Ty, cast<ElementsAttr>(builder.getI64TensorAttr({-1})));
          auto negPLowVal = stablehlo::MulOp::create(
              builder,
              rewriteLocation(op->getLoc(), pc.options.strip_llvm_debuginfo),
              pLowVal, negOneVal);
          dynNegPadLow.push_back(negPLowVal);

          auto negPHighVal = stablehlo::MulOp::create(
              builder,
              rewriteLocation(op->getLoc(), pc.options.strip_llvm_debuginfo),
              pHighVal, negOneVal);
          dynNegPadHigh.push_back(negPHighVal);
        } else {
          if (limit > sz + pLow) {
            pHigh = limit - (sz + pLow);
          }

          if (pLow != 0 || pHigh != 0) {
            needPad = true;
          }

          auto pLowVal = stablehlo::ConstantOp::create(
              builder,
              rewriteLocation(op->getLoc(), pc.options.strip_llvm_debuginfo),
              ti64Ty, cast<ElementsAttr>(builder.getI64TensorAttr({pLow})));
          dynPadLow.push_back(pLowVal);

          auto pHighVal = stablehlo::ConstantOp::create(
              builder,
              rewriteLocation(op->getLoc(), pc.options.strip_llvm_debuginfo),
              ti64Ty, cast<ElementsAttr>(builder.getI64TensorAttr({pHigh})));
          dynPadHigh.push_back(pHighVal);

          auto negPLowVal = stablehlo::ConstantOp::create(
              builder,
              rewriteLocation(op->getLoc(), pc.options.strip_llvm_debuginfo),
              ti64Ty, cast<ElementsAttr>(builder.getI64TensorAttr({-pLow})));
          dynNegPadLow.push_back(negPLowVal);

          auto negPHighVal = stablehlo::ConstantOp::create(
              builder,
              rewriteLocation(op->getLoc(), pc.options.strip_llvm_debuginfo),
              ti64Ty, cast<ElementsAttr>(builder.getI64TensorAttr({-pHigh})));
          dynNegPadHigh.push_back(negPHighVal);
        }
      } else {
        auto pLowVal = stablehlo::ConstantOp::create(
            builder,
            rewriteLocation(op->getLoc(), pc.options.strip_llvm_debuginfo),
            ti64Ty, cast<ElementsAttr>(builder.getI64TensorAttr({0})));
        dynPadLow.push_back(pLowVal);

        auto pHighVal = stablehlo::ConstantOp::create(
            builder,
            rewriteLocation(op->getLoc(), pc.options.strip_llvm_debuginfo),
            ti64Ty, cast<ElementsAttr>(builder.getI64TensorAttr({0})));
        dynPadHigh.push_back(pHighVal);

        dynNegPadLow.push_back(pLowVal);
        dynNegPadHigh.push_back(pHighVal);
      }

      padLow.push_back(pLow);
      padHigh.push_back(pHigh);

      auto zeroVal = stablehlo::ConstantOp::create(
          builder,
          rewriteLocation(op->getLoc(), pc.options.strip_llvm_debuginfo),
          ti64Ty, cast<ElementsAttr>(builder.getI64TensorAttr({0})));
      dynPaddingInterior.push_back(zeroVal);

      Value startIndex;
      if (E.isSymbolicOrConstant()) {
        startIndex =
            stablehlo::ConstantOp::create(
                builder,
                rewriteLocation(op->getLoc(), pc.options.strip_llvm_debuginfo),
                unrankedTensorType,
                SplatElementsAttr::get(
                    unrankedTensorType,
                    ArrayRef<Attribute>(IntegerAttr::get(
                        Ty, cast<AffineConstantExpr>(E).getValue()))))
                .getResult();
        updateShape.push_back(1);
      } else {

        unsigned dim = 0;
        for (unsigned e = accessValueMap.getAffineMap().getNumDims(); dim < e;
             ++dim) {
          if (E.isFunctionOfDim(dim))
            break;
        }

        auto iv = accessValueMap.getOperands()[dim];

        AffineExpr exprToEmit = E;
        if (pc.isParallelIV(iv)) {
          auto r = computeExprRange(accessValueMap, E);
          auto lb = r->step < 0 ? r->ub - r->step : r->lb;
          exprToEmit = mlir::getAffineConstantExpr(lb, iv.getContext());
          updateShape.push_back(r->getNumIters());
        } else {
          updateShape.push_back(1);
        }

        auto [startIndex_, _] = expandAffineExpr(
            builder,
            rewriteLocation(iv.getLoc(), pc.options.strip_llvm_debuginfo),
            exprToEmit, accessValueMap.getOperands(), mapping, maps,
            accessValueMap.getAffineMap().getNumDims(), pc);
        if (!startIndex_)
          return failure();
        startIndex = startIndex_;
      }

      if (pLow != 0) {
        auto pLowVal = stablehlo::ConstantOp::create(
            builder,
            rewriteLocation(op->getLoc(), pc.options.strip_llvm_debuginfo),
            unrankedTensorType,
            SplatElementsAttr::get(
                unrankedTensorType,
                ArrayRef<Attribute>(IntegerAttr::get(Ty, pLow))));
        startIndex =
            stablehlo::AddOp::create(
                builder,
                rewriteLocation(op->getLoc(), pc.options.strip_llvm_debuginfo),
                startIndex, pLowVal.getResult())
                .getResult();
      }

      startIndicesValues.push_back(startIndex);

      if (E.isSymbolicOrConstant())
        continue;

      // find dim in update which varies along the same iv
      Value storeIv = getIVForExpr(accessValueMap, E);

      for (auto [updateIdx, EE] :
           llvm::enumerate(updateValueMap.getAffineMap().getResults())) {
        Value updateIv = getIVForExpr(updateValueMap, EE);
        if (storeIv == updateIv) {
          broadcastDims[updateIdx] = (updateShape.size() - 1);
          break;
        }
      }
    }

    // A value varying along an axis the destination does not index is a racy
    // write: every lane along that axis stores to the same location, and the
    // original program does not define which lane wins. Refining to lane 0
    // is sound — unless a mask varies along the axis, since then which lanes
    // write at all is data-dependent.
    {
      SmallVector<int64_t> raceDims;
      for (auto [updateIdx, dim] : llvm::enumerate(broadcastDims))
        if (dim == -1)
          raceDims.push_back((int64_t)updateIdx);
      bool refine = !raceDims.empty();
      if (refine && pc.mask) {
        affine::AffineValueMap mm = maps.lookup(pc.mask);
        for (int64_t ri : raceDims) {
          Value riv = getIVForExpr(updateValueMap,
                                   updateValueMap.getAffineMap().getResult(ri));
          for (auto E : mm.getAffineMap().getResults())
            if (getIVForExpr(mm, E) == riv)
              refine = false;
        }
      }
      // A guard varying along the race axis admits some subset of lanes:
      // extract any admitted lane's value with a masked-pick reduction (for
      // a one-hot guard that is the writing lane; equal racing values give
      // the same answer; a genuine race permits any).
      bool maskedPick = false;
      if (!refine && !raceDims.empty() && pc.mask) {
        Value mAligned = pc.mask;
        Value updAligned = update;
        affine::AffineValueMap mm = maps.lookup(pc.mask);
        bool alignOk = true;
        affine::AffineValueMap unionMap = alignMemoryAccess(
            mAligned, mm, updAligned, updateValueMap, builder, pc, &alignOk);
        if (alignOk && updAligned != update &&
            mAligned.getType() ==
                RankedTensorType::get(
                    cast<RankedTensorType>(updAligned.getType()).getShape(),
                    builder.getI1Type())) {
          // The mask spans axes the update lacks: pick in the union space,
          // keeping the axes the store indexes.
          auto loc =
              rewriteLocation(op->getLoc(), pc.options.strip_llvm_debuginfo);
          auto AT = cast<RankedTensorType>(updAligned.getType());
          SmallVector<int64_t> unionRace;
          SmallVector<int64_t> keptShape;
          SmallVector<AffineExpr> keptExprs;
          SmallVector<int64_t> keptBroadcastDims;
          bool ok = true;
          for (auto [i, E] :
               llvm::enumerate(unionMap.getAffineMap().getResults())) {
            Value uiv = getIVForExpr(unionMap, E);
            int64_t storeDim = -1;
            if (uiv)
              for (auto [k, SE] :
                   llvm::enumerate(storeOp.getMap().getResults())) {
                if (SE.isSymbolicOrConstant())
                  continue;
                if (getIVForExpr(accessValueMap,
                                 accessValueMap.getAffineMap().getResult(k)) ==
                    uiv)
                  storeDim = (int64_t)k;
              }
            if (uiv && storeDim != -1) {
              keptShape.push_back(AT.getShape()[i]);
              keptExprs.push_back(E);
              keptBroadcastDims.push_back(storeDim);
            } else {
              unionRace.push_back((int64_t)i);
            }
          }
          if (ok && !unionRace.empty()) {
            auto elemTy = RankedTensorType::get({}, AT.getElementType());
            auto boolTy = RankedTensorType::get({}, builder.getI1Type());
            Value zeroInit = stablehlo::ConstantOp::create(
                builder, loc, elemTy,
                SplatElementsAttr::get(
                    elemTy, builder.getZeroAttr(AT.getElementType())));
            Value falseInit = stablehlo::ConstantOp::create(
                builder, loc, boolTy,
                SplatElementsAttr::get(boolTy, builder.getBoolAttr(false)));
            auto reduce = stablehlo::ReduceOp::create(
                builder, loc, ValueRange{updAligned, mAligned},
                ValueRange{zeroInit, falseInit},
                builder.getDenseI64ArrayAttr(unionRace));
            {
              Block *rb = new Block();
              reduce.getBody().push_back(rb);
              Value av = rb->addArgument(elemTy, loc);
              Value am = rb->addArgument(boolTy, loc);
              Value bv = rb->addArgument(elemTy, loc);
              Value bm = rb->addArgument(boolTy, loc);
              OpBuilder::InsertionGuard guard(builder);
              builder.setInsertionPointToStart(rb);
              Value v = stablehlo::SelectOp::create(builder, loc, am, av, bv);
              Value m = stablehlo::OrOp::create(builder, loc, am, bm);
              stablehlo::ReturnOp::create(builder, loc, ValueRange{v, m});
            }
            update = stablehlo::ReshapeOp::create(
                builder, loc,
                RankedTensorType::get(keptShape, AT.getElementType()),
                reduce.getResult(0));
            updateValueMap = affine::AffineValueMap(
                AffineMap::get(unionMap.getAffineMap().getNumDims(),
                               unionMap.getAffineMap().getNumSymbols(),
                               keptExprs, op->getContext()),
                unionMap.getOperands());
            updateValueMap.composeSimplifyAndCanonicalize();
            maps[update] = updateValueMap;
            broadcastDims.assign(keptBroadcastDims.begin(),
                                 keptBroadcastDims.end());
            raceDims.clear();
          }
        } else if (alignOk && updAligned == update &&
                   mAligned.getType() ==
                       RankedTensorType::get(
                           cast<RankedTensorType>(update.getType()).getShape(),
                           builder.getI1Type())) {
          auto loc =
              rewriteLocation(op->getLoc(), pc.options.strip_llvm_debuginfo);
          auto UT = cast<RankedTensorType>(update.getType());
          auto elemTy = RankedTensorType::get({}, UT.getElementType());
          auto boolTy = RankedTensorType::get({}, builder.getI1Type());
          Value zeroInit = stablehlo::ConstantOp::create(
              builder, loc, elemTy,
              SplatElementsAttr::get(elemTy,
                                     builder.getZeroAttr(UT.getElementType())));
          Value falseInit = stablehlo::ConstantOp::create(
              builder, loc, boolTy,
              SplatElementsAttr::get(boolTy, builder.getBoolAttr(false)));
          auto reduce = stablehlo::ReduceOp::create(
              builder, loc, ValueRange{update, mAligned},
              ValueRange{zeroInit, falseInit},
              builder.getDenseI64ArrayAttr(raceDims));
          {
            Block *rb = new Block();
            reduce.getBody().push_back(rb);
            Value av = rb->addArgument(elemTy, loc);
            Value am = rb->addArgument(boolTy, loc);
            Value bv = rb->addArgument(elemTy, loc);
            Value bm = rb->addArgument(boolTy, loc);
            OpBuilder::InsertionGuard guard(builder);
            builder.setInsertionPointToStart(rb);
            Value v = stablehlo::SelectOp::create(builder, loc, am, av, bv);
            Value m = stablehlo::OrOp::create(builder, loc, am, bm);
            stablehlo::ReturnOp::create(builder, loc, ValueRange{v, m});
          }
          // Reinstall the race axes as unit dims so the shared kept-axis
          // rebuild below slices them away.
          SmallVector<int64_t> unitShape(UT.getShape().begin(),
                                         UT.getShape().end());
          for (int64_t ri : raceDims)
            unitShape[ri] = 1;
          update = stablehlo::ReshapeOp::create(
              builder, loc,
              RankedTensorType::get(unitShape, UT.getElementType()),
              reduce.getResult(0));
          maskedPick = true;
          refine = true;
        }
      }
      if (refine) {
        auto UT = cast<RankedTensorType>(update.getType());
        if (!maskedPick) {
          SmallVector<int64_t> starts(UT.getRank(), 0);
          SmallVector<int64_t> limits(UT.getShape().begin(),
                                      UT.getShape().end());
          SmallVector<int64_t> ones(UT.getRank(), 1);
          for (int64_t ri : raceDims)
            limits[ri] = 1;
          update = stablehlo::SliceOp::create(
              builder,
              rewriteLocation(op->getLoc(), pc.options.strip_llvm_debuginfo),
              update, starts, limits, ones);
        }
        SmallVector<int64_t> keptShape;
        SmallVector<AffineExpr> keptExprs;
        SmallVector<int64_t> keptBroadcastDims;
        for (auto [updateIdx, dim] : llvm::enumerate(broadcastDims)) {
          if (llvm::is_contained(raceDims, (int64_t)updateIdx))
            continue;
          keptShape.push_back(UT.getShape()[updateIdx]);
          keptExprs.push_back(
              updateValueMap.getAffineMap().getResult(updateIdx));
          keptBroadcastDims.push_back(dim);
        }
        update = stablehlo::ReshapeOp::create(
            builder,
            rewriteLocation(op->getLoc(), pc.options.strip_llvm_debuginfo),
            RankedTensorType::get(keptShape, UT.getElementType()), update);
        updateValueMap = affine::AffineValueMap(
            AffineMap::get(updateValueMap.getAffineMap().getNumDims(),
                           updateValueMap.getAffineMap().getNumSymbols(),
                           keptExprs, op->getContext()),
            updateValueMap.getOperands());
        updateValueMap.composeSimplifyAndCanonicalize();
        maps[update] = updateValueMap;
        broadcastDims.assign(keptBroadcastDims.begin(),
                             keptBroadcastDims.end());
      }
    }

    // Store has less ivs than load which can signify a reduction that is not
    // handled.
    if (llvm::any_of(broadcastDims, [](int64_t dim) { return dim == -1; })) {
      auto err =
          op->emitError(
              "affine.store is dependent on less dims than stored value:\n")
          << *op;
      for (auto iv : accessValueMap.getOperands()) {
        printAsOperand(err, iv, OpPrintingFlags());
        err << ", ";
      }
      err << "\n";
      err << accessValueMap.getAffineMap();
      for (auto iv : updateValueMap.getOperands()) {
        printAsOperand(err, iv, OpPrintingFlags());
        err << ", ";
      }
      err << "\n";
      err << updateValueMap.getAffineMap();
      return err;
    }

    update = stablehlo::BroadcastInDimOp::create(
        builder, rewriteLocation(op->getLoc(), pc.options.strip_llvm_debuginfo),
        cast<RankedTensorType>(update.getType()).clone(updateShape), update,
        broadcastDims);

    if (!update)
      return failure();

    if (reverseDims.size())
      update = stablehlo::ReverseOp::create(
          builder,
          rewriteLocation(storeOp.getLoc(), pc.options.strip_llvm_debuginfo),
          update, reverseDims);

    if (pc.mask) {
      Value mask = pc.mask;
      affine::AffineValueMap maskMap = maps.lookup(mask);

      // A mask axis the destination does not index (a single-lane broadcast
      // store like tid==0) writes one representative value: when the stored
      // value is invariant along that axis, or-reduce the mask over it and
      // store under the reduced predicate.
      {
        SmallVector<int64_t> reduceDims;
        bool invariant = true;
        for (auto [i, E] :
             llvm::enumerate(maskMap.getAffineMap().getResults())) {
          Value iv = getIVForExpr(maskMap, E);
          if (!iv || llvm::is_contained(storeOp.getIndices(), iv))
            continue;
          for (auto EE : updateValueMap.getAffineMap().getResults())
            if (getIVForExpr(updateValueMap, EE) == iv)
              invariant = false;
          reduceDims.push_back(i);
        }
        if (!reduceDims.empty() && invariant) {
          auto loc =
              rewriteLocation(op->getLoc(), pc.options.strip_llvm_debuginfo);
          auto maskTy = cast<RankedTensorType>(mask.getType());
          auto boolTy = RankedTensorType::get({}, maskTy.getElementType());
          Value initFalse = stablehlo::ConstantOp::create(
              builder, loc, boolTy,
              SplatElementsAttr::get(boolTy, builder.getBoolAttr(false)));
          auto reduce = stablehlo::ReduceOp::create(
              builder, loc, ValueRange{mask}, ValueRange{initFalse},
              builder.getDenseI64ArrayAttr(reduceDims));
          {
            Block *body = new Block();
            reduce.getBody().push_back(body);
            body->addArgument(boolTy, loc);
            body->addArgument(boolTy, loc);
            OpBuilder::InsertionGuard guard(builder);
            builder.setInsertionPointToStart(body);
            Value ored = stablehlo::OrOp::create(
                builder, loc, body->getArgument(0), body->getArgument(1));
            stablehlo::ReturnOp::create(builder, loc, ored);
          }
          mask = reduce.getResult(0);
          SmallVector<AffineExpr> keptExprs;
          for (auto [i, E] :
               llvm::enumerate(maskMap.getAffineMap().getResults()))
            if (!llvm::is_contained(reduceDims, (int64_t)i))
              keptExprs.push_back(E);
          maskMap = affine::AffineValueMap(
              AffineMap::get(maskMap.getAffineMap().getNumDims(),
                             maskMap.getAffineMap().getNumSymbols(), keptExprs,
                             op->getContext()),
              maskMap.getOperands());
          maskMap.composeSimplifyAndCanonicalize();
          maps[mask] = maskMap;
        }
      }

      // A scalar (or lower-rank) update stores one element along the store
      // dims it does not carry: give it explicit size-1 dims so every store
      // dimension has an update dimension.
      {
        ShapedType ut = cast<ShapedType>(update.getType());
        int64_t numResults = (int64_t)storeOp.getMap().getNumResults();
        if (ut.getRank() < numResults) {
          SmallVector<int64_t> shape(numResults - ut.getRank(), 1);
          shape.append(ut.getShape().begin(), ut.getShape().end());
          auto newTy = RankedTensorType::get(shape, ut.getElementType());
          update = stablehlo::ReshapeOp::create(
              builder,
              rewriteLocation(op->getLoc(), pc.options.strip_llvm_debuginfo),
              newTy, update);
        }
      }

      // A lane axis wider than the destination: lanes past the end could
      // only have stored out of bounds, so their mask is necessarily false
      // and the excess slices away. Only tiles anchored at the buffer
      // origin are handled; anything else still reports below.
      {
        ShapedType ut = cast<ShapedType>(update.getType());
        auto operandShape =
            cast<RankedTensorType>(operand.getType()).getShape();
        if (ut.getRank() == (int64_t)operandShape.size()) {
          for (auto [i, E] : llvm::enumerate(storeOp.getMap().getResults())) {
            int64_t u = ut.getShape()[i], o = operandShape[i];
            if (u == ShapedType::kDynamic || o == ShapedType::kDynamic ||
                u <= o)
              continue;
            DenseIntElementsAttr startAttr;
            if (!matchPattern(startIndicesValues[i], m_Constant(&startAttr)) ||
                !startAttr.getSplatValue<APInt>().isZero())
              continue;
            Value iv = getIVForExpr(accessValueMap, E);
            if (!iv)
              continue;
            // The mask must carry the same axis at full extent so the pair
            // stays aligned after the cut.
            int64_t maskAxis = -1;
            auto maskTy = cast<RankedTensorType>(mask.getType());
            for (auto [j, ME] :
                 llvm::enumerate(maskMap.getAffineMap().getResults()))
              if (getIVForExpr(maskMap, ME) == iv) {
                maskAxis = (int64_t)j;
                break;
              }
            if (maskAxis < 0 || maskTy.getRank() <= maskAxis ||
                maskTy.getShape()[maskAxis] != u)
              continue;
            auto loc =
                rewriteLocation(op->getLoc(), pc.options.strip_llvm_debuginfo);
            auto sliceTo = [&](Value v, int64_t axis, int64_t len) {
              auto vt = cast<RankedTensorType>(v.getType());
              SmallVector<int64_t> starts(vt.getRank(), 0),
                  limits(vt.getShape().begin(), vt.getShape().end()),
                  strides(vt.getRank(), 1);
              limits[axis] = len;
              return stablehlo::SliceOp::create(builder, loc, v, starts,
                                                limits, strides)
                  .getResult();
            };
            update = sliceTo(update, (int64_t)i, o);
            ut = cast<ShapedType>(update.getType());
            mask = sliceTo(mask, maskAxis, o);
            maps[mask] = maskMap;
          }
        }
      }

      // here this is a bit annoying but alignMemoryAccess expects non constant
      // dims in its value maps. as such, we remove constant dims from the
      // update and subsequent previous value as to use the storeValueMap.
      // we will put these constant dims (size = 1) to the masked update
      // afterwards.
      SmallVector<unsigned> nonConstantDims;

      ShapedType updateType = cast<ShapedType>(update.getType());
      SmallVector<int64_t> updateShapeWithoutConstantDims;

      for (auto [i, E] : llvm::enumerate(storeOp.getMap().getResults())) {
        if (!E.isSymbolicOrConstant()) {
          nonConstantDims.push_back(i);
          updateShapeWithoutConstantDims.push_back(updateType.getShape()[i]);
        }
      }

      affine::AffineValueMap storeValueMap(
          storeOp.getMap().getSubMap(nonConstantDims), storeOp.getIndices());

      SmallVector<int64_t> updateShape(updateType.getShape().begin(),
                                       updateType.getShape().end());
      // An update that does not fit the buffer -- a stray lane axis wider
      // than the destination (per-lane scratch modeled as one shared buffer;
      // privatization is the real fix), or a dynamic dim -- would make the
      // slice below fatally fail result type inference. Report it instead.
      auto operandShape = cast<RankedTensorType>(operand.getType()).getShape();
      bool unfit = updateShape.size() != operandShape.size();
      if (!unfit)
        for (auto [u, o] : llvm::zip(updateShape, operandShape))
          if (u == ShapedType::kDynamic ||
              (o != ShapedType::kDynamic && u > o))
            unfit = true;
      if (unfit) {
        auto err =
            op->emitError("masked store update shape does not fit buffer: ");
        err << "update " << update.getType() << " mask " << pc.mask.getType()
            << " buffer " << operand.getType();
        return err;
      }
      Value prev = stablehlo::DynamicSliceOp::create(
          builder,
          rewriteLocation(op->getLoc(), pc.options.strip_llvm_debuginfo),
          operand, startIndicesValues, updateShape);

      Value updateWithoutConstantDims = stablehlo::ReshapeOp::create(
          builder,
          rewriteLocation(op->getLoc(), pc.options.strip_llvm_debuginfo),
          updateType.clone(updateShapeWithoutConstantDims), update);
      Value prevWithoutConstantDims = stablehlo::ReshapeOp::create(
          builder,
          rewriteLocation(op->getLoc(), pc.options.strip_llvm_debuginfo),
          updateType.clone(updateShapeWithoutConstantDims), prev);

      Value vals[] = {updateWithoutConstantDims, prevWithoutConstantDims};
      affine::AffineValueMap dsts[] = {storeValueMap, storeValueMap};

      // update what if cond has more ivs dependence than the update?
      // or different?
      bool alignOk = true;
      storeValueMap =
          alignMemoryAccess(mask, maskMap, vals, dsts, builder, pc, &alignOk);
      if (!alignOk)
        return op->emitError("cannot align masked store");

      for (auto dim : storeValueMap.getOperands()) {
        // This dim is present in the masked update and not in the stored
        // dimensions.
        if (!llvm::is_contained(storeOp.getIndices(), dim)) {
          auto err = op->emitError(
                         "masked affine.store is dependent on less dimensions "
                         "than masked stored value:\n")
                     << *op << "\n";
          for (auto iv : accessValueMap.getOperands()) {
            printAsOperand(err, iv, OpPrintingFlags());
            err << ", ";
          }
          err << ": ";
          err << accessValueMap.getAffineMap() << "\n";
          for (auto iv : storeValueMap.getOperands()) {
            printAsOperand(err, iv, OpPrintingFlags());
            err << ", ";
          }
          err << ": ";
          err << storeValueMap.getAffineMap();
          return err;
        }
      }

      Value maskedUpdate = stablehlo::SelectOp::create(
          builder,
          rewriteLocation(op->getLoc(), pc.options.strip_llvm_debuginfo), mask,
          vals[0], vals[1]);

      SmallVector<int64_t> maskedUpdateBroadcastDims(
          storeValueMap.getNumResults(), -1);

      for (auto [i, E] :
           llvm::enumerate(storeValueMap.getAffineMap().getResults())) {
        assert(!E.isSymbolicOrConstant()); // constant dims have been removed
        auto iv = getIVForExpr(storeValueMap, E);

        for (auto [j, EE] : llvm::enumerate(storeOp.getMap().getResults())) {
          if (EE.isSymbolicOrConstant())
            continue;

          int ivPos = 0;
          for (int e = storeOp.getMap().getNumDims(); ivPos < e; ++ivPos) {
            if (EE.isFunctionOfDim(ivPos))
              break;
          }

          auto storeIV = storeOp.getIndices()[ivPos];

          if (iv == storeIV) {
            assert(maskedUpdateBroadcastDims[i] == -1);
            maskedUpdateBroadcastDims[i] = j;
            break;
          }
        }
      }

      if (llvm::any_of(maskedUpdateBroadcastDims,
                       [](int64_t dim) { return dim == -1; })) {
        return op->emitError(
            "could not align masked update to the store location");
      }

      update = stablehlo::BroadcastInDimOp::create(
          builder,
          rewriteLocation(op->getLoc(), pc.options.strip_llvm_debuginfo),
          updateType, maskedUpdate, maskedUpdateBroadcastDims);
    }

    if (needPad) {
      auto elemType =
          cast<RankedTensorType>(operand.getType()).getElementType();
      auto tensorType = RankedTensorType::get({}, elemType);
      auto padVal = stablehlo::ConstantOp::create(
          builder,
          rewriteLocation(op->getLoc(), pc.options.strip_llvm_debuginfo),
          tensorType, cast<ElementsAttr>(builder.getZeroAttr(tensorType)));

      if (hasDynamicEdgePadding) {
        auto edgePaddingLow = stablehlo::ConcatenateOp::create(
            builder,
            rewriteLocation(op->getLoc(), pc.options.strip_llvm_debuginfo),
            dynPadLow, 0);
        auto edgePaddingHigh = stablehlo::ConcatenateOp::create(
            builder,
            rewriteLocation(op->getLoc(), pc.options.strip_llvm_debuginfo),
            dynPadHigh, 0);
        auto interiorPadding = stablehlo::ConcatenateOp::create(
            builder,
            rewriteLocation(op->getLoc(), pc.options.strip_llvm_debuginfo),
            dynPaddingInterior, 0);

        SmallVector<int64_t> paddedShape(
            cast<RankedTensorType>(operand.getType()).getShape().size(),
            ShapedType::kDynamic);

        operand = stablehlo::DynamicPadOp::create(
            builder,
            rewriteLocation(op->getLoc(), pc.options.strip_llvm_debuginfo),
            cast<RankedTensorType>(operand.getType()).clone(paddedShape),
            operand, padVal, edgePaddingLow, edgePaddingHigh, interiorPadding);
      } else {
        SmallVector<int64_t> paddedShape;
        SmallVector<int64_t> interior(
            cast<RankedTensorType>(operand.getType()).getShape().size(), 0);
        for (auto [sz, low, high] :
             llvm::zip(cast<RankedTensorType>(operand.getType()).getShape(),
                       padLow, padHigh)) {
          paddedShape.push_back(sz + low + high);
        }

        operand = stablehlo::PadOp::create(
            builder,
            rewriteLocation(op->getLoc(), pc.options.strip_llvm_debuginfo),
            cast<RankedTensorType>(operand.getType()).clone(paddedShape),
            operand, padVal, padLow, padHigh, interior);
      }
    }

    auto newOperand = stablehlo::DynamicUpdateSliceOp::create(
        builder, rewriteLocation(op->getLoc(), pc.options.strip_llvm_debuginfo),
        operand, update, startIndicesValues);

    Value finalResult = newOperand.getResult();

    if (needPad) {
      if (hasDynamicEdgePadding) {
        auto elemType =
            cast<RankedTensorType>(finalResult.getType()).getElementType();
        auto tensorType = RankedTensorType::get({}, elemType);
        auto padVal = stablehlo::ConstantOp::create(
            builder,
            rewriteLocation(op->getLoc(), pc.options.strip_llvm_debuginfo),
            tensorType, cast<ElementsAttr>(builder.getZeroAttr(tensorType)));

        auto edgePaddingLow = stablehlo::ConcatenateOp::create(
            builder,
            rewriteLocation(op->getLoc(), pc.options.strip_llvm_debuginfo),
            dynNegPadLow, 0);
        auto edgePaddingHigh = stablehlo::ConcatenateOp::create(
            builder,
            rewriteLocation(op->getLoc(), pc.options.strip_llvm_debuginfo),
            dynNegPadHigh, 0);
        auto interiorPadding = stablehlo::ConcatenateOp::create(
            builder,
            rewriteLocation(op->getLoc(), pc.options.strip_llvm_debuginfo),
            dynPaddingInterior, 0);

        finalResult = stablehlo::DynamicPadOp::create(
            builder,
            rewriteLocation(op->getLoc(), pc.options.strip_llvm_debuginfo),
            cast<RankedTensorType>(finalResult.getType())
                .clone(
                    cast<ShapedType>(storeOp.getMemref().getType()).getShape()),
            finalResult, padVal, edgePaddingLow, edgePaddingHigh,
            interiorPadding);
      } else {
        SmallVector<int64_t> startSlice;
        SmallVector<int64_t> limitSlice;
        SmallVector<int64_t> stridesSlice;
        for (auto [sz, low, high] : llvm::zip(
                 cast<ShapedType>(storeOp.getMemref().getType()).getShape(),
                 padLow, padHigh)) {
          startSlice.push_back(low);
          limitSlice.push_back(low + sz);
          stridesSlice.push_back(1);
        }
        finalResult = stablehlo::SliceOp::create(
            builder,
            rewriteLocation(op->getLoc(), pc.options.strip_llvm_debuginfo),
            cast<RankedTensorType>(finalResult.getType())
                .clone(
                    cast<ShapedType>(storeOp.getMemref().getType()).getShape()),
            finalResult, startSlice, limitSlice, stridesSlice);
      }
    }

    mapping.map(storeOp.getMemref(), finalResult);
    return success();
  }

  if (auto loadOp = dyn_cast<memref::LoadOp>(op)) {
    auto memref = loadOp.getMemref();

    SmallVector<Value> lIndices;
    for (auto idx : loadOp.getIndices()) {
      Value mapped = mapping.lookupOrNull(idx);
      if (!mapped || !maps.count(mapped))
        return failure();
      lIndices.push_back(mapped);
    }
    if (!mapping.lookupOrNull(memref))
      return failure();

    Value res = emitLoadAsGather(
        rewriteLocation(op->getLoc(), pc.options.strip_llvm_debuginfo),
        mapping.lookup(memref), lIndices, builder, maps);
    if (!res) {
      return failure();
    }
    mapping.map(loadOp.getResult(), res);

    return success();
  }

  if (auto storeOp = dyn_cast<memref::StoreOp>(op)) {
    Value value = storeOp.getValueToStore();
    Value memref = storeOp.getMemref();

    SmallVector<Value> sIndices;
    for (auto idx : storeOp.getIndices()) {
      Value mapped = mapping.lookupOrNull(idx);
      if (!mapped || !maps.count(mapped))
        return failure();
      sIndices.push_back(mapped);
    }
    if (!mapping.lookupOrNull(value) || !mapping.lookupOrNull(memref))
      return failure();

    Value res = emitStoreAsScatter(
        rewriteLocation(op->getLoc(), pc.options.strip_llvm_debuginfo),
        mapping.lookup(value), mapping.lookup(memref), sIndices, builder, maps,
        pc);
    if (!res) {
      return op->emitError(
                 "memref.store is dependent on less dims than stored value: ")
             << *op;
    }

    mapping.map(memref, res);

    return success();
  }

  if (isa<ub::PoisonOp, LLVM::PoisonOp, LLVM::ZeroOp, LLVM::UndefOp>(op)) {
    affine::AffineValueMap accessMap(AffineMap::get(op->getContext()), {});

    Type ET = op->getResult(0).getType();
    auto unrankedTensorType = RankedTensorType::get({}, ET);

    if (!ET.isInteger() && !isa<FloatType>(ET))
      return failure();

    auto newConst = stablehlo::ConstantOp::create(
        builder, rewriteLocation(op->getLoc(), pc.options.strip_llvm_debuginfo),
        unrankedTensorType,
        SplatElementsAttr::get(
            unrankedTensorType,
            ArrayRef<Attribute>(
                ET.isInteger()
                    ? (Attribute)IntegerAttr::get(ET, 0)
                    : (Attribute)FloatAttr::get(
                          ET, APFloat::getZero(
                                  cast<FloatType>(ET).getFloatSemantics())))));

    auto newVal = newConst.getResult();
    mapping.map(op->getResult(0), newVal);
    maps[newVal] = accessMap;

    return success();
  }

  if (auto constOp = dyn_cast<arith::ConstantOp>(op)) {
    affine::AffineValueMap accessMap(AffineMap::get(op->getContext()), {});

    auto isIndex = isa<IndexType>(constOp.getType());
    auto ET = isIndex ? builder.getI64Type() : constOp.getType();
    auto unrankedTensorType = RankedTensorType::get({}, ET);
    auto newConst = stablehlo::ConstantOp::create(
        builder, rewriteLocation(op->getLoc(), pc.options.strip_llvm_debuginfo),
        unrankedTensorType,
        SplatElementsAttr::get(
            unrankedTensorType,
            ArrayRef<Attribute>(
                isIndex
                    ? IntegerAttr::get(
                          ET, cast<IntegerAttr>(constOp.getValue()).getValue())
                    : constOp.getValueAttr())));
    auto newVal = newConst.getResult();
    mapping.map(constOp.getResult(), newVal);
    maps[newVal] = accessMap;

    return success();
  }

  // Identity
  if (isa<enzymexla::Memref2PointerOp>(op)) {
    Value operand = op->getOperand(0), result = op->getResult(0);
    Value mappedOperand = mapping.lookupOrNull(operand);
    if (!mappedOperand)
      return failure();
    mapping.map(result, mappedOperand);
    return success();
  }

  if (isa<arith::IndexCastUIOp, arith::IndexCastOp>(op)) {
    Value operand = op->getOperand(0), result = op->getResult(0);
    Value mappedResult = mapping.lookupOrNull(operand);
    if (!mappedResult)
      return failure();

    Type targetType = makeIndexToI64(result.getType());
    auto currentType =
        cast<RankedTensorType>(mappedResult.getType()).getElementType();

    if (currentType != targetType) {
      Value newMappedResult =
          stablehlo::ConvertOp::create(
              builder,
              rewriteLocation(op->getLoc(), pc.options.strip_llvm_debuginfo),
              RankedTensorType::get(
                  cast<ShapedType>(mappedResult.getType()).getShape(),
                  targetType),
              mappedResult)
              .getResult();
      maps[newMappedResult] = maps.lookup(mappedResult);
      mappedResult = newMappedResult;
    }

    mapping.map(result, mappedResult);
    return success();
  }

  if (auto apply = dyn_cast<affine::AffineApplyOp>(op)) {
    auto avm = apply.getAffineValueMap();
    // See tryRaisingForOpToStableHLOUnroll
    avm.composeSimplifyAndCanonicalize();
    auto [expanded, expandedMap] = expandAffineExpr(
        builder,
        rewriteLocation(apply.getLoc(), pc.options.strip_llvm_debuginfo),
        avm.getAffineMap().getResult(0), avm.getOperands(), mapping, maps,
        avm.getAffineMap().getNumDims(), pc);
    if (!expanded)
      return failure();
    mapping.map(apply.getResult(), expanded);
    maps[expanded] = expandedMap;
    return success();
  }

  if (auto p2m = dyn_cast<enzymexla::Pointer2MemrefOp>(op)) {
    Value operand = op->getOperand(0), result = op->getResult(0);
    // A view of a buffer that itself is raised gets a value-semantics
    // snapshot here: a store through it would silently diverge from the
    // base. Flattening removes the scratch chains; any store-through view
    // that remains is unsound.
    if (operand.getDefiningOp<enzymexla::Memref2PointerOp>())
      for (Operation *user : result.getUsers())
        if ((isa<affine::AffineStoreOp, memref::StoreOp>(user) &&
             user->getOperand(1) == result))
          return op->emitError(
              "cannot raise a store through an aliasing view");
    auto input = mapping.lookupOrNull(operand);
    if (!input)
      return failure();
    auto MT = p2m.getType();
    if (!isXLACompatiblePrimitive(MT.getElementType())) {
      return op->emitError("unsupported element type for XLA: ")
             << MT.getElementType();
    }

    auto ty = RankedTensorType::get(MT.getShape(), MT.getElementType());

    auto inTy = cast<RankedTensorType>(input.getType());
    size_t outSize =
        cast<AutoDiffTypeInterface>(ty.getElementType()).getApproxSize();
    size_t inSize =
        cast<AutoDiffTypeInterface>(inTy.getElementType()).getApproxSize();

    Value res;
    if (outSize == inSize) {
      res = stablehlo::BitcastConvertOp::create(
          builder,
          rewriteLocation(p2m.getLoc(), pc.options.strip_llvm_debuginfo), ty,
          input);
    } else if (outSize < inSize) {
      SmallVector<int64_t> dims2 = llvm::to_vector(ty.getShape());
      auto oidx = dims2.size();
      dims2.push_back(inSize / outSize);
      if (oidx != 0 && dims2[oidx - 1] != ShapedType::kDynamic) {
        dims2[oidx - 1] /= inSize / outSize;
      }
      res = stablehlo::BitcastConvertOp::create(
          builder,
          rewriteLocation(p2m.getLoc(), pc.options.strip_llvm_debuginfo),
          RankedTensorType::get(dims2, ty.getElementType()), input);
      bool anyDynamic = false;
      for (auto idx : dims2) {
        if (idx == ShapedType::kDynamic) {
          anyDynamic = true;
          break;
        }
      }
      if (anyDynamic) {
        SmallVector<Value> vals;
        for (size_t i = 0; i < ty.getShape().size(); i++) {
          auto val = stablehlo::GetDimensionSizeOp::create(
              builder,
              rewriteLocation(p2m.getLoc(), pc.options.strip_llvm_debuginfo),
              input, i);
          Value vval = val;
          if (i == ty.getShape().size() - 1) {
            auto cst = arith::ConstantOp::create(
                builder,
                rewriteLocation(p2m.getLoc(), pc.options.strip_llvm_debuginfo),
                val.getType(),
                cast<ElementsAttr>(makeAttr(val.getType(), inSize / outSize)));
            vval = stablehlo::MulOp::create(
                builder,
                rewriteLocation(p2m.getLoc(), pc.options.strip_llvm_debuginfo),
                vval, cst);
          }
          vval = stablehlo::ReshapeOp::create(
              builder,
              rewriteLocation(p2m.getLoc(), pc.options.strip_llvm_debuginfo),
              RankedTensorType::get({1}, val.getType().getElementType()), vval);
          vals.push_back(vval);
        }

        auto idxs = stablehlo::ConcatenateOp::create(
            builder,
            rewriteLocation(p2m.getLoc(), pc.options.strip_llvm_debuginfo),
            vals, 0);
        res = stablehlo::DynamicReshapeOp::create(
            builder,
            rewriteLocation(p2m.getLoc(), pc.options.strip_llvm_debuginfo), ty,
            res, idxs);
      } else {
        res = stablehlo::ReshapeOp::create(
            builder,
            rewriteLocation(p2m.getLoc(), pc.options.strip_llvm_debuginfo), ty,
            res);
      }
    } else {
      SmallVector<int64_t> dims2 = llvm::to_vector(ty.getShape());
      auto oidx = dims2.size();
      dims2.push_back(outSize / inSize);
      if (oidx != 0 && dims2[oidx - 1] != ShapedType::kDynamic) {
        dims2[oidx - 1] /= outSize / inSize;
      }
      res = stablehlo::ReshapeOp::create(
          builder,
          rewriteLocation(p2m.getLoc(), pc.options.strip_llvm_debuginfo),
          RankedTensorType::get(dims2, inTy.getElementType()), input);
      res = stablehlo::BitcastConvertOp::create(
          builder,
          rewriteLocation(p2m.getLoc(), pc.options.strip_llvm_debuginfo), ty,
          res);
    }
    mapping.map(result, res);
    return success();
  }

  // unary ops
  if (mlir::isPure(op) && op->hasTrait<OpTrait::Elementwise>() &&
      op->getNumOperands() == 1 && op->getNumResults() == 1) {

    auto operand = op->getOperand(0);
    auto newOperand = mapping.lookupOrNull(operand);
    if (!newOperand)
      return failure();

    auto IT = cast<RankedTensorType>(newOperand.getType());
    auto T = RankedTensorType::get(IT.getShape(),
                                   makeIndexToI64(op->getResult(0).getType()));

    auto newOp = Operation::create(
        rewriteLocation(op->getLoc(), pc.options.strip_llvm_debuginfo),
        op->getName(), {T}, {newOperand}, op->getAttrs(), mlir::PropertyRef(),
        {}, 0);
    mapping.map(op->getResult(0), newOp->getResult(0));
    maps[newOp->getResult(0)] = maps.lookup(newOperand);

    builder.insert(newOp);
    return success();
  }

  // binary ops
  if (isa<arith::MulIOp, arith::MulFOp, arith::AddIOp, arith::AddFOp,
          arith::SubIOp, arith::SubFOp, arith::DivFOp, arith::DivSIOp,
          arith::DivUIOp, arith::OrIOp, arith::AndIOp, arith::XOrIOp,
          arith::CmpIOp, arith::CmpFOp, arith::ShRUIOp, arith::ShRSIOp,
          arith::ShLIOp, arith::MinimumFOp, arith::MaximumFOp, arith::MaxNumFOp,
          arith::MinNumFOp, arith::MinUIOp, arith::MinSIOp, arith::MaxUIOp,
          arith::MaxSIOp, arith::RemSIOp, arith::RemUIOp, arith::RemFOp,
          math::CopySignOp, math::Atan2Op, math::PowFOp>(op)) {
    assert(op->getNumOperands() == 2 && op->getNumResults() == 1);

    Value a = mapping.lookupOrNull(op->getOperand(0)),
          b = mapping.lookupOrNull(op->getOperand(1));
    if (!a || !b)
      return failure();

    auto itA = maps.find(a), itB = maps.find(b);
    if (itA == maps.end() || itB == maps.end())
      return failure();
    bool alignOk = true;
    auto outputMap = alignMemoryAccess(a, itA->second, b, itB->second, builder,
                                       pc, &alignOk);
    if (!alignOk)
      return failure();
    assert(a.getType() == b.getType());

    auto IT = cast<RankedTensorType>(a.getType());
    Type result = RankedTensorType::get(
        IT.getShape(), makeIndexToI64(op->getResult(0).getType()));

    auto newOp = Operation::create(
        rewriteLocation(op->getLoc(), pc.options.strip_llvm_debuginfo),
        op->getName(), {result}, {a, b}, op->getAttrs(), mlir::PropertyRef(),
        {}, 0);

    builder.insert(newOp);

    for (auto [oldRes, newRes] :
         llvm::zip_equal(op->getResults(), newOp->getResults())) {
      mapping.map(oldRes, newRes);
      maps[newRes] = outputMap;
    }

    return success();
  }

  // ternary ops
  if (isa<arith::SelectOp, math::FmaOp, enzymexla::FMulAddOp>(op)) {
    assert(op->getNumOperands() == 3 && op->getNumResults() == 1);

    Value a = mapping.lookupOrNull(op->getOperand(0)),
          b = mapping.lookupOrNull(op->getOperand(1)),
          c = mapping.lookupOrNull(op->getOperand(2));
    if (!a || !b || !c || !maps.count(a) || !maps.count(b) || !maps.count(c))
      return failure();

    auto mapA = maps.lookup(a), mapB = maps.lookup(b), mapC = maps.lookup(c);

    Value dsts[] = {b, c};
    affine::AffineValueMap submaps[] = {mapB, mapC};
    bool alignOk = true;
    auto outputMap =
        alignMemoryAccess(a, mapA, dsts, submaps, builder, pc, &alignOk);
    if (!alignOk)
      return failure();
    b = dsts[0];
    c = dsts[1];
    assert(b.getType() == c.getType());

    Type result = b.getType();

    auto newOp = Operation::create(
        rewriteLocation(op->getLoc(), pc.options.strip_llvm_debuginfo),
        op->getName(), {result}, {a, b, c}, op->getAttrs(), mlir::PropertyRef(),
        {}, 0);

    builder.insert(newOp);

    for (auto [oldRes, newRes] :
         llvm::zip_equal(op->getResults(), newOp->getResults())) {
      mapping.map(oldRes, newRes);
      maps[newRes] = outputMap;
    }

    return success();
  }

  if (auto ifOp = dyn_cast<scf::IfOp>(op)) {

    Value cond = mapping.lookupOrNull(ifOp.getCondition());
    if (!cond || !maps.count(cond))
      return failure();
    if (emitIfAsSelect(op, cond, maps.lookup(cond), builder, mapping, maps, pc)
            .failed())
      return failure();

    return success();
  }

  if (auto ifOp = dyn_cast<affine::AffineIfOp>(op)) {

    auto is = ifOp.getIntegerSet();

    Value cond = nullptr;
    affine::AffineValueMap map(AffineMap::get(ifOp.getContext()), {});

    affine::AffineValueMap constraintMap(
        AffineMap::get(is.getNumDims(), is.getNumSymbols(), is.getConstraints(),
                       is.getContext()),
        ifOp.getOperands());
    constraintMap.composeSimplifyAndCanonicalize();

    for (auto [constraint, eq] : llvm::zip_equal(
             constraintMap.getAffineMap().getResults(), is.getEqFlags())) {
      auto [expandedExpr, outputMap] = expandAffineExpr(
          builder,
          rewriteLocation(ifOp.getLoc(), pc.options.strip_llvm_debuginfo),
          constraint, constraintMap.getOperands(), mapping, maps,
          constraintMap.getNumDims(), pc);
      if (!expandedExpr)
        return failure();
      Value zero = stablehlo::ConstantOp::create(
          builder,
          rewriteLocation(ifOp.getLoc(), pc.options.strip_llvm_debuginfo),
          cast<ShapedType>(expandedExpr.getType()),
          SplatElementsAttr::get(
              cast<ShapedType>(expandedExpr.getType()),
              ArrayRef<Attribute>(IntegerAttr::get(builder.getI64Type(), 0))));
      Value newCond = stablehlo::CompareOp::create(
          builder,
          rewriteLocation(ifOp.getLoc(), pc.options.strip_llvm_debuginfo),
          expandedExpr, zero,
          eq ? stablehlo::ComparisonDirection::EQ
             : stablehlo::ComparisonDirection::GE);
      if (cond) {
        bool alignOk = true;
        map = alignMemoryAccess(cond, map, newCond, outputMap, builder, pc,
                                &alignOk);
        if (!alignOk)
          return failure();
        cond = stablehlo::AndOp::create(
            builder,
            rewriteLocation(ifOp.getLoc(), pc.options.strip_llvm_debuginfo),
            cond, newCond);
      } else {
        cond = newCond;
        map = outputMap;
      }
    }

    maps[cond] = map;

    if (emitIfAsSelect(op, cond, map, builder, mapping, maps, pc).failed())
      return failure();

    return success();
  }

  // Inner parallel for loops
  if (auto parallelOp = dyn_cast<affine::AffineParallelOp>(op)) {
    return tryRaisingParallelOpToStableHLO(parallelOp, mapping, builder, maps,
                                           pc);
  }

  // Inner for op
  if (auto forOp = dyn_cast<affine::AffineForOp>(op)) {
    if (pc.options.enableLockstepFor && forOp.hasConstantBounds() &&
        tryRaisingLockStepForOpToStableHLO(forOp, mapping, builder, maps, pc)
            .succeeded()) {
      return success();
    }
    // Nested constant-bound loops unroll multiplicatively (a generic
    // max-degree kernel reaches millions of ops and gigabytes of module
    // text); past a budget, iterating as a while beats unrolling even when
    // the preference says otherwise.
    std::function<int64_t(Operation *)> unrollCost =
        [&](Operation *o) -> int64_t {
      int64_t body = 1;
      for (Region &r : o->getRegions())
        for (Block &b : r)
          for (Operation &inner : b)
            body = std::min(body + unrollCost(&inner),
                            (int64_t)1 << 40);
      if (auto f = dyn_cast<affine::AffineForOp>(o)) {
        if (!f.hasConstantBounds())
          return body;
        int64_t step = f.getStepAsInt();
        int64_t trip =
            (f.getConstantUpperBound() - f.getConstantLowerBound() + step - 1) /
            step;
        return std::min(std::max(trip, (int64_t)1) * body, (int64_t)1 << 40);
      }
      return body;
    };
    bool hugeUnroll = forOp.hasConstantBounds() &&
                      unrollCost(forOp.getOperation()) > (1 << 16);
    // A loop whose trip count is only known at runtime can still iterate as
    // a while, whatever the preference says.
    if ((pc.options.preferWhileRaising || !forOp.hasConstantBounds() ||
         hugeUnroll) &&
        tryRaisingForOpToStableHLOWhile(forOp, mapping, builder, maps, pc)
            .succeeded()) {
      return success();
    }
    if (tryRaisingForOpToStableHLOUnroll(forOp, mapping, builder, maps, pc)
            .succeeded()) {
      return success();
    }
    // The preference skipped the while for constant bounds; as a last
    // resort a sequential while still raises what lockstep and unrolling
    // could not (accumulators chained through nested reductions).
    if (!pc.options.preferWhileRaising && forOp.hasConstantBounds() &&
        tryRaisingForOpToStableHLOWhile(forOp, mapping, builder, maps, pc)
            .succeeded()) {
      return success();
    }
    if (tryRaisingAffineForOpToMaskedWhile(forOp, mapping, builder, maps, pc)
            .succeeded()) {
      return success();
    }
  }

  if (auto scfFor = dyn_cast<scf::ForOp>(op)) {
    if (tryRaisingSCFForOpToStableHLOWhile(scfFor, mapping, builder, maps, pc)
            .succeeded())
      return success();
    if (tryRaisingSCFForOpToMaskedWhile(scfFor, mapping, builder, maps, pc)
            .succeeded())
      return success();
  }

  if (auto scfWhile = dyn_cast<scf::WhileOp>(op)) {
    if (tryRaisingSCFWhileOpToStableHLO(scfWhile, mapping, builder, maps, pc)
            .succeeded())
      return success();
  }

  if (auto alloca = dyn_cast<memref::AllocaOp>(op)) {
    // Kernel scratch: a fresh buffer per iteration of whatever loop holds
    // it, which is exactly what materializing its initial value where the
    // alloca sits gives. Reads before any write see zeros.
    auto MT = alloca.getType();
    if (!isXLACompatiblePrimitive(MT.getElementType()))
      return op->emitError("cannot raise dynamic or non-primitive alloca")
             << *op;
    if (!MT.hasStaticShape()) {
      // A scratch of runtime extent: a dynamic broadcast of the zero makes
      // the buffer, its shape assembled from the mapped extents.
      auto loc =
          rewriteLocation(op->getLoc(), pc.options.strip_llvm_debuginfo);
      SmallVector<Value> dims;
      unsigned dynIdx = 0;
      auto i64Ty = RankedTensorType::get({}, builder.getI64Type());
      auto i64x1Ty = RankedTensorType::get({1}, builder.getI64Type());
      for (int64_t d = 0; d < MT.getRank(); ++d) {
        Value dv;
        if (MT.isDynamicDim(d)) {
          if (dynIdx >= alloca.getDynamicSizes().size())
            return op->emitError("cannot raise dynamic or non-primitive "
                                 "alloca")
                   << *op;
          Value szv =
              mapping.lookupOrNull(alloca.getDynamicSizes()[dynIdx++]);
          if (!szv || cast<RankedTensorType>(szv.getType()).getRank() != 0)
            return op->emitError("cannot raise dynamic or non-primitive "
                                 "alloca")
                   << *op;
          if (!cast<RankedTensorType>(szv.getType())
                   .getElementType()
                   .isInteger(64))
            szv = stablehlo::ConvertOp::create(builder, loc, i64Ty, szv);
          dv = szv;
        } else {
          dv = stablehlo::ConstantOp::create(
              builder, loc, i64Ty,
              SplatElementsAttr::get(
                  i64Ty, builder.getI64IntegerAttr(MT.getDimSize(d))));
        }
        dims.push_back(
            stablehlo::ReshapeOp::create(builder, loc, i64x1Ty, dv));
      }
      Value shape =
          dims.size() == 1
              ? dims[0]
              : stablehlo::ConcatenateOp::create(builder, loc, dims, 0)
                    .getResult();
      auto ST = RankedTensorType::get({}, MT.getElementType());
      Value zeroScalar = stablehlo::ConstantOp::create(
          builder, loc, ST,
          SplatElementsAttr::get(ST, builder.getZeroAttr(MT.getElementType())));
      SmallVector<int64_t> dynShape(MT.getRank(), ShapedType::kDynamic);
      for (int64_t d = 0; d < MT.getRank(); ++d)
        if (!MT.isDynamicDim(d))
          dynShape[d] = MT.getDimSize(d);
      auto TT = RankedTensorType::get(dynShape, MT.getElementType());
      Value dyn = stablehlo::DynamicBroadcastInDimOp::create(
          builder, loc, TT, zeroScalar, shape,
          builder.getDenseI64ArrayAttr({}));
      mapping.map(alloca.getResult(), dyn);
      maps[dyn] = affine::AffineValueMap(AffineMap::get(op->getContext()), {});
      return success();
    }
    auto TT = RankedTensorType::get(MT.getShape(), MT.getElementType());
    Value zero = stablehlo::ConstantOp::create(
        builder, rewriteLocation(op->getLoc(), pc.options.strip_llvm_debuginfo),
        TT,
        SplatElementsAttr::get(TT, builder.getZeroAttr(MT.getElementType())));
    mapping.map(alloca.getResult(), zero);
    maps[zero] = affine::AffineValueMap(AffineMap::get(op->getContext()), {});
    return success();
  }

  if (isa<LLVM::NoAliasScopeDeclOp>(op)) {
    return success();
  }

  // A shape-erasing cast of an already-raised buffer is the identity on the
  // underlying tensor.
  if (auto castOp = dyn_cast<memref::CastOp>(op)) {
    Value src = mapping.lookupOrNull(castOp.getSource());
    if (src && maps.count(src)) {
      mapping.map(castOp.getResult(), src);
      maps[src] = maps.lookup(src);
      return success();
    }
  }

  // Raised execution is ordered over whole tensors: a store over a batched
  // thread axis completes for the entire axis before the next op runs, which
  // is exactly what the barrier guaranteed.
  if (isa<enzymexla::BarrierOp>(op)) {
    // The no-op treatment is only sound for the axes the barrier spans when
    // they are batched: an induction variable of a dynamically sized loop
    // raises serialized, with no whole-tensor ordering to lean on.
    for (Value iv : op->getOperands()) {
      auto ba = dyn_cast<BlockArgument>(iv);
      if (!ba)
        continue;
      Operation *owner = ba.getOwner()->getParentOp();
      if (auto par = dyn_cast<affine::AffineParallelOp>(owner)) {
        if (!par.getConstantRanges()) {
          if (getenv("DEBUG_BOUND")) {
            llvm::errs() << "dynamic barrier axis " << ba.getArgNumber()
                         << " ubmap " << par.getUpperBoundsMap() << " ops:\n";
            for (Value o : par.getUpperBoundsOperands()) {
              llvm::errs() << "  operand: " << o << "\n";
              Value w = o;
              for (int k = 0; k < 6 && w.getDefiningOp(); ++k) {
                llvm::errs() << "    <- " << *w.getDefiningOp() << "\n";
                if (w.getDefiningOp()->getNumOperands() == 0)
                  break;
                w = w.getDefiningOp()->getOperand(0);
              }
            }
          }
          return op->emitError(
              "barrier over a dynamically sized parallel axis");
        }
        continue;
      }
      if (auto forOwner = dyn_cast<affine::AffineForOp>(owner)) {
        // A constant-trip for raises in lockstep with whole-tensor ordering,
        // making the barrier a no-op exactly like a batched parallel axis.
        if (forOwner.hasConstantBounds())
          continue;
        return op->emitError(
            "barrier over a dynamically sized parallel axis");
      }
      if (isa<scf::ForOp, scf::WhileOp>(owner))
        return op->emitError(
            "barrier over a dynamically sized parallel axis");
    }
    return success();
  }

  return op->emitError("cannot raise op to stablehlo") << *op;
}

static void
replaceAffineFuncWithStableHLOFunc(func::FuncOp oldFunc, func::FuncOp newFunc,
                                   llvm::ArrayRef<Operation *> users,
                                   bool strip_llvm_debuginfo) {
  for (auto op : users) {
    auto user = dyn_cast<enzymexla::JITCallOp>(op);

    OpBuilder builder(user);
    auto newCall = func::CallOp::create(
        builder, rewriteLocation(user->getLoc(), strip_llvm_debuginfo), newFunc,
        user->getOperands());

    auto operand_aliases = user.getOutputOperandAliases();
    assert(operand_aliases.size() == user.getNumResults());

    SmallVector<Value> replacements;
    for (auto alias_attr : operand_aliases) {
      auto alias = cast<mlir::stablehlo::OutputOperandAliasAttr>(alias_attr);
      auto operandIndex = alias.getOperandIndex();
      replacements.push_back(newCall.getResult(operandIndex));
    }

    for (auto [oldRes, newRes] :
         llvm::zip_equal(user->getResults(), replacements)) {
      oldRes.replaceAllUsesWith(newRes);
    }

    user->erase();
  }

  oldFunc->erase();
}

static bool tryRaisingToStableHLO(func::FuncOp func,
                                  ArrayRef<Operation *> users,
                                  ParallelContext::Options &options) {
  Block *body = &func->getRegion(0).front();
  Block *newBlock = new Block();

  IRMapping mapping;
  mapping.map(body, newBlock);

  SmallVector<Type> tensorTypes;
  for (auto arg : body->getArguments()) {
    auto MT = cast<MemRefType>(arg.getType());
    if (!isXLACompatiblePrimitive(MT.getElementType())) {
      func.emitError("unsupported element type for argument for XLA: ")
          << MT.getElementType();
      delete newBlock;
      return false;
    }
    auto TT = RankedTensorType::get(MT.getShape(), MT.getElementType());
    auto newArg = newBlock->addArgument(
        TT, rewriteLocation(arg.getLoc(), options.strip_llvm_debuginfo));
    mapping.map(arg, newArg);
    tensorTypes.push_back(TT);
  }

  auto modOp = func->getParentOfType<ModuleOp>();

  auto newFuncType =
      FunctionType::get(func->getContext(), tensorTypes, tensorTypes);

  std::string name = func.getSymName().str();
  name += "_raised";

  auto newFunc = func::FuncOp::create(
      rewriteLocation(func->getLoc(), options.strip_llvm_debuginfo), name,
      newFuncType);
  newFunc.setVisibility(mlir::SymbolTable::Visibility::Private);
  newFunc.getBody().push_back(newBlock);

  OpBuilder builder(newBlock, newBlock->end());

  bool anyFailed = false;

  llvm::DenseMap<Value, affine::AffineValueMap> maps;

  ParallelContext emptyPc = ParallelContext::getEmpty(options);
  for (auto &it : body->without_terminator()) {
    anyFailed =
        tryRaisingOpToStableHLO(&it, mapping, builder, maps, emptyPc).failed();
    if (anyFailed)
      break;
  }

  if (anyFailed) {
    newFunc->erase();
    return false;
  }

  SmallVector<Value> results;
  for (auto arg : body->getArguments()) {
    auto val = mapping.lookup(arg);
    results.push_back(val);
  }

  func::ReturnOp::create(
      builder, rewriteLocation(func->getLoc(), options.strip_llvm_debuginfo),
      results);
  modOp.getBody()->push_back(newFunc);

  replaceAffineFuncWithStableHLOFunc(func, newFunc, users,
                                     options.strip_llvm_debuginfo);

  return true;
}

namespace {

// select(!op, lhs, rhs) --> select(op, rhs, lhs)
struct PushReductionsDown : public OpRewritePattern<arith::AddFOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::AddFOp op,
                                PatternRewriter &rewriter) const final {
    for (int i = 0; i < 2; i++) {
      auto lhs = op->getOperand(i).getDefiningOp<arith::AddFOp>();
      if (!lhs)
        continue;
      auto rhs = op->getOperand(1 - i);
      if (auto ba0 = dyn_cast<BlockArgument>(rhs)) {
        if (isa<affine::AffineForOp>(ba0.getOwner()->getParentOp()))
          continue;
      }
      for (int j = 0; j < 2; j++) {
        auto ba = dyn_cast<BlockArgument>(lhs->getOperand(j));
        if (!ba)
          continue;
        if (!isa<affine::AffineForOp>(ba.getOwner()->getParentOp()))
          continue;
        auto add2 =
            arith::AddFOp::create(rewriter, rewriteLocation(op.getLoc(), false),
                                  rhs, lhs->getOperand(1 - j));
        rewriter.replaceOpWithNewOp<arith::AddFOp>(op, add2, ba);
        return success();
      }
    }
    return failure();
  }
};

struct AffineToStableHLORaisingPass
    : public enzyme::impl::AffineToStableHLORaisingBase<
          AffineToStableHLORaisingPass> {
  using AffineToStableHLORaisingBase::AffineToStableHLORaisingBase;

  // An access does not care about the address space of its base, but the
  // raising identifies buffers by SSA root: a memory_space_cast view would
  // split one buffer into two roots and lose store propagation. Retarget the
  // accesses to the source and drop the cast.
  // An alloca scope only delimits stack lifetime, which the raised value
  // semantics make meaningless: splice its body into the parent.
  // Straight-line CFG inside a cloned callee region folds into one block,
  // so the scope inlining below can dissolve it.
  static void linearizeRegionBlocks(Region &r) {
    auto isTrapBlock = [](Block *b) {
      return isa<LLVM::UnreachableOp>(b->getTerminator());
    };
    bool changed = true;
    while (changed) {
      changed = false;
      // Branches whose other targets only trap take their one live successor
      // unconditionally.
      for (Block &b : r) {
        Operation *term = b.getTerminator();
        SmallVector<std::pair<Block *, SmallVector<Value>>> live;
        bool anyTrap = false;
        if (isa<cf::SwitchOp, cf::CondBranchOp>(term)) {
          for (auto [i, succ] : llvm::enumerate(term->getSuccessors())) {
            if (isTrapBlock(succ)) {
              anyTrap = true;
              continue;
            }
            auto sops =
                cast<BranchOpInterface>(term).getSuccessorOperands(i);
            SmallVector<Value> args(sops.getForwardedOperands().begin(),
                                    sops.getForwardedOperands().end());
            live.push_back({succ, std::move(args)});
          }
        }
        if (anyTrap && live.size() == 1) {
          OpBuilder tb(term);
          cf::BranchOp::create(tb, term->getLoc(), live[0].first,
                               live[0].second);
          term->erase();
          changed = true;
          break;
        }
      }
      if (changed)
        continue;
      // Trap blocks with no remaining predecessors disappear.
      for (Block &b : llvm::make_early_inc_range(r)) {
        if (&b != &r.front() && b.hasNoPredecessors()) {
          b.dropAllDefinedValueUses();
          b.erase();
          changed = true;
        }
      }
      if (changed)
        continue;
      for (Block &b : r) {
        Operation *term = b.getTerminator();
        Block *succ = nullptr;
        SmallVector<Value> args;
        if (auto br = dyn_cast<cf::BranchOp>(term)) {
          succ = br.getDest();
          args.assign(br.getDestOperands().begin(),
                      br.getDestOperands().end());
        } else if (auto br = dyn_cast<LLVM::BrOp>(term)) {
          succ = br.getDest();
          args.assign(br.getDestOperands().begin(),
                      br.getDestOperands().end());
        } else {
          continue;
        }
        if (!succ || succ == &b || succ->getSinglePredecessor() != &b)
          continue;
        for (auto [ba, v] : llvm::zip(succ->getArguments(), args))
          ba.replaceAllUsesWith(v);
        term->erase();
        b.getOperations().splice(b.end(), succ->getOperations());
        succ->erase();
        changed = true;
        break;
      }
    }
  }

  static void inlineAllocaScopes(Operation *g) {
    // Inliner wrappers stack alloca_scope/execute_region pairs, so inlining
    // one can expose another: iterate to a fixed point.
    bool changed = true;
    while (changed) {
      changed = false;
      SmallVector<Operation *> scopes;
      g->walk([&](Operation *op) {
        if (isa<memref::AllocaScopeOp, scf::ExecuteRegionOp>(op))
          scopes.push_back(op);
      });
      for (Operation *sc : scopes) {
        Region &r = sc->getRegion(0);
        if (!r.hasOneBlock())
          linearizeRegionBlocks(r);
        if (!r.hasOneBlock()) {
          if (getenv("DEBUG_SCOPES")) {
            llvm::errs() << "scope multiblock after linearize: "
                         << std::distance(r.begin(), r.end()) << " blocks;";
            for (Block &b : r)
              llvm::errs() << " term=" << b.getTerminator()->getName()
                           << " preds="
                           << std::distance(b.pred_begin(), b.pred_end());
            llvm::errs() << "\n";
          }
          continue;
        }
        Block *body = &r.front();
        Operation *term = body->getTerminator();
        for (auto [res, yielded] :
             llvm::zip(sc->getResults(), term->getOperands()))
          res.replaceAllUsesWith(yielded);
        term->erase();
        sc->getBlock()->getOperations().splice(sc->getIterator(),
                                               body->getOperations());
        sc->erase();
        changed = true;
      }
    }
  }

  // A barrier under a parallel axis of dynamic extent raises serialized, so
  // it cannot be dropped as a no-op: distribute the loops around it first,
  // cpuify-style, so every pre-barrier phase completes for the whole axis
  // before the next phase starts.
  static void distributeSerializedBarriers(Operation *g) {
    bool need = false;
    g->walk([&](enzymexla::BarrierOp b) {
      for (Operation *p = b->getParentOp(); p && p != g;
           p = p->getParentOp())
        if (auto ap = dyn_cast<affine::AffineParallelOp>(p))
          if (!ap.getConstantRanges())
            need = true;
    });
    if (need)
      (void)enzymexla::distributeAroundBarriers(g);
  }

  // A pure scalar computed entirely from values defined outside the wrapper
  // (a null check of an optional buffer, a host-side flag chain) is the
  // host's to compute: hoist it out, so the kernel captures the resulting
  // scalar instead of pointers no tensor can stand for.
  static void hoistWrapperInvariantScalars(Operation *g) {
    auto definedOutside = [&](Value v) {
      if (auto ba = dyn_cast<BlockArgument>(v))
        return !g->isProperAncestor(ba.getOwner()->getParentOp()) &&
               ba.getOwner()->getParentOp() != g;
      return !g->isProperAncestor(v.getDefiningOp());
    };
    bool changed = true;
    while (changed) {
      changed = false;
      SmallVector<Operation *> toHoist;
      g->walk([&](Operation *op) {
        if (op->getNumOperands() == 0 || op->getNumRegions() ||
            op->hasTrait<OpTrait::IsTerminator>() || !isMemoryEffectFree(op))
          return;
        if (!llvm::all_of(op->getResultTypes(),
                          [](Type t) { return t.isIntOrIndexOrFloat(); }))
          return;
        if (!llvm::all_of(op->getOperands(), definedOutside))
          return;
        toHoist.push_back(op);
      });
      for (Operation *op : toHoist) {
        op->moveBefore(g);
        changed = true;
      }
    }
  }

  // A view taken of a gep result pins the kernel operand to the gep, which
  // no tensor can stand for. Rebase the view onto the underlying pointer and
  // fold the gep's element offset into each access index; data-dependent
  // offsets make the accesses plain memref ops, which raising gathers.
  static void rebaseViewedGeps(Operation *root) {
    SmallVector<LLVM::GEPOp> geps;
    root->walk([&](LLVM::GEPOp g) { geps.push_back(g); });
    for (auto gep : geps) {
      auto idxs = gep.getIndices();
      if (idxs.size() != 1)
        continue;
      DataLayout dl = DataLayout::closest(gep);
      int64_t elemSz = dl.getTypeSize(gep.getElemType());
      bool ok = true;
      SmallVector<enzymexla::Pointer2MemrefOp> views;
      for (Operation *u : gep->getUsers()) {
        auto p2m = dyn_cast<enzymexla::Pointer2MemrefOp>(u);
        if (!p2m || p2m.getType().getRank() != 1 ||
            !p2m.getType().getElementType().isIntOrFloat() ||
            (int64_t)dl.getTypeSize(p2m.getType().getElementType()) !=
                elemSz) {
          ok = false;
          break;
        }
        for (Operation *a : p2m->getUsers()) {
          if (auto ld = dyn_cast<affine::AffineLoadOp>(a)) {
            if (ld.getMap().getNumResults() == 1)
              continue;
          } else if (auto st = dyn_cast<affine::AffineStoreOp>(a)) {
            if (st.getMap().getNumResults() == 1 &&
                st.getValueToStore() != p2m.getResult())
              continue;
          } else if (auto ld = dyn_cast<memref::LoadOp>(a)) {
            if (ld.getIndices().size() == 1)
              continue;
          } else if (auto st = dyn_cast<memref::StoreOp>(a)) {
            if (st.getIndices().size() == 1 &&
                st.getValueToStore() != p2m.getResult())
              continue;
          }
          ok = false;
          break;
        }
        if (!ok)
          break;
        views.push_back(p2m);
      }
      if (!ok || views.empty())
        continue;
      OpBuilder gb(gep);
      Location loc = gep.getLoc();
      Value off;
      if (!gep.getDynamicIndices().empty()) {
        off = gep.getDynamicIndices()[0];
        if (!isa<IndexType>(off.getType()))
          off = arith::IndexCastOp::create(gb, loc, gb.getIndexType(), off);
      } else {
        off = arith::ConstantIndexOp::create(
            gb, loc, cast<IntegerAttr>(idxs[0]).getInt());
      }
      for (auto p2m : views) {
        OpBuilder vb(p2m);
        Value newView = enzymexla::Pointer2MemrefOp::create(
            vb, p2m.getLoc(), p2m.getType(), gep.getBase());
        for (Operation *a : llvm::make_early_inc_range(p2m->getUsers())) {
          OpBuilder ab(a);
          auto toIdx = [&](AffineMap map, ValueRange operands) -> Value {
            auto expanded =
                affine::expandAffineMap(ab, a->getLoc(), map, operands);
            return (*expanded)[0];
          };
          if (auto ld = dyn_cast<affine::AffineLoadOp>(a)) {
            Value idx = toIdx(ld.getMap(), ld.getMapOperands());
            idx = arith::AddIOp::create(ab, a->getLoc(), idx, off);
            Value nl = memref::LoadOp::create(ab, a->getLoc(), newView,
                                              ValueRange{idx});
            a->getResult(0).replaceAllUsesWith(nl);
            a->erase();
          } else if (auto st = dyn_cast<affine::AffineStoreOp>(a)) {
            Value idx = toIdx(st.getMap(), st.getMapOperands());
            idx = arith::AddIOp::create(ab, a->getLoc(), idx, off);
            memref::StoreOp::create(ab, a->getLoc(), st.getValueToStore(),
                                    newView, ValueRange{idx});
            a->erase();
          } else if (auto ld = dyn_cast<memref::LoadOp>(a)) {
            Value idx = arith::AddIOp::create(ab, a->getLoc(),
                                              ld.getIndices()[0], off);
            Value nl = memref::LoadOp::create(ab, a->getLoc(), newView,
                                              ValueRange{idx});
            a->getResult(0).replaceAllUsesWith(nl);
            a->erase();
          } else {
            auto st = cast<memref::StoreOp>(a);
            Value idx = arith::AddIOp::create(ab, a->getLoc(),
                                              st.getIndices()[0], off);
            memref::StoreOp::create(ab, a->getLoc(), st.getValueToStore(),
                                    newView, ValueRange{idx});
            a->erase();
          }
        }
        p2m.erase();
      }
      if (gep->use_empty())
        gep.erase();
    }
  }

  // Data-dependent indexing (CSR-style loops over runtime offsets) can never
  // become affine, so llvm-to-affine-access leaves it as raw gep+load. The
  // access still addresses whole elements of the loaded type; a plain memref
  // access through a flat view carries that, and raising gathers it.
  static void convertRawGepAccesses(Operation *root) {
    SmallVector<Operation *> accesses;
    root->walk([&](Operation *op) {
      if (isa<LLVM::LoadOp, LLVM::StoreOp>(op))
        accesses.push_back(op);
    });
    for (Operation *op : accesses) {
      bool isLoad = isa<LLVM::LoadOp>(op);
      if (isLoad ? cast<LLVM::LoadOp>(op).getVolatile_()
                 : cast<LLVM::StoreOp>(op).getVolatile_())
        continue;
      Value addr = isLoad ? op->getOperand(0) : op->getOperand(1);
      Type valTy = isLoad ? op->getResult(0).getType()
                          : op->getOperand(0).getType();
      if (!valTy.isIntOrFloat())
        continue;
      DataLayout dl = DataLayout::closest(op);
      Value base;
      Value dynIdx;
      int64_t constIdx = 0;
      if (auto gep = addr.getDefiningOp<LLVM::GEPOp>()) {
        auto idxs = gep.getIndices();
        if (idxs.size() != 1 ||
            (int64_t)dl.getTypeSize(gep.getElemType()) !=
                (int64_t)dl.getTypeSize(valTy))
          continue;
        base = gep.getBase();
        if (!gep.getDynamicIndices().empty())
          dynIdx = gep.getDynamicIndices()[0];
        else
          constIdx = cast<IntegerAttr>(idxs[0]).getInt();
      } else {
        continue;
      }
      auto basePtrTy = cast<LLVM::LLVMPointerType>(base.getType());
      Attribute space;
      if (basePtrTy.getAddressSpace() != 0)
        space = IntegerAttr::get(IntegerType::get(op->getContext(), 64),
                                 basePtrTy.getAddressSpace());
      OpBuilder b(op);
      Location loc = op->getLoc();
      auto MT = MemRefType::get({ShapedType::kDynamic}, valTy,
                                MemRefLayoutAttrInterface{}, space);
      Value view = enzymexla::Pointer2MemrefOp::create(b, loc, MT, base);
      Value idx;
      if (dynIdx) {
        idx = dynIdx;
        if (!isa<IndexType>(idx.getType()))
          idx = arith::IndexCastOp::create(b, loc, b.getIndexType(), idx);
      } else {
        idx = arith::ConstantIndexOp::create(b, loc, constIdx);
      }
      if (isLoad) {
        Value ld =
            memref::LoadOp::create(b, loc, view, ValueRange{idx});
        op->getResult(0).replaceAllUsesWith(ld);
        op->erase();
      } else {
        memref::StoreOp::create(b, loc, op->getOperand(0), view,
                                ValueRange{idx});
        op->erase();
      }
    }
  }

  // Access rewrites leave dead pointer plumbing behind, and raising visits
  // every op in the region: sweep the unused chains.
  static void dropDeadPointerChains(Operation *root) {
    bool changed = true;
    while (changed) {
      changed = false;
      SmallVector<Operation *> dead;
      root->walk([&](Operation *op) {
        if (isa<LLVM::GEPOp, LLVM::AddrSpaceCastOp,
                enzymexla::Pointer2MemrefOp, enzymexla::Memref2PointerOp>(
                op) &&
            op->use_empty())
          dead.push_back(op);
        // Access expansion strands the pointer selects it distributed into.
        else if (auto sel = dyn_cast<arith::SelectOp>(op))
          if (sel.use_empty() &&
              isa<LLVM::LLVMPointerType, MemRefType>(sel.getType()))
            dead.push_back(op);
      });
      for (Operation *op : dead) {
        op->erase();
        changed = true;
      }
    }
  }

  // Whether the pointer's value is only ever consumed as an address of a
  // memory access (through geps, casts, further selects, or memref views):
  // anything that observes the value itself — a comparison, an int cast, a
  // call, a store of the pointer as data — disqualifies it.
  static bool onlyAddressesMemory(Value v) {
    for (OpOperand &use : v.getUses()) {
      Operation *u = use.getOwner();
      if (auto gep = dyn_cast<LLVM::GEPOp>(u)) {
        if (use.get() != gep.getBase() ||
            !onlyAddressesMemory(gep.getResult()))
          return false;
      } else if (isa<LLVM::AddrSpaceCastOp>(u)) {
        if (!onlyAddressesMemory(u->getResult(0)))
          return false;
      } else if (auto sel = dyn_cast<arith::SelectOp>(u)) {
        if (use.get() == sel.getCondition() ||
            !onlyAddressesMemory(sel.getResult()))
          return false;
      } else if (auto p2m = dyn_cast<enzymexla::Pointer2MemrefOp>(u)) {
        for (Operation *mu : p2m->getUsers())
          if (!isa<affine::AffineLoadOp, affine::AffineStoreOp,
                   memref::LoadOp, memref::StoreOp, memref::AtomicRMWOp>(mu))
            return false;
      } else if (isa<LLVM::LoadOp>(u)) {
      } else if (auto store = dyn_cast<LLVM::StoreOp>(u)) {
        if (use.get() == store.getValue())
          return false;
      } else if (auto rmw = dyn_cast<LLVM::AtomicRMWOp>(u)) {
        if (use.get() != rmw.getPtr())
          return false;
      } else {
        return false;
      }
    }
    return true;
  }

  // mfem's Read/Write staging helpers return null for empty buffers, so a
  // captured device pointer arrives as `select(size > 0, ptr, null)`. When
  // the pointer is only dereferenced, the null arm can only fault, so the
  // select collapses to the real pointer.
  static void dropNullPointerSelects(Operation *root) {
    SmallVector<arith::SelectOp> sels;
    root->walk([&](arith::SelectOp s) {
      if (isa<LLVM::LLVMPointerType>(s.getType()))
        sels.push_back(s);
    });
    // The null may hide behind offset arithmetic: `select(p, gep(buf, i),
    // gep(null, i))` still only ever dereferences the real buffer.
    auto isNullDerived = [](Value v) {
      while (true) {
        if (v.getDefiningOp<LLVM::ZeroOp>())
          return true;
        if (auto gep = v.getDefiningOp<LLVM::GEPOp>())
          v = gep.getBase();
        else if (auto c = v.getDefiningOp<LLVM::AddrSpaceCastOp>())
          v = c.getArg();
        else
          return false;
      }
    };
    for (auto s : sels) {
      Value tv = s.getTrueValue(), fv = s.getFalseValue();
      bool tNull = isNullDerived(tv);
      bool fNull = isNullDerived(fv);
      if (tNull == fNull)
        continue;
      if (!onlyAddressesMemory(s.getResult()))
        continue;
      s.getResult().replaceAllUsesWith(tNull ? fv : tv);
      s.erase();
    }
  }

  // An empty optional buffer arrives as a null base pointer used directly
  // (not through a select): every access through it sits on a path that
  // can only fault, so loads read as zero and stores vanish, and the null
  // never has to become a kernel argument.
  static void dropNullBufferAccesses(Operation *root) {
    SmallVector<LLVM::ZeroOp> zeros;
    root->walk([&](LLVM::ZeroOp z) {
      if (isa<LLVM::LLVMPointerType>(z.getType()))
        zeros.push_back(z);
    });
    for (auto z : zeros) {
      // Collect views whose base is provably the null pointer. Unrelated
      // users of the null (a null check, a select) are left alone: dropping
      // a dereference of null is sound no matter what else observes it.
      SmallVector<Value> work{z.getResult()};
      SmallVector<enzymexla::Pointer2MemrefOp> views;
      while (!work.empty()) {
        Value v = work.pop_back_val();
        for (Operation *u : v.getUsers()) {
          if (auto gep = dyn_cast<LLVM::GEPOp>(u)) {
            if (gep.getBase() == v)
              work.push_back(gep.getResult());
          } else if (isa<LLVM::AddrSpaceCastOp>(u)) {
            work.push_back(u->getResult(0));
          } else if (auto p2m = dyn_cast<enzymexla::Pointer2MemrefOp>(u)) {
            views.push_back(p2m);
          }
        }
      }
      for (auto p2m : views) {
        bool allAccesses = llvm::all_of(p2m->getUsers(), [&](Operation *a) {
          return isa<affine::AffineLoadOp, memref::LoadOp>(a) ||
                 (isa<affine::AffineStoreOp, memref::StoreOp>(a) &&
                  a->getOperand(0) != p2m.getResult());
        });
        if (!allAccesses)
          continue;
        for (Operation *a : llvm::make_early_inc_range(p2m->getUsers())) {
          if (isa<affine::AffineLoadOp, memref::LoadOp>(a)) {
            OpBuilder b(a);
            Type ty = a->getResult(0).getType();
            Value zc = arith::ConstantOp::create(b, a->getLoc(),
                                                 b.getZeroAttr(ty));
            a->getResult(0).replaceAllUsesWith(zc);
          }
          a->erase();
        }
        p2m.erase();
      }
    }
  }

  // A loop that walks a pointer forward by a constant stride each iteration
  // carries it as an iter arg the raising cannot type as a tensor. The
  // pointer is a pure function of the induction variable, so accesses
  // through it rebase onto the init pointer at `orig + k*stride` and the
  // carried pointer disappears from the loop.
  static void rewritePointerInduction(Operation *root) {
    SmallVector<affine::AffineForOp> fors;
    // Post-order walk: inner loops rewrite before the outer loops that
    // contain them.
    root->walk([&](affine::AffineForOp f) {
      if (f.getNumIterOperands() > 0)
        fors.push_back(f);
    });
    if (getenv("DEBUG_PTRIND"))
      llvm::errs() << "PTRIND: " << fors.size() << " candidate fors\n";
    for (auto f : fors) {
      if (f.getStepAsInt() != 1 || f.getLowerBoundMap().getNumResults() != 1)
        continue;
      auto yield = cast<affine::AffineYieldOp>(f.getBody()->getTerminator());
      DataLayout dl = DataLayout::closest(f);
      unsigned n = f.getNumIterOperands();
      SmallVector<bool> drop(n, false);
      struct Induction {
        unsigned idx;
        int64_t byteStep;
        LLVM::GEPOp advance;
      };
      SmallVector<Induction> inductions;
      bool ok = true;
      bool anyPtr = false;
      for (unsigned i = 0; ok && i < n; ++i) {
        Value init = f.getInits()[i];
        if (!isa<LLVM::LLVMPointerType>(init.getType()))
          continue;
        anyPtr = true;
        if (!f.getResult(i).use_empty()) {
          ok = false;
          break;
        }
        BlockArgument arg = f.getRegionIterArgs()[i];
        if (arg.use_empty()) {
          drop[i] = true;
          continue;
        }
        auto gep = yield.getOperand(i).getDefiningOp<LLVM::GEPOp>();
        if (!gep || gep.getBase() != arg || !gep.getDynamicIndices().empty() ||
            gep.getIndices().size() != 1) {
          ok = false;
          break;
        }
        int64_t byteStep = cast<IntegerAttr>(gep.getIndices()[0]).getInt() *
                           (int64_t)dl.getTypeSize(gep.getElemType());
        for (Operation *u : arg.getUsers()) {
          if (u == gep.getOperation())
            continue;
          auto p2m = dyn_cast<enzymexla::Pointer2MemrefOp>(u);
          if (!p2m || p2m.getType().getRank() != 1 ||
              !p2m.getType().getElementType().isIntOrFloat() ||
              byteStep % (int64_t)dl.getTypeSize(
                              p2m.getType().getElementType()) !=
                  0) {
            ok = false;
            break;
          }
          for (Operation *a : p2m->getUsers()) {
            if (auto ld = dyn_cast<affine::AffineLoadOp>(a)) {
              if (ld.getMap().getNumResults() == 1)
                continue;
            } else if (auto st = dyn_cast<affine::AffineStoreOp>(a)) {
              if (st.getMap().getNumResults() == 1 &&
                  st.getValueToStore() != p2m.getResult())
                continue;
            } else if (auto ld = dyn_cast<memref::LoadOp>(a)) {
              if (ld.getIndices().size() == 1)
                continue;
            } else if (auto st = dyn_cast<memref::StoreOp>(a)) {
              if (st.getIndices().size() == 1 &&
                  st.getValueToStore() != p2m.getResult())
                continue;
            }
            ok = false;
            break;
          }
          if (!ok)
            break;
        }
        if (!ok)
          break;
        inductions.push_back({i, byteStep, gep});
        drop[i] = true;
      }
      if (!ok || !anyPtr || llvm::none_of(drop, [](bool d) { return d; }))
        continue;

      // Completed iterations: k = iv - lb.
      OpBuilder kb(f.getBody(), f.getBody()->begin());
      Location loc = f.getLoc();
      Value k = f.getInductionVar();
      AffineMap lbMap = f.getLowerBoundMap();
      if (!(lbMap.isSingleConstant() && lbMap.getSingleConstantResult() == 0)) {
        Value lb = affine::AffineApplyOp::create(kb, loc, lbMap,
                                                 f.getLowerBoundOperands());
        k = arith::SubIOp::create(kb, loc, k, lb);
      }

      for (auto &ind : inductions) {
        Value init = f.getInits()[ind.idx];
        BlockArgument arg = f.getRegionIterArgs()[ind.idx];
        for (Operation *u : llvm::make_early_inc_range(arg.getUsers())) {
          auto p2m = dyn_cast<enzymexla::Pointer2MemrefOp>(u);
          if (!p2m)
            continue;
          int64_t stepElts =
              ind.byteStep /
              (int64_t)dl.getTypeSize(p2m.getType().getElementType());
          OpBuilder vb(p2m);
          Value nv = enzymexla::Pointer2MemrefOp::create(vb, p2m.getLoc(),
                                                         p2m.getType(), init);
          Value off = k;
          if (stepElts != 1)
            off = arith::MulIOp::create(
                vb, loc, k,
                arith::ConstantIndexOp::create(vb, loc, stepElts));
          for (Operation *a : llvm::make_early_inc_range(p2m->getUsers())) {
            OpBuilder ab(a);
            auto add = [&](Value idx) {
              return arith::AddIOp::create(ab, a->getLoc(), idx, off)
                  .getResult();
            };
            if (auto ld = dyn_cast<affine::AffineLoadOp>(a)) {
              auto expanded = affine::expandAffineMap(
                  ab, a->getLoc(), ld.getMap(), ld.getMapOperands());
              Value nl = memref::LoadOp::create(ab, a->getLoc(), nv,
                                                ValueRange{add((*expanded)[0])});
              a->getResult(0).replaceAllUsesWith(nl);
              a->erase();
            } else if (auto st = dyn_cast<affine::AffineStoreOp>(a)) {
              auto expanded = affine::expandAffineMap(
                  ab, a->getLoc(), st.getMap(), st.getMapOperands());
              memref::StoreOp::create(ab, a->getLoc(), st.getValueToStore(),
                                      nv, ValueRange{add((*expanded)[0])});
              a->erase();
            } else if (auto ld = dyn_cast<memref::LoadOp>(a)) {
              Value nl = memref::LoadOp::create(
                  ab, a->getLoc(), nv, ValueRange{add(ld.getIndices()[0])});
              a->getResult(0).replaceAllUsesWith(nl);
              a->erase();
            } else {
              auto st = cast<memref::StoreOp>(a);
              memref::StoreOp::create(ab, a->getLoc(), st.getValueToStore(),
                                      nv, ValueRange{add(st.getIndices()[0])});
              a->erase();
            }
          }
          p2m.erase();
        }
      }

      // Rebuild the loop without the pointer iter args.
      SmallVector<Value> newInits;
      for (unsigned i = 0; i < n; ++i)
        if (!drop[i])
          newInits.push_back(f.getInits()[i]);
      OpBuilder b(f);
      auto nf = affine::AffineForOp::create(
          b, f.getLoc(), f.getLowerBoundOperands(), f.getLowerBoundMap(),
          f.getUpperBoundOperands(), f.getUpperBoundMap(), f.getStepAsInt(),
          newInits);
      Block *ob = f.getBody(), *nb = nf.getBody();
      if (!nb->empty())
        nb->clear();
      f.getInductionVar().replaceAllUsesWith(nf.getInductionVar());
      unsigned kept = 0;
      for (unsigned i = 0; i < n; ++i)
        if (!drop[i])
          f.getRegionIterArgs()[i].replaceAllUsesWith(
              nf.getRegionIterArgs()[kept++]);
      SmallVector<Value> keptYields;
      for (unsigned i = 0; i < n; ++i)
        if (!drop[i])
          keptYields.push_back(yield.getOperand(i));
      OpBuilder yb(yield);
      affine::AffineYieldOp::create(yb, yield.getLoc(), keptYields);
      yield.erase();
      // The advancing geps are dead now that nothing yields them.
      for (auto &ind : inductions)
        if (ind.advance->use_empty())
          ind.advance.erase();
      nb->getOperations().splice(nb->end(), ob->getOperations());
      kept = 0;
      for (unsigned i = 0; i < n; ++i)
        if (!drop[i])
          f.getResult(i).replaceAllUsesWith(nf.getResult(kept++));
      f.erase();
    }
  }

  // A not-quite-inlined device lambda packs its captures into a stack
  // struct and reads them back through typed views at constant offsets.
  // Forward each load to the unique dominating store of the same slot so
  // the pointer-typed members never reach the raising as memory.
  static void forwardPackedScratch(Operation *root) {
    SmallVector<LLVM::AllocaOp> allocas;
    root->walk([&](LLVM::AllocaOp a) { allocas.push_back(a); });
    for (auto a : allocas) {
      // Transitively collect constant-byte-offset accesses.
      struct Access {
        Operation *op;
        int64_t byteOff;
        bool isStore;
        Type ty;
        Value value; // stored value for stores
      };
      SmallVector<Access> accesses;
      SmallVector<Operation *> chain;
      bool ok = true;
      SmallVector<std::pair<Value, int64_t>> work{{a.getResult(), 0}};
      DataLayout dl = DataLayout::closest(a);
      while (ok && !work.empty()) {
        auto [v, off] = work.pop_back_val();
        for (Operation *u : v.getUsers()) {
          if (auto gep = dyn_cast<LLVM::GEPOp>(u)) {
            if (gep.getBase() != v || !gep.getDynamicIndices().empty()) {
              ok = false;
              break;
            }
            int64_t gepOff = 0;
            Type cur = gep.getElemType();
            auto idxs = gep.getIndices();
            bool constGep = idxs.size() >= 1;
            if (constGep)
              gepOff = cast<IntegerAttr>(idxs[0]).getInt() *
                       (int64_t)dl.getTypeSize(cur);
            for (unsigned i = 1; constGep && i < idxs.size(); ++i) {
              int64_t want = cast<IntegerAttr>(idxs[i]).getInt();
              if (auto AT = dyn_cast<LLVM::LLVMArrayType>(cur)) {
                cur = AT.getElementType();
                gepOff += want * (int64_t)dl.getTypeSize(cur);
              } else if (auto ST = dyn_cast<LLVM::LLVMStructType>(cur)) {
                if (ST.isOpaque() || (size_t)want >= ST.getBody().size()) {
                  constGep = false;
                  break;
                }
                for (int64_t k = 0; k < want; ++k) {
                  int64_t sz = (int64_t)dl.getTypeSize(ST.getBody()[k]);
                  int64_t al = (int64_t)dl.getTypeABIAlignment(ST.getBody()[k]);
                  gepOff = (gepOff + al - 1) / al * al + sz;
                }
                int64_t al =
                    (int64_t)dl.getTypeABIAlignment(ST.getBody()[want]);
                gepOff = (gepOff + al - 1) / al * al;
                cur = ST.getBody()[want];
              } else {
                constGep = false;
              }
            }
            if (!constGep) {
              ok = false;
              break;
            }
            chain.push_back(gep);
            work.push_back({gep.getResult(), off + gepOff});
            continue;
          }
          if (auto p2m = dyn_cast<enzymexla::Pointer2MemrefOp>(u)) {
            auto MT = p2m.getType();
            if (MT.getRank() != 1) {
              ok = false;
              break;
            }
            Type ET = MT.getElementType();
            int64_t esz = isa<LLVM::LLVMPointerType>(ET)
                              ? 8
                              : (ET.isIntOrFloat()
                                     ? (ET.getIntOrFloatBitWidth() + 7) / 8
                                     : 0);
            if (esz == 0) {
              ok = false;
              break;
            }
            bool viewOk = true;
            for (Operation *au : p2m->getUsers()) {
              std::optional<int64_t> idx;
              if (auto ld = dyn_cast<affine::AffineLoadOp>(au)) {
                if (auto c = getConstant(ld.getAffineMap()))
                  idx = *c;
              } else if (auto st = dyn_cast<affine::AffineStoreOp>(au)) {
                if (st.getValue() != p2m.getResult())
                  if (auto c = getConstant(st.getAffineMap()))
                    idx = *c;
              }
              if (!idx) {
                viewOk = false;
                break;
              }
              bool isStore = isa<affine::AffineStoreOp>(au);
              accesses.push_back(
                  {au, off + *idx * esz, isStore, ET,
                   isStore ? cast<affine::AffineStoreOp>(au).getValue()
                           : Value()});
            }
            if (!viewOk) {
              ok = false;
              break;
            }
            chain.push_back(p2m);
            continue;
          }
          if (auto ld = dyn_cast<LLVM::LoadOp>(u)) {
            accesses.push_back({u, off, false, ld.getType(), Value()});
            continue;
          }
          if (auto st = dyn_cast<LLVM::StoreOp>(u)) {
            if (st.getValue() == v) {
              ok = false;
              break;
            }
            accesses.push_back(
                {u, off, true, st.getValue().getType(), st.getValue()});
            continue;
          }
          if (isa<LLVM::LifetimeStartOp, LLVM::LifetimeEndOp>(u)) {
            chain.push_back(u);
            continue;
          }
          ok = false;
          break;
        }
      }
      if (!ok)
        continue;
      // One store per slot, dominating every load of that slot.
      DominanceInfo dom(root);
      llvm::DenseMap<int64_t, Access *> storeAt;
      bool legal = true;
      for (auto &acc : accesses)
        if (acc.isStore) {
          if (storeAt.count(acc.byteOff)) {
            legal = false;
            break;
          }
          storeAt[acc.byteOff] = &acc;
        }
      if (!legal)
        continue;
      for (auto &acc : accesses) {
        if (acc.isStore)
          continue;
        auto it = storeAt.find(acc.byteOff);
        if (it == storeAt.end() || it->second->ty != acc.ty ||
            !dom.properlyDominates(it->second->op, acc.op)) {
          legal = false;
          break;
        }
      }
      if (!legal)
        continue;
      for (auto &acc : accesses)
        if (!acc.isStore) {
          acc.op->getResult(0).replaceAllUsesWith(
              storeAt[acc.byteOff]->value);
          acc.op->erase();
        }
      for (auto &acc : accesses)
        if (acc.isStore)
          acc.op->erase();
      for (Operation *c : llvm::reverse(chain))
        if (c->use_empty())
          c->erase();
      if (a->use_empty())
        a.erase();
    }
  }

  // A side-selected view arrives as a select of byte offsets (or of the
  // pointers themselves) feeding geps: distribute the gep and the memref
  // view over the select so the buffer-branch expansion can take over.
  static void distributeGepOverSelect(Operation *root) {
    bool changed = true;
    while (changed) {
      changed = false;
      SmallVector<Operation *> worklist;
      root->walk([&](Operation *op) {
        if (isa<LLVM::GEPOp, enzymexla::Pointer2MemrefOp>(op))
          worklist.push_back(op);
      });
      for (Operation *op : worklist) {
        if (auto p2m = dyn_cast<enzymexla::Pointer2MemrefOp>(op)) {
          auto sel = p2m.getSource().getDefiningOp<arith::SelectOp>();
          if (!sel)
            continue;
          auto condTy = dyn_cast<RankedTensorType>(sel.getCondition().getType());
          if (condTy)
            continue;
          OpBuilder b(op);
          Value t = enzymexla::Pointer2MemrefOp::create(
              b, op->getLoc(), p2m.getType(), sel.getTrueValue());
          Value f = enzymexla::Pointer2MemrefOp::create(
              b, op->getLoc(), p2m.getType(), sel.getFalseValue());
          Value ns = arith::SelectOp::create(b, op->getLoc(),
                                             sel.getCondition(), t, f);
          op->getResult(0).replaceAllUsesWith(ns);
          op->erase();
          changed = true;
          continue;
        }
        auto gep = cast<LLVM::GEPOp>(op);
        // Base is a select of pointers.
        if (auto sel = gep.getBase().getDefiningOp<arith::SelectOp>()) {
          OpBuilder b(op);
          IRMapping mt, mf;
          mt.map(gep.getBase(), sel.getTrueValue());
          mf.map(gep.getBase(), sel.getFalseValue());
          Operation *gt = b.clone(*op, mt);
          Operation *gf = b.clone(*op, mf);
          Value ns = arith::SelectOp::create(
              b, op->getLoc(), sel.getCondition(), gt->getResult(0),
              gf->getResult(0));
          op->getResult(0).replaceAllUsesWith(ns);
          op->erase();
          changed = true;
          continue;
        }
        // A single dynamic index that is a select of two values.
        if (gep.getDynamicIndices().size() == 1) {
          auto sel =
              gep.getDynamicIndices()[0].getDefiningOp<arith::SelectOp>();
          if (!sel || isa<RankedTensorType>(sel.getCondition().getType()))
            continue;
          OpBuilder b(op);
          IRMapping mt, mf;
          mt.map(gep.getDynamicIndices()[0], sel.getTrueValue());
          mf.map(gep.getDynamicIndices()[0], sel.getFalseValue());
          Operation *gt = b.clone(*op, mt);
          Operation *gf = b.clone(*op, mf);
          Value ns = arith::SelectOp::create(
              b, op->getLoc(), sel.getCondition(), gt->getResult(0),
              gf->getResult(0));
          op->getResult(0).replaceAllUsesWith(ns);
          op->erase();
          changed = true;
        }
      }
    }
  }

  static void stripAccessMemorySpaceCasts(Operation *root) {
    SmallVector<memref::MemorySpaceCastOp> casts;
    root->walk([&](memref::MemorySpaceCastOp c) { casts.push_back(c); });
    for (auto c : casts) {
      if (!llvm::all_of(c->getUsers(), [](Operation *u) {
            return isa<affine::AffineLoadOp, affine::AffineStoreOp,
                       memref::LoadOp, memref::StoreOp>(u);
          }))
        continue;
      for (Operation *u : llvm::make_early_inc_range(c->getUsers()))
        u->replaceUsesOfWith(c.getResult(), c.getSource());
      c.erase();
    }
    // Shape-erasing casts of static buffers block raising the same way:
    // accesses go straight to the static source.
    SmallVector<memref::CastOp> shapeCasts;
    root->walk([&](memref::CastOp c) {
      auto src = dyn_cast<MemRefType>(c.getSource().getType());
      auto dst = dyn_cast<MemRefType>(c.getType());
      if (src && dst && src.hasStaticShape() && !dst.hasStaticShape())
        shapeCasts.push_back(c);
    });
    for (auto c : shapeCasts) {
      if (!llvm::all_of(c->getUsers(), [](Operation *u) {
            return isa<affine::AffineLoadOp, affine::AffineStoreOp,
                       memref::LoadOp, memref::StoreOp>(u);
          }))
        continue;
      for (Operation *u : llvm::make_early_inc_range(c->getUsers()))
        u->replaceUsesOfWith(c.getResult(), c.getSource());
      c.erase();
    }
  }

  // A branch yielding one of several buffers blocks raising: no tensor can
  // stand for "one of these two memrefs". Duplicate the branch at every
  // access instead — a load becomes a value-yielding branch loading in each
  // arm, a store becomes a store in each arm — so every access reaches a
  // real buffer and the usual select/mask raising applies. Branch bodies are
  // cloned per access, so only effect-free bodies qualify.
  static void expandBufferBranches(Operation *root) {
    // The same shape also arrives as an arith.select of two buffers: expand
    // each access into an scf.if on the select's condition.
    SmallVector<arith::SelectOp> selects;
    root->walk([&](arith::SelectOp sel) {
      if (isa<MemRefType>(sel.getType()))
        selects.push_back(sel);
    });
    for (auto sel : selects) {
      for (OpOperand &use : llvm::make_early_inc_range(sel->getUses())) {
        Operation *user = use.getOwner();
        bool isLoad = isa<memref::LoadOp, affine::AffineLoadOp>(user);
        bool isStore = isa<memref::StoreOp, affine::AffineStoreOp>(user);
        unsigned memIdx = isLoad ? 0 : 1;
        if ((!isLoad && !isStore) || use.getOperandNumber() != memIdx)
          continue;
        OpBuilder b(user);
        auto newIf = scf::IfOp::create(
            b, user->getLoc(),
            isLoad ? TypeRange(user->getResult(0).getType()) : TypeRange(),
            sel.getCondition(), /*withElseRegion=*/true);
        auto fillArm = [&](Value buf, Block *dstArm) {
          dstArm->clear();
          IRMapping m;
          OpBuilder ab = OpBuilder::atBlockBegin(dstArm);
          Operation *access = ab.clone(*user, m);
          access->setOperand(memIdx, buf);
          scf::YieldOp::create(ab, user->getLoc(),
                               isLoad ? ValueRange(access->getResult(0))
                                      : ValueRange());
        };
        fillArm(sel.getTrueValue(), newIf.thenBlock());
        fillArm(sel.getFalseValue(), newIf.elseBlock());
        if (isLoad)
          user->getResult(0).replaceAllUsesWith(newIf.getResult(0));
        user->erase();
      }
      if (sel->use_empty())
        sel.erase();
    }

    // The coefficient ternary arrives as an scf.if yielding a pointer whose
    // arms compute constant- or index-offset geps; push each viewing access
    // down into a clone of the branch so no pointer crosses the yield.
    SmallVector<scf::IfOp> scfWorklist;
    root->walk([&](scf::IfOp ifOp) {
      if (ifOp.elseBlock() && llvm::any_of(ifOp.getResultTypes(), [](Type t) {
            return isa<LLVM::LLVMPointerType>(t);
          }))
        scfWorklist.push_back(ifOp);
    });
    // Arms may be cloned once per pushed-down access, so they must not
    // write; reads are idempotent and safe to duplicate.
    auto armClonable = [](Block *b) {
      return llvm::all_of(b->without_terminator(), [](Operation &op) {
        if (isMemoryEffectFree(&op))
          return true;
        auto mem = dyn_cast<MemoryEffectOpInterface>(&op);
        if (!mem || op.getNumRegions() != 0)
          return false;
        SmallVector<MemoryEffects::EffectInstance> effects;
        mem.getEffects(effects);
        return llvm::all_of(effects, [](MemoryEffects::EffectInstance &e) {
          return isa<MemoryEffects::Read>(e.getEffect());
        });
      });
    };
    for (auto ifOp : scfWorklist) {
      Block *thenB = ifOp.thenBlock(), *elseB = ifOp.elseBlock();
      if (!armClonable(thenB) || !armClonable(elseB))
        continue;
      for (auto [i, res] : llvm::enumerate(ifOp.getResults())) {
        if (!isa<LLVM::LLVMPointerType>(res.getType()))
          continue;
        Value thenV = thenB->getTerminator()->getOperand(i);
        Value elseV = elseB->getTerminator()->getOperand(i);
        for (OpOperand &use : llvm::make_early_inc_range(res.getUses())) {
          auto p2m = dyn_cast<enzymexla::Pointer2MemrefOp>(use.getOwner());
          if (!p2m)
            continue;
          for (OpOperand &ause :
               llvm::make_early_inc_range(p2m->getUses())) {
            Operation *acc = ause.getOwner();
            bool isLoad = isa<memref::LoadOp, affine::AffineLoadOp>(acc);
            bool isStore = isa<memref::StoreOp, affine::AffineStoreOp>(acc);
            unsigned memIdx = isLoad ? 0 : 1;
            if ((!isLoad && !isStore) || ause.getOperandNumber() != memIdx)
              continue;
            OpBuilder b(acc);
            auto newIf = scf::IfOp::create(
                b, acc->getLoc(),
                isLoad ? TypeRange(acc->getResult(0).getType()) : TypeRange(),
                ifOp.getCondition(), /*withElseRegion=*/true);
            auto fillArm = [&](Block *srcArm, Value yielded, Block *dstArm) {
              dstArm->clear();
              IRMapping m;
              OpBuilder ab = OpBuilder::atBlockBegin(dstArm);
              for (Operation &armOp : srcArm->without_terminator())
                ab.clone(armOp, m);
              Operation *view = ab.clone(*p2m.getOperation(), m);
              view->setOperand(0, m.lookupOrDefault(yielded));
              Operation *access = ab.clone(*acc, m);
              access->setOperand(memIdx, view->getResult(0));
              scf::YieldOp::create(ab, acc->getLoc(),
                                   isLoad ? ValueRange(access->getResult(0))
                                          : ValueRange());
            };
            fillArm(thenB, thenV, newIf.thenBlock());
            fillArm(elseB, elseV, newIf.elseBlock());
            if (isLoad)
              acc->getResult(0).replaceAllUsesWith(newIf.getResult(0));
            acc->erase();
          }
          if (p2m->use_empty())
            p2m.erase();
        }
      }
      // Rebuild without dead pointer results if scalars keep it alive.
      if (llvm::all_of(ifOp.getResults(),
                       [](Value r) { return r.use_empty(); })) {
        ifOp.erase();
        continue;
      }
      if (llvm::any_of(ifOp.getResults(), [](Value r) {
            return isa<LLVM::LLVMPointerType>(r.getType()) && r.use_empty();
          })) {
        SmallVector<unsigned> liveIdx;
        SmallVector<Type> liveTypes;
        for (auto [i, res] : llvm::enumerate(ifOp.getResults())) {
          if (isa<LLVM::LLVMPointerType>(res.getType()) && res.use_empty())
            continue;
          liveIdx.push_back((unsigned)i);
          liveTypes.push_back(res.getType());
        }
        OpBuilder b(ifOp);
        auto newIf = scf::IfOp::create(b, ifOp.getLoc(), liveTypes,
                                       ifOp.getCondition(),
                                       /*withElseRegion=*/true);
        auto rebuildArm = [&](Block *srcArm, Block *dstArm) {
          dstArm->clear();
          IRMapping m;
          OpBuilder ab = OpBuilder::atBlockBegin(dstArm);
          for (Operation &armOp : srcArm->without_terminator())
            ab.clone(armOp, m);
          SmallVector<Value> yields;
          for (unsigned i : liveIdx)
            yields.push_back(
                m.lookupOrDefault(srcArm->getTerminator()->getOperand(i)));
          scf::YieldOp::create(ab, ifOp.getLoc(), yields);
          for (Operation &armOp :
               llvm::make_early_inc_range(dstArm->without_terminator()))
            if (armOp.use_empty() && isMemoryEffectFree(&armOp))
              armOp.erase();
        };
        rebuildArm(thenB, newIf.thenBlock());
        rebuildArm(elseB, newIf.elseBlock());
        for (auto [k, i] : llvm::enumerate(liveIdx))
          ifOp.getResult(i).replaceAllUsesWith(newIf.getResult(k));
        ifOp.erase();
      }
    }

    SmallVector<affine::AffineIfOp> worklist;
    root->walk([&](affine::AffineIfOp ifOp) {
      if (ifOp.hasElse() && llvm::any_of(ifOp.getResultTypes(), [](Type t) {
            return isa<MemRefType>(t);
          }))
        worklist.push_back(ifOp);
    });
    for (auto ifOp : worklist) {
      Block *thenB = ifOp.getThenBlock(), *elseB = ifOp.getElseBlock();
      if (!armClonable(thenB) || !armClonable(elseB))
        continue;
      for (auto [i, res] : llvm::enumerate(ifOp.getResults())) {
        if (!isa<MemRefType>(res.getType()))
          continue;
        Value thenV = thenB->getTerminator()->getOperand(i);
        Value elseV = elseB->getTerminator()->getOperand(i);
        for (OpOperand &use : llvm::make_early_inc_range(res.getUses())) {
          Operation *user = use.getOwner();
          bool isLoad = isa<memref::LoadOp, affine::AffineLoadOp>(user);
          bool isStore = isa<memref::StoreOp, affine::AffineStoreOp>(user);
          unsigned memIdx = isLoad ? 0 : 1;
          if ((!isLoad && !isStore) || use.getOperandNumber() != memIdx)
            continue;
          OpBuilder b(user);
          auto newIf = affine::AffineIfOp::create(
              b, user->getLoc(),
              isLoad ? TypeRange(user->getResult(0).getType()) : TypeRange(),
              ifOp.getIntegerSet(), ifOp.getOperands(),
              /*withElseRegion=*/true);
          auto fillArm = [&](Block *srcArm, Value yielded, Block *dstArm) {
            if (Operation *term = dstArm->empty() ? nullptr : &dstArm->back())
              if (term->hasTrait<OpTrait::IsTerminator>())
                term->erase();
            IRMapping m;
            OpBuilder ab = OpBuilder::atBlockEnd(dstArm);
            for (Operation &armOp : srcArm->without_terminator())
              ab.clone(armOp, m);
            Operation *access = ab.clone(*user, m);
            access->setOperand(memIdx, m.lookupOrDefault(yielded));
            affine::AffineYieldOp::create(
                ab, user->getLoc(),
                isLoad ? ValueRange(access->getResult(0)) : ValueRange());
          };
          fillArm(thenB, thenV, newIf.getThenBlock());
          fillArm(elseB, elseV, newIf.getElseBlock());
          if (isLoad)
            user->getResult(0).replaceAllUsesWith(newIf.getResult(0));
          user->erase();
        }
      }
      if (llvm::all_of(ifOp.getResults(),
                       [](Value r) { return r.use_empty(); })) {
        ifOp.erase();
        continue;
      }
      // Scalar results may keep the branch alive; rebuild it without the
      // now-dead buffer results so no unraisable cast lingers in the arms.
      if (llvm::any_of(ifOp.getResults(), [](Value r) {
            return isa<MemRefType>(r.getType()) && r.use_empty();
          })) {
        SmallVector<unsigned> liveIdx;
        SmallVector<Type> liveTypes;
        for (auto [i, res] : llvm::enumerate(ifOp.getResults())) {
          if (isa<MemRefType>(res.getType()) && res.use_empty())
            continue;
          liveIdx.push_back((unsigned)i);
          liveTypes.push_back(res.getType());
        }
        OpBuilder b(ifOp);
        auto newIf = affine::AffineIfOp::create(
            b, ifOp.getLoc(), liveTypes, ifOp.getIntegerSet(),
            ifOp.getOperands(), /*withElseRegion=*/true);
        auto rebuildArm = [&](Block *srcArm, Block *dstArm) {
          if (Operation *term = dstArm->empty() ? nullptr : &dstArm->back())
            if (term->hasTrait<OpTrait::IsTerminator>())
              term->erase();
          IRMapping m;
          OpBuilder ab = OpBuilder::atBlockEnd(dstArm);
          for (Operation &armOp : srcArm->without_terminator())
            ab.clone(armOp, m);
          SmallVector<Value> yields;
          for (unsigned i : liveIdx)
            yields.push_back(
                m.lookupOrDefault(srcArm->getTerminator()->getOperand(i)));
          affine::AffineYieldOp::create(ab, ifOp.getLoc(), yields);
          // The buffer arms may still hold the dead casts; drop them.
          for (Operation &armOp :
               llvm::make_early_inc_range(dstArm->without_terminator()))
            if (armOp.use_empty() && isMemoryEffectFree(&armOp))
              armOp.erase();
        };
        rebuildArm(thenB, newIf.getThenBlock());
        rebuildArm(elseB, newIf.getElseBlock());
        for (auto [k, i] : llvm::enumerate(liveIdx))
          ifOp.getResult(i).replaceAllUsesWith(newIf.getResult(k));
        ifOp.erase();
      }
    }
  }

  // A parallel dimension whose extent is only known at runtime cannot become
  // a tensor axis, but its iterations are still independent: peel each such
  // dimension into an affine.for tagged enzymexla.parallel, which the while
  // raising then iterates, leaving the constant-extent dimensions to raise as
  // axes. The tag rides onto the stablehlo.while so downstream passes know
  // the iterations commute.
  // The constant upper bound of an extent value, where one can be derived:
  // the value itself when constant, or the constant side of a min it is
  // clamped by (MFEM's block sizes arrive as min(1 << log2(N), 256)).
  // Upper bound from a dominating verify guard: `if (v REL C) <noreturn>`
  // leaves the complementary relation holding on every path that reaches
  // the launch.
  static std::optional<int64_t> guardBound(Value v, Operation *anchor) {
    if (!anchor)
      return std::nullopt;
    std::optional<int64_t> bound;
    auto consider = [&](int64_t b) {
      if (!bound || b < *bound)
        bound = b;
    };
    for (Operation *user : v.getUsers()) {
      auto cmp = dyn_cast<arith::CmpIOp>(user);
      if (getenv("DEBUG_GUARD") && cmp) {
        llvm::errs() << "guard cmp: " << *cmp << "\n";
        for (Operation *cu : cmp->getUsers())
          llvm::errs() << "  user: " << cu->getName() << "\n";
      }
      if (!cmp)
        continue;
      APInt cst;
      bool vLhs;
      if (cmp.getLhs() == v && matchPattern(cmp.getRhs(), m_ConstantInt(&cst)))
        vLhs = true;
      else if (cmp.getRhs() == v &&
               matchPattern(cmp.getLhs(), m_ConstantInt(&cst)))
        vLhs = false;
      else
        continue;
      int64_t C = cst.getSExtValue();
      for (Operation *cu : cmp->getUsers()) {
        auto ifOp = dyn_cast<scf::IfOp>(cu);
        if (!ifOp || ifOp.getCondition() != cmp.getResult())
          continue;
        auto isNoReturn = [](Region &r) {
          bool f = false;
          r.walk([&](LLVM::UnreachableOp) { f = true; });
          return f;
        };
        bool thenNR = isNoReturn(ifOp.getThenRegion());
        bool elseNR =
            !ifOp.getElseRegion().empty() && isNoReturn(ifOp.getElseRegion());
        if (thenNR == elseNR)
          continue;
        Operation *a = anchor;
        while (a && a->getBlock() != ifOp->getBlock())
          a = a->getParentOp();
        if (!a || a == ifOp || !ifOp->isBeforeInBlock(a))
          continue;
        // The surviving path holds cond when the else branch aborts, and
        // !cond when the then branch aborts.
        arith::CmpIPredicate pred = cmp.getPredicate();
        if (thenNR)
          pred = arith::invertPredicate(pred);
        if (!vLhs) {
          switch (pred) {
          case arith::CmpIPredicate::sgt:
            pred = arith::CmpIPredicate::slt;
            break;
          case arith::CmpIPredicate::sge:
            pred = arith::CmpIPredicate::sle;
            break;
          case arith::CmpIPredicate::slt:
            pred = arith::CmpIPredicate::sgt;
            break;
          case arith::CmpIPredicate::sle:
            pred = arith::CmpIPredicate::sge;
            break;
          case arith::CmpIPredicate::ugt:
            pred = arith::CmpIPredicate::ult;
            break;
          case arith::CmpIPredicate::uge:
            pred = arith::CmpIPredicate::ule;
            break;
          case arith::CmpIPredicate::ult:
            pred = arith::CmpIPredicate::ugt;
            break;
          case arith::CmpIPredicate::ule:
            pred = arith::CmpIPredicate::uge;
            break;
          default:
            break;
          }
        }
        switch (pred) {
        case arith::CmpIPredicate::sle:
        case arith::CmpIPredicate::ule:
          consider(C);
          break;
        case arith::CmpIPredicate::slt:
        case arith::CmpIPredicate::ult:
          consider(C - 1);
          break;
        default:
          break;
        }
      }
    }
    return bound;
  }

  // Upper bound from the guards enclosing the launch: inside the surviving
  // branch of `if (v REL C)`, the relation holds.
  static std::optional<int64_t> enclosingGuardBound(Value v,
                                                    Operation *anchor) {
    if (!anchor)
      return std::nullopt;
    std::optional<int64_t> bound;
    auto consider = [&](int64_t b) {
      if (!bound || b < *bound)
        bound = b;
    };
    for (Operation *cur = anchor; cur->getParentOp();
         cur = cur->getParentOp()) {
      auto ifOp = dyn_cast<scf::IfOp>(cur->getParentOp());
      if (!ifOp)
        continue;
      bool inThen = cur->getParentRegion() == &ifOp.getThenRegion();
      auto cmp = ifOp.getCondition().getDefiningOp<arith::CmpIOp>();
      if (getenv("DEBUG_GUARD")) {
        llvm::errs() << "enclosing if cond for " << v << ": ";
        if (auto *d = ifOp.getCondition().getDefiningOp())
          llvm::errs() << *d;
        llvm::errs() << " inThen=" << inThen << "\n";
      }
      if (!cmp)
        continue;
      APInt cst;
      bool vLhs;
      if (cmp.getLhs() == v && matchPattern(cmp.getRhs(), m_ConstantInt(&cst)))
        vLhs = true;
      else if (cmp.getRhs() == v &&
               matchPattern(cmp.getLhs(), m_ConstantInt(&cst)))
        vLhs = false;
      else
        continue;
      int64_t C = cst.getSExtValue();
      arith::CmpIPredicate pred = cmp.getPredicate();
      if (!inThen)
        pred = arith::invertPredicate(pred);
      if (!vLhs) {
        switch (pred) {
        case arith::CmpIPredicate::sgt:
          pred = arith::CmpIPredicate::slt;
          break;
        case arith::CmpIPredicate::sge:
          pred = arith::CmpIPredicate::sle;
          break;
        case arith::CmpIPredicate::slt:
          pred = arith::CmpIPredicate::sgt;
          break;
        case arith::CmpIPredicate::sle:
          pred = arith::CmpIPredicate::sge;
          break;
        case arith::CmpIPredicate::ugt:
          pred = arith::CmpIPredicate::ult;
          break;
        case arith::CmpIPredicate::uge:
          pred = arith::CmpIPredicate::ule;
          break;
        case arith::CmpIPredicate::ult:
          pred = arith::CmpIPredicate::ugt;
          break;
        case arith::CmpIPredicate::ule:
          pred = arith::CmpIPredicate::uge;
          break;
        default:
          break;
        }
      }
      switch (pred) {
      case arith::CmpIPredicate::sle:
      case arith::CmpIPredicate::ule:
        consider(C);
        break;
      case arith::CmpIPredicate::slt:
      case arith::CmpIPredicate::ult:
        consider(C - 1);
        break;
      case arith::CmpIPredicate::eq:
        consider(C);
        break;
      default:
        break;
      }
    }
    return bound;
  }

  static std::optional<int64_t> derivedExtentBound(Value v, unsigned depth = 0,
                                                   Operation *anchor =
                                                       nullptr) {
    if (depth > 8)
      return std::nullopt;
    while (true) {
      if (auto c = v.getDefiningOp<arith::IndexCastOp>()) {
        v = c.getIn();
        continue;
      }
      if (auto c = v.getDefiningOp<arith::IndexCastUIOp>()) {
        v = c.getIn();
        continue;
      }
      break;
    }
    APInt cst;
    if (matchPattern(v, m_ConstantInt(&cst)))
      return cst.getSExtValue();
    if (auto mn = v.getDefiningOp<arith::MinSIOp>()) {
      auto l = derivedExtentBound(mn.getLhs(), depth + 1, anchor);
      auto r = derivedExtentBound(mn.getRhs(), depth + 1, anchor);
      if (l && r)
        return std::min(*l, *r);
      return l ? l : r;
    }
    if (auto mn = v.getDefiningOp<arith::MinUIOp>()) {
      auto l = derivedExtentBound(mn.getLhs(), depth + 1, anchor);
      auto r = derivedExtentBound(mn.getRhs(), depth + 1, anchor);
      if (l && r)
        return std::min(*l, *r);
      return l ? l : r;
    }
    if (auto mn = v.getDefiningOp<LLVM::SMinOp>()) {
      auto l = derivedExtentBound(mn->getOperand(0), depth + 1, anchor);
      auto r = derivedExtentBound(mn->getOperand(1), depth + 1, anchor);
      if (l && r)
        return std::min(*l, *r);
      return l ? l : r;
    }
    if (auto mn = v.getDefiningOp<LLVM::UMinOp>()) {
      auto l = derivedExtentBound(mn->getOperand(0), depth + 1, anchor);
      auto r = derivedExtentBound(mn->getOperand(1), depth + 1, anchor);
      if (l && r)
        return std::min(*l, *r);
      return l ? l : r;
    }
    if (isa_and_nonnull<LLVM::SMaxOp, LLVM::UMaxOp>(v.getDefiningOp())) {
      auto l = derivedExtentBound(v.getDefiningOp()->getOperand(0), depth + 1, anchor);
      auto r = derivedExtentBound(v.getDefiningOp()->getOperand(1), depth + 1, anchor);
      if (l && r)
        return std::max(*l, *r);
      return std::nullopt;
    }
    // A max of bounded values is bounded by the larger bound; both sides
    // must be bounded, unlike min.
    if (auto mx = v.getDefiningOp<arith::MaxSIOp>()) {
      auto l = derivedExtentBound(mx.getLhs(), depth + 1, anchor);
      auto r = derivedExtentBound(mx.getRhs(), depth + 1, anchor);
      if (l && r)
        return std::max(*l, *r);
      return std::nullopt;
    }
    if (auto mx = v.getDefiningOp<arith::MaxUIOp>()) {
      auto l = derivedExtentBound(mx.getLhs(), depth + 1, anchor);
      auto r = derivedExtentBound(mx.getRhs(), depth + 1, anchor);
      if (l && r)
        return std::max(*l, *r);
      return std::nullopt;
    }
    if (auto sel = v.getDefiningOp<arith::SelectOp>()) {
      auto l = derivedExtentBound(sel.getTrueValue(), depth + 1, anchor);
      auto r = derivedExtentBound(sel.getFalseValue(), depth + 1, anchor);
      if (l && r)
        return std::max(*l, *r);
      return std::nullopt;
    }
    if (isa_and_nonnull<arith::ExtUIOp, arith::ExtSIOp>(v.getDefiningOp()))
      return derivedExtentBound(v.getDefiningOp()->getOperand(0), depth + 1, anchor);
    if (auto t = v.getDefiningOp<arith::TruncIOp>()) {
      // dim3 packing replicates a 32-bit dim into both halves of an i64 as
      // x * 0x100000001; either half recovers the dim.
      if (auto mul = t.getIn().getDefiningOp<arith::MulIOp>()) {
        APInt k;
        if (matchPattern(mul.getRhs(), m_ConstantInt(&k)) &&
            k.getZExtValue() == 0x100000001ULL)
          return derivedExtentBound(mul.getLhs(), depth + 1, anchor);
      }
      auto b = derivedExtentBound(t.getIn(), depth + 1, anchor);
      unsigned w = t.getType().getIntOrFloatBitWidth();
      if (b && *b >= 0 && (w >= 63 || *b < (int64_t(1) << w)))
        return b;
      return std::nullopt;
    }
    if (auto sh = v.getDefiningOp<arith::ShRUIOp>()) {
      APInt k;
      if (matchPattern(sh.getRhs(), m_ConstantInt(&k)) &&
          k.getZExtValue() < 63) {
        auto b = derivedExtentBound(sh.getLhs(), depth + 1, anchor);
        if (b && *b >= 0)
          return *b >> k.getZExtValue();
      }
      return std::nullopt;
    }
    // dim3 packing with a constant second half arrives as a disjoint or:
    // either half of (a | c) recovers its own dim.
    if (auto orOp = v.getDefiningOp<arith::OrIOp>()) {
      APInt k;
      if (matchPattern(orOp.getRhs(), m_ConstantInt(&k))) {
        auto b = derivedExtentBound(orOp.getLhs(), depth + 1, anchor);
        // a <= b does not order a|c against b|c bitwise; a|c <= a+c <= b+c.
        if (b && *b >= 0 && k.getSExtValue() >= 0)
          return *b + k.getSExtValue();
      }
      return std::nullopt;
    }
    if (auto mul = v.getDefiningOp<arith::MulIOp>()) {
      APInt k;
      if (matchPattern(mul.getRhs(), m_ConstantInt(&k))) {
        auto b = derivedExtentBound(mul.getLhs(), depth + 1, anchor);
        int64_t c = k.getSExtValue();
        if (b && *b >= 0 && c >= 0 && (c == 0 || *b <= INT64_MAX / c))
          return *b * c;
        return std::nullopt;
      }
      // Launch dims are non-negative by construction, so a product of two
      // bounded dims (a dof count like 2*(D1D-1)*D1D) stays under the
      // product of the bounds.
      auto l = derivedExtentBound(mul.getLhs(), depth + 1, anchor);
      auto r = derivedExtentBound(mul.getRhs(), depth + 1, anchor);
      if (l && r && *l >= 0 && *r >= 0 && (*l == 0 || *r <= INT64_MAX / *l))
        return *l * *r;
      return std::nullopt;
    }
    if (auto add = v.getDefiningOp<arith::AddIOp>()) {
      APInt k;
      if (matchPattern(add.getRhs(), m_ConstantInt(&k))) {
        auto b = derivedExtentBound(add.getLhs(), depth + 1, anchor);
        if (b)
          return *b + k.getSExtValue();
      }
      return std::nullopt;
    }
    if (auto sub = v.getDefiningOp<arith::SubIOp>()) {
      APInt k;
      if (matchPattern(sub.getRhs(), m_ConstantInt(&k))) {
        auto b = derivedExtentBound(sub.getLhs(), depth + 1, anchor);
        if (b)
          return *b - k.getSExtValue();
      }
      return std::nullopt;
    }
    // A scalar hoisted into a staging buffer (gpu.alloc + memcpy from a
    // stored alloca) reads back the value stored on the host side.
    if (auto ld = v.getDefiningOp<affine::AffineLoadOp>()) {
      if (ld.getMapOperands().empty() &&
          ld.getMap().getNumResults() ==
              (unsigned)ld.getMemRefType().getRank() &&
          ld.getMemRefType().getNumElements() == 1) {
        Value buf = ld.getMemRef();
        if (isa_and_nonnull<gpu::AllocOp>(buf.getDefiningOp())) {
          Value src;
          for (Operation *u : buf.getUsers())
            if (auto mc = dyn_cast<enzymexla::MemcpyOp>(u))
              if (mc->getNumOperands() >= 2 && mc->getOperand(0) == buf) {
                if (src)
                  return std::nullopt;
                src = mc->getOperand(1);
              }
          if (src) {
            Value stored;
            for (Operation *u : src.getUsers())
              if (auto st = dyn_cast<affine::AffineStoreOp>(u)) {
                if (st.getMemRef() != src)
                  continue;
                if (stored)
                  return std::nullopt;
                stored = st.getValueToStore();
              }
            if (stored) {
              auto r = derivedExtentBound(stored, depth + 1, anchor);
              if (!r && getenv("DEBUG_BOUND")) {
                llvm::errs() << "staged scalar underivable:\n";
                Value w = stored;
                for (int k = 0; k < 8 && w.getDefiningOp(); ++k) {
                  llvm::errs() << "  <- " << *w.getDefiningOp() << "\n";
                  if (w.getDefiningOp()->getNumOperands() == 0)
                    break;
                  w = w.getDefiningOp()->getOperand(0);
                }
              }
              return r;
            }
          }
        }
        if (getenv("DEBUG_BOUND"))
          llvm::errs() << "staged scalar: no source found\n";
      }
      return std::nullopt;
    }
    // A launch-stub argument takes its bound from what the callers pass:
    // the max over all call sites, each of which must itself be bounded.
    if (auto ba = dyn_cast<BlockArgument>(v)) {
      if (getenv("DEBUG_BOUND"))
        llvm::errs() << "interproc: blockarg " << ba.getArgNumber()
                     << " depth " << depth << " owner "
                     << ba.getOwner()->getParentOp()->getName() << "\n";
      if (depth > 4)
        return std::nullopt;
      auto func =
          dyn_cast_or_null<FunctionOpInterface>(ba.getOwner()->getParentOp());
      if (!func || func.getFunctionBody().empty() ||
          ba.getOwner() != &func.getFunctionBody().front())
        return std::nullopt;
      auto mod = func->getParentOfType<ModuleOp>();
      if (!mod)
        return std::nullopt;
      auto uses = SymbolTable::getSymbolUses(func, mod);
      if (!uses) {
        if (getenv("DEBUG_BOUND"))
          llvm::errs() << "interproc: no uses view for "
                       << func.getNameAttr() << "\n";
        return std::nullopt;
      }
      std::optional<int64_t> bound;
      bool anyCall = false;
      for (const SymbolTable::SymbolUse &use : *uses) {
        Operation *call = use.getUser();
        Value actual;
        if (auto c = dyn_cast<LLVM::CallOp>(call)) {
          if (ba.getArgNumber() >= c.getArgOperands().size())
            return std::nullopt;
          actual = c.getArgOperands()[ba.getArgNumber()];
        } else if (auto c = dyn_cast<func::CallOp>(call)) {
          if (ba.getArgNumber() >= c.getOperands().size())
            return std::nullopt;
          actual = c.getOperands()[ba.getArgNumber()];
        } else if (isa<LLVM::AddressOfOp>(call)) {
          // Launch stubs keep an addressof for kernel registration; the
          // launches themselves come in as direct calls.
          continue;
        } else {
          if (getenv("DEBUG_BOUND"))
            llvm::errs() << "interproc: non-call use " << call->getName()
                         << " of " << func.getNameAttr() << "\n";
          return std::nullopt;
        }
        anyCall = true;
        // Guards around the call site hold for the launch inside.
        auto b = derivedExtentBound(actual, depth + 1, call);
        if (!b) {
          if (getenv("DEBUG_BOUND")) {
            llvm::errs() << "interproc: underivable actual in "
                         << call->getParentOfType<FunctionOpInterface>()
                                .getNameAttr()
                         << "\n";
            Value w = actual;
            for (int k = 0; k < 8 && w.getDefiningOp(); ++k) {
              llvm::errs() << "   <- " << *w.getDefiningOp() << "\n";
              if (w.getDefiningOp()->getNumOperands() == 0)
                break;
              w = w.getDefiningOp()->getOperand(0);
            }
          }
          return std::nullopt;
        }
        bound = bound ? std::max(*bound, *b) : *b;
      }
      if (!anyCall && getenv("DEBUG_BOUND"))
        llvm::errs() << "interproc: no direct callers of "
                     << func.getNameAttr() << "\n";
      if (anyCall)
        return bound;
      if (auto g = guardBound(v, anchor))
        return g;
      return enclosingGuardBound(v, anchor);
    }
    if (auto g = guardBound(v, anchor))
      return g;
    return enclosingGuardBound(v, anchor);
  }

  // Bound on a parallel axis implied by the static scratch buffers its iv
  // indexes: a lane past the buffer extent would access out of bounds, so
  // the axis cannot exceed it. Only accesses every lane is guaranteed to
  // execute count -- directly in the body, or under constant-trip loops.
  static std::optional<int64_t> allocaIndexBound(Operation *loop, Block *body,
                                                 Value iv) {
    std::optional<int64_t> bound;
    auto consider = [&](int64_t b) {
      if (!bound || b < *bound)
        bound = b;
    };
    body->walk([&](Operation *op) {
      if (!isa<affine::AffineLoadOp, affine::AffineStoreOp, memref::LoadOp,
               memref::StoreOp>(op))
        return;
      for (Operation *a = op->getParentOp(); a != loop;
           a = a->getParentOp()) {
        auto f = dyn_cast<affine::AffineForOp>(a);
        if (!f || !f.hasConstantBounds() ||
            f.getConstantLowerBound() >= f.getConstantUpperBound())
          return;
      }
      MemRefType MT;
      Value memref;
      if (auto ld = dyn_cast<affine::AffineLoadOp>(op)) {
        MT = ld.getMemRefType();
        memref = ld.getMemRef();
      } else if (auto st = dyn_cast<affine::AffineStoreOp>(op)) {
        MT = st.getMemRefType();
        memref = st.getMemRef();
      } else if (auto mld = dyn_cast<memref::LoadOp>(op)) {
        MT = mld.getMemRefType();
        memref = mld.getMemRef();
      } else {
        auto mst = cast<memref::StoreOp>(op);
        MT = mst.getMemRefType();
        memref = mst.getMemRef();
      }
      if (!isa_and_nonnull<memref::AllocaOp>(memref.getDefiningOp()) ||
          !MT.hasStaticShape())
        return;
      if (isa<affine::AffineLoadOp, affine::AffineStoreOp>(op)) {
        AffineMap map = isa<affine::AffineLoadOp>(op)
                            ? cast<affine::AffineLoadOp>(op).getMap()
                            : cast<affine::AffineStoreOp>(op).getMap();
        auto operands = isa<affine::AffineLoadOp>(op)
                            ? cast<affine::AffineLoadOp>(op).getMapOperands()
                            : cast<affine::AffineStoreOp>(op).getMapOperands();
        for (auto &&[ri, expr] : llvm::enumerate(map.getResults()))
          if (auto de = dyn_cast<AffineDimExpr>(expr))
            if (operands[de.getPosition()] == iv)
              consider(MT.getShape()[ri]);
      } else {
        auto indices = isa<memref::LoadOp>(op)
                           ? cast<memref::LoadOp>(op).getIndices()
                           : cast<memref::StoreOp>(op).getIndices();
        for (auto &&[ri, idx] : llvm::enumerate(indices))
          if (idx == iv)
            consider(MT.getShape()[ri]);
      }
    });
    return bound;
  }

  // A thread-private array lives inside the lane-batched parallel: every
  // lane owns a copy. The raising models buffers as whole tensors, and a
  // store whose index map does not involve the lane axes looks uniform, so
  // reads would collapse to one lane's value. Give the buffer one leading
  // dimension per lane axis and index every access with the lane IVs.
  static void privatizeLaneScratch(Operation *root) {
    SmallVector<memref::AllocaOp> allocas;
    root->walk([&](memref::AllocaOp a) { allocas.push_back(a); });
    for (auto a : allocas) {
      auto par = a->getParentOfType<affine::AffineParallelOp>();
      if (!par)
        continue;
      // Only the nested (thread) parallel batches into lanes; scratch
      // directly under the grid parallel is genuinely shared.
      if (!par->getParentOfType<affine::AffineParallelOp>())
        continue;
      if (par.hasMinMaxBounds())
        continue;
      auto ranges = par.getConstantRanges();
      if (!ranges)
        continue;
      int64_t total = 1;
      bool ok = true;
      for (auto [i, ext] : llvm::enumerate(*ranges)) {
        auto lb = getConstant(par.getLowerBoundMap(i));
        if (!lb || *lb != 0 || par.getSteps()[i] != 1 || ext <= 0) {
          ok = false;
          break;
        }
        total *= ext;
      }
      auto MT = cast<MemRefType>(a.getType());
      if (!ok || !MT.hasStaticShape() ||
          total * MT.getNumElements() > (1 << 16))
        continue;
      SmallVector<Operation *> accesses;
      bool legal = true;
      for (Operation *u : a->getUsers()) {
        bool isAccess =
            (isa<affine::AffineLoadOp, memref::LoadOp>(u) &&
             u->getOperand(0) == a.getResult()) ||
            (isa<affine::AffineStoreOp, memref::StoreOp>(u) &&
             u->getOperand(1) == a.getResult());
        if (!isAccess || !par->isProperAncestor(u)) {
          legal = false;
          break;
        }
        accesses.push_back(u);
      }
      if (!legal)
        continue;

      SmallVector<int64_t> newShape(ranges->begin(), ranges->end());
      newShape.append(MT.getShape().begin(), MT.getShape().end());
      OpBuilder b(par);
      auto newAlloca = memref::AllocaOp::create(
          b, a.getLoc(),
          MemRefType::get(newShape, MT.getElementType(),
                          MemRefLayoutAttrInterface{}, MT.getMemorySpace()));
      auto ivs = par.getIVs();
      unsigned K = ivs.size();
      for (Operation *u : accesses) {
        if (auto ld = dyn_cast<memref::LoadOp>(u)) {
          OpBuilder ub(u);
          SmallVector<Value> idx(ivs.begin(), ivs.end());
          idx.append(ld.getIndices().begin(), ld.getIndices().end());
          Value nl = memref::LoadOp::create(ub, u->getLoc(), newAlloca, idx);
          u->getResult(0).replaceAllUsesWith(nl);
          u->erase();
          continue;
        }
        if (auto st = dyn_cast<memref::StoreOp>(u)) {
          OpBuilder ub(u);
          SmallVector<Value> idx(ivs.begin(), ivs.end());
          idx.append(st.getIndices().begin(), st.getIndices().end());
          memref::StoreOp::create(ub, u->getLoc(), st.getValue(), newAlloca,
                                  idx);
          u->erase();
          continue;
        }
        AffineMap map;
        SmallVector<Value> mapOperands;
        if (auto ld = dyn_cast<affine::AffineLoadOp>(u)) {
          map = ld.getAffineMap();
          mapOperands.assign(ld.getMapOperands().begin(),
                             ld.getMapOperands().end());
        } else {
          auto st = cast<affine::AffineStoreOp>(u);
          map = st.getAffineMap();
          mapOperands.assign(st.getMapOperands().begin(),
                             st.getMapOperands().end());
        }
        unsigned nd = map.getNumDims();
        SmallVector<AffineExpr> exprs;
        for (unsigned k = 0; k < K; ++k)
          exprs.push_back(getAffineDimExpr(nd + k, par.getContext()));
        for (AffineExpr e : map.getResults())
          exprs.push_back(e);
        auto newMap =
            AffineMap::get(nd + K, map.getNumSymbols(), exprs,
                           par.getContext());
        SmallVector<Value> newOperands(mapOperands.begin(),
                                       mapOperands.begin() + nd);
        newOperands.append(ivs.begin(), ivs.end());
        newOperands.append(mapOperands.begin() + nd, mapOperands.end());
        OpBuilder ub(u);
        if (auto ld = dyn_cast<affine::AffineLoadOp>(u)) {
          Value nl = affine::AffineLoadOp::create(ub, u->getLoc(), newAlloca,
                                                  newMap, newOperands);
          u->getResult(0).replaceAllUsesWith(nl);
        } else {
          auto st = cast<affine::AffineStoreOp>(u);
          affine::AffineStoreOp::create(ub, u->getLoc(), st.getValue(),
                                        newAlloca, newMap, newOperands);
        }
        u->erase();
      }
      a.erase();
    }
  }

  // A parallel axis whose extent is dynamic but provably bounded (a block
  // size clamped by a min against a constant, or an iv indexing a static
  // scratch buffer) batches at the bound instead of peeling to a serial
  // loop: the axis becomes constant-extent and the body sits behind an
  // `iv < extent` guard, which the masking machinery already understands.
  // Barriers over the axis then stay batched no-ops.
  static void boundParallelAxes(Operation *root) {
    SmallVector<affine::AffineParallelOp> worklist;
    root->walk([&](affine::AffineParallelOp par) { worklist.push_back(par); });
    for (auto par : worklist) {
      if (!par.getReductions().empty())
        continue;
      unsigned n = par.getNumDims();
      struct BoundedDim {
        unsigned dim;
        int64_t bound;
        Value extent;
      };
      SmallVector<BoundedDim> bounded;
      bool dbg = getenv("DEBUG_BOUND") != nullptr;
      for (unsigned i = 0; i < n; ++i) {
        auto lb = getConstant(par.getLowerBoundMap(i));
        if (!lb || *lb != 0 || par.getSteps()[i] != 1) {
          if (dbg)
            llvm::errs() << "bpa dim " << i << ": lb/step skip\n";
          continue;
        }
        if (getConstant(par.getUpperBoundMap(i)))
          continue;
        auto um = par.getUpperBoundMap(i);
        if (um.getNumResults() != 1)
          continue;
        Value ext;
        if (auto se = dyn_cast<AffineSymbolExpr>(um.getResult(0)))
          ext = par.getUpperBoundsOperands()[par.getUpperBoundsMap()
                                                .getNumDims() +
                                            se.getPosition()];
        else if (auto de = dyn_cast<AffineDimExpr>(um.getResult(0)))
          ext = par.getUpperBoundsOperands()[de.getPosition()];
        else {
          if (dbg)
            llvm::errs() << "bpa dim " << i << ": ub form skip "
                         << par.getUpperBoundMap(i) << "\n";
          continue;
        }
        if (dbg) {
          llvm::errs() << "bpa dim " << i << " ext: " << ext << "\n";
        }
        if (auto c = derivedExtentBound(ext, 0, par))
          bounded.push_back({i, *c, ext});
        else if (auto ab = allocaIndexBound(par.getOperation(), par.getBody(),
                                            par.getBody()->getArgument(i)))
          bounded.push_back({i, *ab, ext});
        else if (getenv("DEBUG_BOUND")) {
          llvm::errs() << "unbounded extent: " << ext << "\n";
          std::function<void(Value, int)> dump = [&](Value v, int ind) {
            if (ind > 4)
              return;
            for (int k = 0; k < ind; ++k)
              llvm::errs() << "  ";
            if (auto ba = dyn_cast<BlockArgument>(v)) {
              llvm::errs() << "blockarg " << ba.getArgNumber() << " of "
                           << ba.getOwner()->getParentOp()->getName()
                           << "\n";
              return;
            }
            if (!v.getDefiningOp()) {
              llvm::errs() << "?\n";
              return;
            }
            llvm::errs() << *v.getDefiningOp() << "\n";
            for (Value o : v.getDefiningOp()->getOperands())
              dump(o, ind + 1);
          };
          dump(ext, 1);
        }
      }
      if (bounded.empty())
        continue;

      OpBuilder b(par);
      Location loc = par.getLoc();
      SmallVector<AffineExpr> lbounds, ubounds;
      SmallVector<int32_t> lboundGroup, uboundGroup;
      SmallVector<int64_t> steps;
      for (unsigned i = 0; i < n; ++i) {
        auto lm = par.getLowerBoundMap(i);
        lbounds.append(lm.getResults().begin(), lm.getResults().end());
        lboundGroup.push_back(lm.getNumResults());
        auto um = par.getUpperBoundMap(i);
        auto bit = llvm::find_if(
            bounded, [&](const BoundedDim &bd) { return bd.dim == i; });
        if (bit != bounded.end()) {
          ubounds.push_back(getAffineConstantExpr(bit->bound, par.getContext()));
          uboundGroup.push_back(1);
        } else {
          ubounds.append(um.getResults().begin(), um.getResults().end());
          uboundGroup.push_back(um.getNumResults());
        }
        steps.push_back(par.getSteps()[i]);
      }
      // When every bound came out constant, drop the stale symbols and
      // operands entirely: downstream batching expects clean constant maps.
      bool allConstant =
          llvm::all_of(lbounds,
                       [](AffineExpr e) { return isa<AffineConstantExpr>(e); }) &&
          llvm::all_of(ubounds,
                       [](AffineExpr e) { return isa<AffineConstantExpr>(e); });
      unsigned lbDims = par.getLowerBoundsMap().getNumDims(),
               lbSyms = par.getLowerBoundsMap().getNumSymbols(),
               ubDims = par.getUpperBoundsMap().getNumDims(),
               ubSyms = par.getUpperBoundsMap().getNumSymbols();
      SmallVector<Value> mapOperands(par.getOperands());
      if (allConstant) {
        lbDims = lbSyms = ubDims = ubSyms = 0;
        mapOperands.clear();
      }
      auto newPar = affine::AffineParallelOp::create(
          b, loc, TypeRange(), b.getArrayAttr({}),
          AffineMapAttr::get(
              AffineMap::get(lbDims, lbSyms, lbounds, par.getContext())),
          b.getI32TensorAttr(lboundGroup),
          AffineMapAttr::get(
              AffineMap::get(ubDims, ubSyms, ubounds, par.getContext())),
          b.getI32TensorAttr(uboundGroup), b.getI64ArrayAttr(steps),
          mapOperands);
      Block *blk = new Block();
      SmallVector<Value> ivRepl;
      for (unsigned i = 0; i < n; ++i)
        ivRepl.push_back(blk->addArgument(b.getIndexType(), loc));
      newPar.getRegion().push_back(blk);
      b.setInsertionPointToEnd(blk);
      auto yield = affine::AffineYieldOp::create(b, loc);

      // Guard: every bounded axis only runs its true extent.
      SmallVector<AffineExpr> constraints;
      SmallVector<bool> eqs;
      SmallVector<Value> setOperands;
      for (auto [k, bd] : llvm::enumerate(bounded)) {
        constraints.push_back(getAffineSymbolExpr(k, par.getContext()) -
                              getAffineDimExpr(k, par.getContext()) - 1);
        eqs.push_back(false);
      }
      auto iset =
          IntegerSet::get(bounded.size(), bounded.size(), constraints, eqs);
      for (auto &bd : bounded)
        setOperands.push_back(ivRepl[bd.dim]);
      for (auto &bd : bounded)
        setOperands.push_back(bd.extent);
      b.setInsertionPoint(yield);
      auto ifOp = affine::AffineIfOp::create(b, loc, TypeRange(), iset,
                                             setOperands,
                                             /*withElseRegion=*/false);
      Block *oldBody = par.getBody();
      for (unsigned i = 0; i < n; ++i)
        oldBody->getArgument(i).replaceAllUsesWith(ivRepl[i]);
      Block *thenBlk = ifOp.getThenBlock();
      thenBlk->getOperations().splice(
          std::prev(thenBlk->getOperations().end()), oldBody->getOperations(),
          oldBody->getOperations().begin(),
          std::prev(oldBody->getOperations().end()));
      par.erase();
    }
  }

  // The same batching-by-bound for a parallel-marked affine.for that never
  // became an affine.parallel: constant trip count at the bound, body behind
  // an `iv < extent` guard.
  static void boundParallelFors(Operation *root) {
    // The loops needing a constant trip count are the ones barriers span:
    // their ivs appear as barrier operands.
    llvm::SetVector<Operation *> forSet;
    root->walk([&](enzymexla::BarrierOp bar) {
      for (Value iv : bar->getOperands())
        if (auto ba = dyn_cast<BlockArgument>(iv))
          if (auto f =
                  dyn_cast<affine::AffineForOp>(ba.getOwner()->getParentOp()))
            forSet.insert(f);
    });
    SmallVector<affine::AffineForOp> fors;
    for (Operation *f : forSet)
      fors.push_back(cast<affine::AffineForOp>(f));
    if (getenv("DEBUG_BOUND")) {
      root->walk([&](enzymexla::BarrierOp bar) {
        llvm::errs() << "bar operands:";
        for (Value iv : bar->getOperands()) {
          if (auto ba = dyn_cast<BlockArgument>(iv))
            llvm::errs() << " arg-of-"
                         << ba.getOwner()->getParentOp()->getName();
          else if (iv.getDefiningOp())
            llvm::errs() << " " << iv.getDefiningOp()->getName();
        }
        llvm::errs() << "\n";
        return WalkResult::interrupt();
      });
    }
    for (auto f : fors) {
      if (f.hasConstantUpperBound())
        continue;
      if (!f.hasConstantLowerBound() || f.getConstantLowerBound() != 0 ||
          f.getStepAsInt() != 1)
        continue;
      auto um = f.getUpperBoundMap();
      if (um.getNumResults() != 1)
        continue;
      Value ext;
      if (auto se = dyn_cast<AffineSymbolExpr>(um.getResult(0)))
        ext = f.getUpperBoundOperands()[um.getNumDims() + se.getPosition()];
      else if (auto de = dyn_cast<AffineDimExpr>(um.getResult(0)))
        ext = f.getUpperBoundOperands()[de.getPosition()];
      else
        continue;
      auto b = derivedExtentBound(ext, 0, f);
      if (!b)
        b = allocaIndexBound(f, f.getBody(), f.getInductionVar());
      if (!b || *b <= 0)
        continue;

      OpBuilder bld(f);
      Location loc = f.getLoc();
      auto newFor = affine::AffineForOp::create(bld, loc, 0, *b, 1);
      newFor->setAttrs(f->getAttrs());
      Block *nb = newFor.getBody();
      bld.setInsertionPointToStart(nb);
      auto iset = IntegerSet::get(
          1, 1,
          {getAffineSymbolExpr(0, f.getContext()) -
           getAffineDimExpr(0, f.getContext()) - 1},
          {false});
      Value setOperands[] = {newFor.getInductionVar(), ext};
      auto ifOp = affine::AffineIfOp::create(bld, loc, TypeRange(), iset,
                                             setOperands,
                                             /*withElseRegion=*/false);
      Block *oldBody = f.getBody();
      oldBody->getArgument(0).replaceAllUsesWith(newFor.getInductionVar());
      Block *thenBlk = ifOp.getThenBlock();
      thenBlk->getOperations().splice(
          std::prev(thenBlk->getOperations().end()), oldBody->getOperations(),
          oldBody->getOperations().begin(),
          std::prev(oldBody->getOperations().end()));
      f.erase();
    }
  }

  static void peelDynamicParallelDims(Operation *root) {
    SmallVector<affine::AffineParallelOp> worklist;
    root->walk([&](affine::AffineParallelOp par) { worklist.push_back(par); });
    for (auto par : worklist) {
      if (!par.getReductions().empty())
        continue;
      unsigned n = par.getNumDims();
      SmallVector<unsigned> dyn, stat;
      for (unsigned i = 0; i < n; ++i) {
        if (getConstant(par.getLowerBoundMap(i)) &&
            getConstant(par.getUpperBoundMap(i)))
          stat.push_back(i);
        else
          dyn.push_back(i);
      }
      if (dyn.empty())
        continue;

      OpBuilder b(par);
      Location loc = par.getLoc();
      SmallVector<Value> ivRepl(n);
      for (unsigned idx : dyn) {
        auto forOp = affine::AffineForOp::create(
            b, loc, par.getLowerBoundsOperands(), par.getLowerBoundMap(idx),
            par.getUpperBoundsOperands(), par.getUpperBoundMap(idx),
            par.getSteps()[idx]);
        forOp->setAttr("enzymexla.parallel", b.getUnitAttr());
        ivRepl[idx] = forOp.getInductionVar();
        b.setInsertionPointToStart(forOp.getBody());
      }

      Block *target;
      if (!stat.empty()) {
        SmallVector<AffineExpr> lbounds, ubounds;
        SmallVector<int32_t> lboundGroup, uboundGroup;
        SmallVector<int64_t> steps;
        for (unsigned idx : stat) {
          auto lm = par.getLowerBoundMap(idx);
          auto um = par.getUpperBoundMap(idx);
          lbounds.append(lm.getResults().begin(), lm.getResults().end());
          ubounds.append(um.getResults().begin(), um.getResults().end());
          lboundGroup.push_back(lm.getNumResults());
          uboundGroup.push_back(um.getNumResults());
          steps.push_back(par.getSteps()[idx]);
        }
        auto inner = affine::AffineParallelOp::create(
            b, loc, TypeRange(), b.getArrayAttr({}),
            AffineMapAttr::get(
                AffineMap::get(par.getLowerBoundsMap().getNumDims(),
                               par.getLowerBoundsMap().getNumSymbols(), lbounds,
                               par.getContext())),
            b.getI32TensorAttr(lboundGroup),
            AffineMapAttr::get(
                AffineMap::get(par.getUpperBoundsMap().getNumDims(),
                               par.getUpperBoundsMap().getNumSymbols(), ubounds,
                               par.getContext())),
            b.getI32TensorAttr(uboundGroup), b.getI64ArrayAttr(steps),
            par.getOperands());
        Block *blk = new Block();
        for (auto [j, idx] : llvm::enumerate(stat))
          ivRepl[idx] = blk->addArgument(b.getIndexType(), loc);
        inner.getRegion().push_back(blk);
        b.setInsertionPointToEnd(blk);
        affine::AffineYieldOp::create(b, loc);
        target = blk;
      } else {
        target = b.getInsertionBlock();
      }

      Block *oldBody = par.getBody();
      for (unsigned i = 0; i < n; ++i)
        oldBody->getArgument(i).replaceAllUsesWith(ivRepl[i]);
      target->getOperations().splice(std::prev(target->getOperations().end()),
                                     oldBody->getOperations(),
                                     oldBody->getOperations().begin(),
                                     std::prev(oldBody->getOperations().end()));
      par.erase();
    }
  }

  // Residue of an integer/index value modulo m, when the defining arith
  // chain pins it down. Extension and truncation preserve residues for the
  // power-of-two moduli struct layouts produce.
  static std::optional<int64_t> staticResidue(Value v, int64_t m,
                                              unsigned depth = 0) {
    if (m == 1)
      return 0;
    if (depth > 16)
      return std::nullopt;
    APInt cst;
    if (matchPattern(v, m_ConstantInt(&cst)))
      return ((cst.getSExtValue() % m) + m) % m;
    Operation *op = v.getDefiningOp();
    if (!op)
      return std::nullopt;
    if (isa<arith::IndexCastOp, arith::IndexCastUIOp, arith::ExtSIOp,
            arith::ExtUIOp>(op) ||
        (isa<arith::TruncIOp>(op) && llvm::isPowerOf2_64(m)))
      return staticResidue(op->getOperand(0), m, depth + 1);
    if (auto add = dyn_cast<arith::AddIOp>(op)) {
      auto a = staticResidue(add.getLhs(), m, depth + 1);
      auto b = staticResidue(add.getRhs(), m, depth + 1);
      if (a && b)
        return (*a + *b) % m;
      return std::nullopt;
    }
    if (auto mul = dyn_cast<arith::MulIOp>(op)) {
      auto a = staticResidue(mul.getLhs(), m, depth + 1);
      auto b = staticResidue(mul.getRhs(), m, depth + 1);
      if (a && b)
        return (*a * *b) % m;
      if ((a && *a == 0) || (b && *b == 0))
        return 0;
      return std::nullopt;
    }
    if (auto shl = dyn_cast<arith::ShLIOp>(op)) {
      APInt sh;
      if (matchPattern(shl.getRhs(), m_ConstantInt(&sh)) &&
          sh.getZExtValue() < 63) {
        int64_t f = (int64_t(1) << sh.getZExtValue()) % m;
        if (f == 0)
          return 0;
        auto a = staticResidue(shl.getLhs(), m, depth + 1);
        if (a)
          return (*a * f) % m;
      }
      return std::nullopt;
    }
    return std::nullopt;
  }

  // Struct-element scratch (e.g. reduce-with-location value/index pairs)
  // cannot become a tensor whole. Every access reaches it through a flat
  // primitive view whose affine index resolves to a fixed byte offset within
  // the struct, so the array-of-structs splits into one primitive scratch
  // per field, with whole-struct integer moves split into their fields.
  static void splitStructScratch(Operation *root) {
    SmallVector<memref::AllocaOp> allocas;
    root->walk([&](memref::AllocaOp a) { allocas.push_back(a); });
    for (auto alloca : allocas) {
      auto MT = alloca.getType();
      if (!MT.hasStaticShape())
        continue;
      auto ST = dyn_cast<LLVM::LLVMStructType>(MT.getElementType());
      if (!ST)
        continue;
      DataLayout dl = DataLayout::closest(alloca);
      struct Field {
        uint64_t off, size;
        Type ty;
      };
      SmallVector<Field> fields;
      uint64_t byte = 0;
      bool ok = true;
      for (Type member : ST.getBody()) {
        if (!member.isIntOrFloat()) {
          ok = false;
          break;
        }
        if (!ST.isPacked())
          byte = llvm::alignTo(byte, dl.getTypeABIAlignment(member));
        fields.push_back({byte, dl.getTypeSize(member), member});
        byte += dl.getTypeSize(member);
      }
      if (!ok)
        continue;
      int64_t pairSize = dl.getTypeSize(ST);

      struct FieldAccess {
        Operation *op;
        unsigned field;
        AffineMap pairMap;
        bool bitcast;
      };
      struct WideAccess {
        Operation *op;
        SmallVector<unsigned> covered;
        uint64_t base;
        AffineMap pairMap;
      };
      struct DynAccess {
        Operation *op;
        unsigned field;
        int64_t q;
        bool bitcast;
      };
      SmallVector<FieldAccess> fieldAccesses;
      SmallVector<WideAccess> wideAccesses;
      SmallVector<DynAccess> dynAccesses;
      SmallVector<enzymexla::Pointer2MemrefOp> views;
      SmallVector<enzymexla::Memref2PointerOp> casts;

      auto classify = [&](Operation *op, Type elemTy,
                          AffineMap map) -> bool {
        if (map.getNumResults() != 1)
          return false;
        int64_t s = dl.getTypeSize(elemTy);
        AffineExpr byteExpr = map.getResult(0) * s;
        AffineExpr rem = simplifyAffineExpr(byteExpr % pairSize,
                                            map.getNumDims(),
                                            map.getNumSymbols());
        auto remCst = dyn_cast<AffineConstantExpr>(rem);
        if (!remCst)
          return false;
        uint64_t off = remCst.getValue();
        AffineMap pairMap = AffineMap::get(
            map.getNumDims(), map.getNumSymbols(),
            simplifyAffineExpr(byteExpr.floorDiv(pairSize), map.getNumDims(),
                               map.getNumSymbols()));
        for (auto &&[i, f] : llvm::enumerate(fields))
          if (f.off == off && f.size == (uint64_t)s) {
            fieldAccesses.push_back({op, (unsigned)i, pairMap,
                                     /*bitcast=*/f.ty != elemTy});
            return true;
          }
        // A wider integer move covering whole fields splits into them.
        if (!isa<IntegerType>(elemTy))
          return false;
        SmallVector<unsigned> covered;
        for (auto &&[i, f] : llvm::enumerate(fields)) {
          if (f.off + f.size <= off || f.off >= off + s)
            continue;
          if (f.off < off || f.off + f.size > off + s ||
              !isa<IntegerType>(f.ty))
            return false;
          covered.push_back(i);
        }
        if (covered.empty())
          return false;
        wideAccesses.push_back({op, covered, off, pairMap});
        return true;
      };

      // The same classification for accesses whose index arrives through
      // plain arithmetic instead of an affine map: the field is fixed when
      // the index has a static residue modulo the per-struct element count.
      auto classifyDyn = [&](Operation *op, Type elemTy, Value idx) -> bool {
        int64_t s = dl.getTypeSize(elemTy);
        if (!s || pairSize % s)
          return false;
        int64_t q = pairSize / s;
        auto r = staticResidue(idx, q);
        if (!r)
          return false;
        uint64_t off = (uint64_t)*r * s;
        for (auto &&[i, f] : llvm::enumerate(fields))
          if (f.off == off && f.size == (uint64_t)s) {
            dynAccesses.push_back({op, (unsigned)i, q,
                                   /*bitcast=*/f.ty != elemTy});
            return true;
          }
        return false;
      };

      SmallVector<Operation *> spaceCasts;
      llvm::MapVector<Operation *, Value> pairGeps;
      llvm::SetVector<Operation *> pairCopies;
      DenseSet<Value> basePtrs;
      bool viewedOnly = true;
      for (Operation *user : alloca->getUsers()) {
        auto m2p = dyn_cast<enzymexla::Memref2PointerOp>(user);
        if (!m2p) {
          viewedOnly = false;
          break;
        }
        casts.push_back(m2p);
        basePtrs.insert(m2p.getResult());
        SmallVector<Operation *> viewUsers(m2p->getUsers());
        for (unsigned vi = 0; vi < viewUsers.size() && viewedOnly; ++vi) {
          Operation *viewUser = viewUsers[vi];
          if (isa<LLVM::AddrSpaceCastOp>(viewUser)) {
            spaceCasts.push_back(viewUser);
            basePtrs.insert(viewUser->getResult(0));
            viewUsers.append(viewUser->getUsers().begin(),
                             viewUser->getUsers().end());
            continue;
          }
          // A copy of pair zero addresses the scratch base directly.
          if (auto mc = dyn_cast<LLVM::MemcpyOp>(viewUser)) {
            APInt len;
            if (mc.getIsVolatile() ||
                !matchPattern(mc.getLen(), m_ConstantInt(&len)) ||
                len.getSExtValue() != pairSize) {
              viewedOnly = false;
              break;
            }
            pairCopies.insert(mc);
            continue;
          }
          // A whole-struct move arrives as a fixed-size memcpy between
          // struct-strided geps into the scratch: split it per field.
          if (auto gep = dyn_cast<LLVM::GEPOp>(viewUser)) {
            auto idxs = gep.getIndices();
            if (idxs.size() != 1 ||
                (int64_t)dl.getTypeSize(gep.getElemType()) != pairSize) {
              viewedOnly = false;
              break;
            }
            bool copiesOnly = true;
            for (Operation *gu : gep->getUsers()) {
              auto mc = dyn_cast<LLVM::MemcpyOp>(gu);
              APInt len;
              if (!mc || mc.getIsVolatile() ||
                  !matchPattern(mc.getLen(), m_ConstantInt(&len)) ||
                  len.getSExtValue() != pairSize) {
                copiesOnly = false;
                break;
              }
              pairCopies.insert(mc);
            }
            if (!copiesOnly) {
              viewedOnly = false;
              break;
            }
            Value gepIdx;
            if (!gep.getDynamicIndices().empty()) {
              gepIdx = gep.getDynamicIndices()[0];
            } else {
              OpBuilder gb(gep);
              gepIdx = arith::ConstantIndexOp::create(
                  gb, gep.getLoc(), cast<IntegerAttr>(idxs[0]).getInt());
            }
            pairGeps.insert({gep, gepIdx});
            continue;
          }
          auto p2m = dyn_cast<enzymexla::Pointer2MemrefOp>(viewUser);
          if (!p2m || p2m.getType().getRank() != 1 ||
              !p2m.getType().getElementType().isIntOrFloat()) {
            viewedOnly = false;
            break;
          }
          for (Operation *access : p2m->getUsers()) {
            if (auto ld = dyn_cast<affine::AffineLoadOp>(access)) {
              if (classify(ld, ld.getType(), ld.getMap()))
                continue;
            } else if (auto st = dyn_cast<affine::AffineStoreOp>(access)) {
              if (st.getValueToStore() != p2m.getResult() &&
                  classify(st, st.getValueToStore().getType(), st.getMap()))
                continue;
            } else if (auto mld = dyn_cast<memref::LoadOp>(access)) {
              if (mld.getIndices().size() == 1 &&
                  classifyDyn(mld, mld.getType(), mld.getIndices()[0]))
                continue;
            } else if (auto mst = dyn_cast<memref::StoreOp>(access)) {
              if (mst.getMemRef() == p2m.getResult() &&
                  mst.getValueToStore() != p2m.getResult() &&
                  mst.getIndices().size() == 1 &&
                  classifyDyn(mst, mst.getValueToStore().getType(),
                              mst.getIndices()[0]))
                continue;
            }
            viewedOnly = false;
            break;
          }
          if (!viewedOnly)
            break;
          views.push_back(p2m);
        }
        if (!viewedOnly)
          break;
      }
      // Every pair copy must connect two classified geps or scratch bases.
      auto copyEnd = [&](Value p) {
        return basePtrs.contains(p) ||
               (p.getDefiningOp() && pairGeps.count(p.getDefiningOp()));
      };
      for (Operation *mc : pairCopies) {
        auto cp = cast<LLVM::MemcpyOp>(mc);
        if (!copyEnd(cp.getDst()) || !copyEnd(cp.getSrc()))
          viewedOnly = false;
      }
      if (!viewedOnly || (fieldAccesses.empty() && wideAccesses.empty()))
        continue;

      OpBuilder b(alloca);
      SmallVector<Value> fieldBufs;
      for (auto &f : fields)
        fieldBufs.push_back(memref::AllocaOp::create(
            b, alloca.getLoc(),
            MemRefType::get({MT.getNumElements()}, f.ty)));

      for (auto &fa : fieldAccesses) {
        if (auto ld = dyn_cast<affine::AffineLoadOp>(fa.op)) {
          OpBuilder ab(ld);
          Value newLd = affine::AffineLoadOp::create(
              ab, ld.getLoc(), fieldBufs[fa.field], fa.pairMap,
              ld.getMapOperands());
          if (fa.bitcast)
            newLd = arith::BitcastOp::create(ab, ld.getLoc(), ld.getType(),
                                             newLd);
          ld.getResult().replaceAllUsesWith(newLd);
          ld.erase();
        } else {
          auto st = cast<affine::AffineStoreOp>(fa.op);
          OpBuilder ab(st);
          Value val = st.getValueToStore();
          if (fa.bitcast)
            val = arith::BitcastOp::create(ab, st.getLoc(),
                                           fields[fa.field].ty, val);
          affine::AffineStoreOp::create(ab, st.getLoc(), val,
                                        fieldBufs[fa.field], fa.pairMap,
                                        st.getMapOperands());
          st.erase();
        }
      }
      for (auto &wa : wideAccesses) {
        if (auto ld = dyn_cast<affine::AffineLoadOp>(wa.op)) {
          OpBuilder ab(ld);
          Location loc = ld.getLoc();
          Type wideTy = ld.getType();
          Value acc = arith::ConstantOp::create(
              ab, loc, wideTy, ab.getIntegerAttr(wideTy, 0));
          for (unsigned i : wa.covered) {
            Value v = affine::AffineLoadOp::create(
                ab, loc, fieldBufs[i], wa.pairMap, ld.getMapOperands());
            Value z = arith::ExtUIOp::create(ab, loc, wideTy, v);
            uint64_t sh = (fields[i].off - wa.base) * 8;
            if (sh) {
              Value shv = arith::ConstantOp::create(
                  ab, loc, wideTy, ab.getIntegerAttr(wideTy, sh));
              z = arith::ShLIOp::create(ab, loc, z, shv);
            }
            acc = arith::OrIOp::create(ab, loc, acc, z);
          }
          ld.getResult().replaceAllUsesWith(acc);
          ld.erase();
        } else {
          auto st = cast<affine::AffineStoreOp>(wa.op);
          OpBuilder ab(st);
          Location loc = st.getLoc();
          Value val = st.getValueToStore();
          Type wideTy = val.getType();
          for (unsigned i : wa.covered) {
            Value part = val;
            uint64_t sh = (fields[i].off - wa.base) * 8;
            if (sh) {
              Value shv = arith::ConstantOp::create(
                  ab, loc, wideTy, ab.getIntegerAttr(wideTy, sh));
              part = arith::ShRUIOp::create(ab, loc, part, shv);
            }
            part = arith::TruncIOp::create(ab, loc, fields[i].ty, part);
            affine::AffineStoreOp::create(ab, loc, part, fieldBufs[i],
                                          wa.pairMap, st.getMapOperands());
          }
          st.erase();
        }
      }
      for (auto &da : dynAccesses) {
        OpBuilder ab(da.op);
        Location loc = da.op->getLoc();
        Value idx = isa<memref::LoadOp>(da.op)
                        ? cast<memref::LoadOp>(da.op).getIndices()[0]
                        : cast<memref::StoreOp>(da.op).getIndices()[0];
        Value pairIdx = idx;
        if (da.q != 1) {
          Value qc = arith::ConstantIndexOp::create(ab, loc, da.q);
          pairIdx = arith::DivUIOp::create(ab, loc, idx, qc);
        }
        if (auto mld = dyn_cast<memref::LoadOp>(da.op)) {
          Value newLd = memref::LoadOp::create(ab, loc, fieldBufs[da.field],
                                               ValueRange{pairIdx});
          if (da.bitcast)
            newLd = arith::BitcastOp::create(ab, loc, mld.getType(), newLd);
          mld.getResult().replaceAllUsesWith(newLd);
          mld.erase();
        } else {
          auto mst = cast<memref::StoreOp>(da.op);
          Value val = mst.getValueToStore();
          if (da.bitcast)
            val = arith::BitcastOp::create(ab, loc, fields[da.field].ty, val);
          memref::StoreOp::create(ab, loc, val, fieldBufs[da.field],
                                  ValueRange{pairIdx});
          mst.erase();
        }
      }
      for (Operation *mc : pairCopies) {
        auto cp = cast<LLVM::MemcpyOp>(mc);
        OpBuilder ab(cp);
        Location loc = cp.getLoc();
        // Constant pair indices stay affine so the accesses raise directly.
        auto toIdx = [&](Value p) -> std::pair<Value, std::optional<int64_t>> {
          if (basePtrs.contains(p))
            return {Value(), 0};
          Value v = pairGeps.find(p.getDefiningOp())->second;
          APInt c;
          if (matchPattern(v, m_ConstantInt(&c)))
            return {Value(), c.getSExtValue()};
          if (!isa<IndexType>(v.getType()))
            v = arith::IndexCastUIOp::create(ab, loc, ab.getIndexType(), v);
          return {v, std::nullopt};
        };
        auto [dstIdx, dstC] = toIdx(cp.getDst());
        auto [srcIdx, srcC] = toIdx(cp.getSrc());
        for (auto &&[i, f] : llvm::enumerate(fields)) {
          Value v;
          if (srcC)
            v = affine::AffineLoadOp::create(
                ab, loc, fieldBufs[i],
                AffineMap::getConstantMap(*srcC, ab.getContext()),
                ValueRange());
          else
            v = memref::LoadOp::create(ab, loc, fieldBufs[i],
                                       ValueRange{srcIdx});
          if (dstC)
            affine::AffineStoreOp::create(
                ab, loc, v, fieldBufs[i],
                AffineMap::getConstantMap(*dstC, ab.getContext()),
                ValueRange());
          else
            memref::StoreOp::create(ab, loc, v, fieldBufs[i],
                                    ValueRange{dstIdx});
        }
        cp.erase();
      }
      for (auto &g : pairGeps)
        g.first->erase();
      for (auto p2m : views)
        p2m.erase();
      for (Operation *sc : llvm::reverse(spaceCasts))
        sc->erase();
      for (auto m2p : casts)
        m2p.erase();
      alloca.erase();
    }
  }

  // Shared-memory scratch arrives as a static alloca viewed through a
  // memref2pointer/pointer2memref round trip that changes shape and address
  // space, which no access-based raising can see through. When the alloca is
  // only ever read and written through such flat views, replace the whole
  // chain with one flat static alloca the raising handles directly.
  static void flattenViewedScratch(Operation *root) {
    SmallVector<memref::AllocaOp> allocas;
    root->walk([&](memref::AllocaOp a) { allocas.push_back(a); });
    for (auto alloca : allocas) {
      auto MT = alloca.getType();
      if (!MT.hasStaticShape())
        continue;
      // Views reach the scratch through chains of address-space casts and
      // constant-offset geps; each view carries the byte offset its chain
      // accumulated.
      SmallVector<std::pair<enzymexla::Pointer2MemrefOp, int64_t>> views;
      SmallVector<Operation *> chainOps;
      SmallVector<Operation *> directAccesses;
      bool viewedOnly = true;
      if (!MT.getElementType().isIntOrFloat())
        continue;
      int64_t elemBytes =
          (MT.getElementType().getIntOrFloatBitWidth() + 7) / 8;
      SmallVector<std::pair<Value, int64_t>> ptrWork;
      for (Operation *user : alloca->getUsers()) {
        auto m2p = dyn_cast<enzymexla::Memref2PointerOp>(user);
        if (!m2p) {
          // A direct access can move to the flat buffer with its indexing
          // linearized; anything else keeps the chain intact.
          if (isa<affine::AffineLoadOp, affine::AffineStoreOp,
                  memref::LoadOp>(user)) {
            directAccesses.push_back(user);
            continue;
          }
          if (auto st = dyn_cast<memref::StoreOp>(user)) {
            if (st.getMemRef() == alloca.getResult()) {
              directAccesses.push_back(user);
              continue;
            }
          }
          viewedOnly = false;
          break;
        }
        chainOps.push_back(m2p);
        ptrWork.push_back({m2p.getResult(), 0});
      }
      while (viewedOnly && !ptrWork.empty()) {
        auto [ptr, off] = ptrWork.pop_back_val();
        for (Operation *viewUser : ptr.getUsers()) {
          if (isa<LLVM::AddrSpaceCastOp>(viewUser)) {
            chainOps.push_back(viewUser);
            ptrWork.push_back({viewUser->getResult(0), off});
            continue;
          }
          if (auto gep = dyn_cast<LLVM::GEPOp>(viewUser)) {
            // Only all-constant geps carry a static byte offset.
            int64_t gepOff = 0;
            bool constGep = gep.getDynamicIndices().empty() &&
                            gep.getIndices().size() >= 1;
            if (constGep) {
              DataLayout dl = DataLayout::closest(gep);
              Type cur = gep.getElemType();
              auto idxs = gep.getIndices();
              gepOff = cast<IntegerAttr>(idxs[0]).getInt() *
                       (int64_t)dl.getTypeSize(cur);
              for (unsigned i = 1; constGep && i < idxs.size(); ++i) {
                int64_t want = cast<IntegerAttr>(idxs[i]).getInt();
                if (auto AT = dyn_cast<LLVM::LLVMArrayType>(cur)) {
                  cur = AT.getElementType();
                  gepOff += want * (int64_t)dl.getTypeSize(cur);
                } else {
                  constGep = false;
                }
              }
            }
            if (!constGep) {
              viewedOnly = false;
              break;
            }
            chainOps.push_back(gep);
            ptrWork.push_back({gep.getResult(), off + gepOff});
            continue;
          }
          auto p2m = dyn_cast<enzymexla::Pointer2MemrefOp>(viewUser);
          if (!p2m || p2m.getType().getRank() != 1 ||
              p2m.getType().getElementType() != MT.getElementType() ||
              off % elemBytes != 0 ||
              (p2m.getType().hasStaticShape() &&
               off / elemBytes + p2m.getType().getShape()[0] >
                   MT.getNumElements())) {
            viewedOnly = false;
            break;
          }
          if (off != 0) {
            // Offset views only rewrite through plain accesses.
            if (!llvm::all_of(p2m->getUsers(), [&](Operation *u) {
                  if (isa<affine::AffineLoadOp, memref::LoadOp>(u))
                    return u->getOperand(0) == p2m.getResult();
                  if (isa<affine::AffineStoreOp, memref::StoreOp>(u))
                    return u->getOperand(1) == p2m.getResult();
                  return false;
                })) {
              viewedOnly = false;
              break;
            }
          }
          views.push_back({p2m, off / elemBytes});
        }
      }
      if (!viewedOnly || views.empty())
        continue;
      OpBuilder b(alloca);
      auto flatTy = MemRefType::get({MT.getNumElements()}, MT.getElementType());
      auto flat = memref::AllocaOp::create(b, alloca.getLoc(), flatTy);
      for (auto [p2m, elemOff] : views) {
        if (elemOff != 0) {
          for (Operation *u :
               llvm::make_early_inc_range(p2m->getUsers())) {
            if (auto ld = dyn_cast<affine::AffineLoadOp>(u)) {
              auto m = ld.getMap();
              ld.setMap(AffineMap::get(m.getNumDims(), m.getNumSymbols(),
                                       m.getResult(0) + elemOff));
            } else if (auto st = dyn_cast<affine::AffineStoreOp>(u)) {
              auto m = st.getMap();
              st.setMap(AffineMap::get(m.getNumDims(), m.getNumSymbols(),
                                       m.getResult(0) + elemOff));
            } else {
              OpBuilder ab(u);
              unsigned idxPos = isa<memref::LoadOp>(u) ? 1 : 2;
              Value c = arith::ConstantIndexOp::create(ab, u->getLoc(),
                                                       elemOff);
              Value ni = arith::AddIOp::create(ab, u->getLoc(),
                                               u->getOperand(idxPos), c);
              u->setOperand(idxPos, ni);
            }
          }
        }
        p2m.getResult().replaceAllUsesWith(flat.getResult());
        p2m.erase();
      }
      for (Operation *c : llvm::reverse(chainOps))
        if (c->use_empty())
          c->erase();
      auto linearizeValues = [&](OpBuilder &ab, Location loc,
                                 ValueRange idxs) -> Value {
        Value lin = arith::ConstantIndexOp::create(ab, loc, 0);
        int64_t stride = 1;
        SmallVector<Value> scaled;
        for (int64_t d = MT.getRank() - 1; d >= 0; --d) {
          Value term = idxs[d];
          if (stride != 1) {
            Value c = arith::ConstantIndexOp::create(ab, loc, stride);
            term = arith::MulIOp::create(ab, loc, term, c);
          }
          lin = arith::AddIOp::create(ab, loc, lin, term);
          stride *= MT.getShape()[d];
        }
        return lin;
      };
      for (Operation *access : directAccesses) {
        if (auto mload = dyn_cast<memref::LoadOp>(access)) {
          OpBuilder ab(mload);
          Value lin =
              linearizeValues(ab, mload.getLoc(), mload.getIndices());
          Value idxs[] = {lin};
          mload.getMemrefMutable().assign(flat);
          mload.getIndicesMutable().assign(idxs);
          continue;
        }
        if (auto mstore = dyn_cast<memref::StoreOp>(access)) {
          OpBuilder ab(mstore);
          Value lin =
              linearizeValues(ab, mstore.getLoc(), mstore.getIndices());
          Value idxs[] = {lin};
          mstore.getMemrefMutable().assign(flat);
          mstore.getIndicesMutable().assign(idxs);
          continue;
        }
        AffineMap oldMap = isa<affine::AffineLoadOp>(access)
                               ? cast<affine::AffineLoadOp>(access).getMap()
                               : cast<affine::AffineStoreOp>(access).getMap();
        AffineExpr lin = getAffineConstantExpr(0, oldMap.getContext());
        int64_t stride = 1;
        for (int64_t d = MT.getRank() - 1; d >= 0; --d) {
          lin = lin + oldMap.getResult(d) * stride;
          stride *= MT.getShape()[d];
        }
        auto linMap =
            AffineMap::get(oldMap.getNumDims(), oldMap.getNumSymbols(), lin);
        if (auto load = dyn_cast<affine::AffineLoadOp>(access)) {
          load.setMap(linMap);
          load.getMemrefMutable().assign(flat);
        } else {
          auto store = cast<affine::AffineStoreOp>(access);
          store.setMap(linMap);
          store.getMemrefMutable().assign(flat);
        }
      }
      alloca.erase();
    }
  }

  void runOnOperation() override {
    ParallelContext::Options options{enable_lockstep_for, dump_failed_lockstep,
                                     prefer_while_raising,
                                     strip_llvm_debuginfo};
    std::vector<func::FuncOp> funcs;

    auto context = getOperation()->getContext();

    if (enable_lockstep_for) {

      RewritePatternSet patterns(context);
      patterns.add<PushReductionsDown>(context);
      GreedyRewriteConfig config;
      config.enableFolding();
      if (failed(applyPatternsGreedily(getOperation(), std::move(patterns),
                                       config))) {
        signalPassFailure();
      }
    }

    auto op = getOperation();

    op->walk([&](func::FuncOp func) {
      auto FT = dyn_cast<FunctionType>(func.getFunctionType());

      // Identify raised kernels which takes in memrefs instead of tensors
      if (FT &&
          llvm::all_of(FT.getInputs(),
                       [](Type argTy) { return isa<MemRefType>(argTy); }) &&
          FT.getNumResults() == 0 && FT.getNumInputs() != 0) {
        funcs.push_back(func);
      }
    });

    // Peeling rewrites loops, so it stays scoped to the regions this pass
    // actually raises.
    for (auto func : funcs) {
      inlineAllocaScopes(func);
      for (int round = 0; round < 2; ++round) {
        dropNullPointerSelects(func);
        dropNullBufferAccesses(func);
        distributeGepOverSelect(func);
        forwardPackedScratch(func);
        stripAccessMemorySpaceCasts(func);
        rebaseViewedGeps(func);
        convertRawGepAccesses(func);
        expandBufferBranches(func);
        rewritePointerInduction(func);
        splitStructScratch(func);
        flattenViewedScratch(func);
      }
      dropDeadPointerChains(func);
      boundParallelAxes(func);
      boundParallelFors(func);
      privatizeLaneScratch(func);
      peelDynamicParallelDims(func);
    }

    SymbolTableCollection symbolTable;
    SymbolUserMap userMap(symbolTable, op);

    bool anyRaised = false;
    while (!funcs.empty()) {
      auto kernelFunc = funcs.back();
      ArrayRef<Operation *> users = userMap.getUsers(kernelFunc);
      bool raised = tryRaisingToStableHLO(kernelFunc, users, options);
      anyRaised |= raised;
      if (!raised && err_if_not_fully_raised) {
        llvm::errs() << "failed to raise func: " << *kernelFunc << "\n";
        signalPassFailure();
      }
      funcs.pop_back();
    }
    std::vector<enzymexla::GPUWrapperOp> gwrap;
    op->walk([&](enzymexla::GPUWrapperOp g) { gwrap.push_back(g); });
    for (auto g : gwrap) {
      // Scope inlining hoists scratch allocas to the surrounding function,
      // so the buffer normalizations must see the whole function, not just
      // the wrapper region; the rewrites also expose one another (a rebase
      // creates the direct views a flatten wants), so iterate once more.
      Operation *root = g->getParentOfType<FunctionOpInterface>();
      if (!root)
        root = g;
      inlineAllocaScopes(root);
      for (int round = 0; round < 2; ++round) {
        dropNullPointerSelects(root);
        dropNullBufferAccesses(root);
        distributeGepOverSelect(root);
        forwardPackedScratch(root);
        stripAccessMemorySpaceCasts(root);
        rebaseViewedGeps(root);
        convertRawGepAccesses(root);
        expandBufferBranches(root);
        rewritePointerInduction(root);
        splitStructScratch(root);
        flattenViewedScratch(root);
      }
      hoistWrapperInvariantScalars(g);
      dropDeadPointerChains(root);
      boundParallelAxes(g);
      boundParallelFors(g);
      privatizeLaneScratch(g);
      peelDynamicParallelDims(g);
    }
    size_t raised_count = 0;
    for (auto g : gwrap) {
      auto modOp = g->getParentOfType<ModuleOp>();
      Block *body = &g->getRegion(0).front();
      Block *newBlock = new Block();

      IRMapping mapping;
      llvm::DenseMap<Value, affine::AffineValueMap> maps;
      mapping.map(body, newBlock);

      SetVector<Value> operands;
      {
        SetVector<Value> operands0;
        getUsedValuesDefinedAbove(g->getRegion(0), operands0);

        DenseMap<Value, Value> buffered;
        SmallVector<Operation *> loads;

        for (auto arg : operands0) {

          Attribute attr;

          // Only splat what a tensor can hold; a pointer constant (null, a
          // global's address) falls through to the pointer handling and, if
          // unhandled there, to the unraised-operand report.
          if (isa<IntegerType, FloatType, IndexType>(arg.getType()) &&
              matchPattern(arg, m_Constant(&attr))) {
            affine::AffineValueMap accessMap(AffineMap::get(arg.getContext()),
                                             {});

            auto isIndex = isa<IndexType>(arg.getType());
            auto ET = isIndex ? IntegerType::get(arg.getContext(), 64)
                              : arg.getType();
            auto unrankedTensorType = RankedTensorType::get({}, ET);
            OpBuilder builder(arg.getContext());
            builder.setInsertionPointToEnd(newBlock);
            Value newVal;
            if (arg.getDefiningOp<ub::PoisonOp>() ||
                arg.getDefiningOp<LLVM::UndefOp>() ||
                arg.getDefiningOp<LLVM::PoisonOp>() ||
                arg.getDefiningOp<LLVM::ZeroOp>()) {
              // A poison scalar reads as zero, as a rank-0 tensor like every
              // other raised scalar so loop carrying can broadcast it.
              auto newConst = stablehlo::ConstantOp::create(
                  builder,
                  rewriteLocation(arg.getLoc(), options.strip_llvm_debuginfo),
                  unrankedTensorType,
                  SplatElementsAttr::get(unrankedTensorType,
                                         builder.getZeroAttr(ET)));
              newVal = newConst.getResult();
            } else {
              auto newConst = stablehlo::ConstantOp::create(
                  builder,
                  rewriteLocation(arg.getLoc(), options.strip_llvm_debuginfo),
                  unrankedTensorType,
                  SplatElementsAttr::get(
                      unrankedTensorType,
                      ArrayRef<Attribute>(
                          isIndex ? IntegerAttr::get(
                                        ET, cast<IntegerAttr>(attr).getValue())
                                  : attr)));
              newVal = newConst.getResult();
            }
            mapping.map(arg, newVal);
            maps[newVal] = accessMap;
            continue;
          }

          if (auto ic = arg.getDefiningOp<arith::IndexCastOp>()) {
            if (arg.getType().isIndex()) {
              OpBuilder b(g);
              b.setInsertionPointToStart(body);
              auto cl = b.clone(*ic);

              auto found = buffered.find(ic.getOperand());
              if (found != buffered.end()) {
                cast<arith::IndexCastOp>(cl).setOperand(found->second);
              }

              arg.replaceUsesWithIf(
                  cl->getResult(0), [&](OpOperand &opOperand) {
                    return g->isProperAncestor(opOperand.getOwner());
                  });
              arg = ic.getOperand();

              llvm::errs() << " unfolded cast to index new arg: " << arg
                           << ", old arg: " << ic << "\n";
            }
          }

          if (isa<LLVM::LLVMPointerType>(arg.getType())) {
            OpBuilder b(g);
            b.setInsertionPoint(g);
            bool legal = true;
            MemRefType T = nullptr;
            for (auto &U : arg.getUses()) {
              if (g->isProperAncestor(U.getOwner())) {
                auto p2m = dyn_cast<enzymexla::Pointer2MemrefOp>(U.getOwner());
                if (!p2m) {
                  legal = false;
                  llvm::errs()
                      << " non pointermemref user of pointer arg in kernel: "
                      << *U.getOwner() << "\n";
                  break;
                }
                if (!T) {
                  T = p2m.getType();
                } else {
                  if (T != p2m.getType()) {
                    if (T.getElementType().isInteger(8)) {
                      T = p2m.getType();
                      continue;
                    }
                    if (p2m.getType().getElementType().isInteger(8)) {
                      continue;
                    }
                    legal = false;
                    llvm::errs() << " inconsistent pointer2memref type " << T
                                 << " and " << p2m << " \n";
                    break;
                  }
                }
              }
            }

            if (legal) {
              auto cl = enzymexla::Pointer2MemrefOp::create(
                  b,
                  rewriteLocation(arg.getLoc(), options.strip_llvm_debuginfo),
                  T, arg);
              for (auto U : llvm::make_early_inc_range(arg.getUsers())) {
                if (!g->isProperAncestor(U))
                  continue;
                if (U == cl)
                  continue;
                if (U->getResult(0).getType() == T) {
                  U->replaceAllUsesWith(cl);
                  U->erase();
                } else {
                  OpBuilder B(U);
                  U->setOperand(
                      0, enzymexla::Memref2PointerOp::create(
                             B,
                             rewriteLocation(arg.getLoc(),
                                             options.strip_llvm_debuginfo),
                             arg.getType(), cl));
                }
              }
              operands.insert(cl);
              continue;
            }
          }

          if (buffered.find(arg) != buffered.end()) {
            continue;
          }

          if (isa<IntegerType, FloatType, IndexType>(arg.getType())) {
            OpBuilder b(g);
            b.setInsertionPoint(g);
            auto isIndex = isa<IndexType>(arg.getType());
            auto ET = isIndex ? b.getI64Type() : arg.getType();

            auto MT0 = MemRefType::get({}, ET, MemRefLayoutAttrInterface{},
                                       b.getI64IntegerAttr(0));
            auto MT = MemRefType::get({}, ET, MemRefLayoutAttrInterface{},
                                      b.getI64IntegerAttr(1));

            auto res =
                gpu::AllocOp::create(
                    b,
                    rewriteLocation(g.getLoc(), options.strip_llvm_debuginfo),
                    MT, (mlir::Type) nullptr, ValueRange(), ValueRange(),
                    ValueRange())
                    ->getResult(0);

            Block *allocaBlock = getAllocaBlock(g);
            assert(allocaBlock &&
                   "GPUWrapperOp must be inside an allocation scope");
            Value res0;
            {
              OpBuilder::InsertionGuard guard(b);
              b.setInsertionPointToStart(allocaBlock);
              res0 = memref::AllocaOp::create(
                  b, rewriteLocation(g.getLoc(), options.strip_llvm_debuginfo),
                  MT0);
            }

            Value storeVal = arg;
            if (isIndex) {
              storeVal = arith::IndexCastOp::create(
                  b, rewriteLocation(g.getLoc(), options.strip_llvm_debuginfo),
                  b.getI64Type(), arg);
            }

            affine::AffineStoreOp::create(
                b, rewriteLocation(g.getLoc(), options.strip_llvm_debuginfo),
                storeVal, res0, b.getMultiDimIdentityMap(0), ValueRange());
            // The memcpy size is in bytes: a count of one only copies the
            // low byte of the scalar.
            int64_t elemBytes =
                (cast<MemRefType>(res0.getType())
                     .getElementType()
                     .getIntOrFloatBitWidth() +
                 7) /
                8;
            auto csz = arith::ConstantIndexOp::create(
                b, rewriteLocation(g.getLoc(), options.strip_llvm_debuginfo),
                elemBytes);
            enzymexla::MemcpyOp::create(
                b, rewriteLocation(g.getLoc(), options.strip_llvm_debuginfo),
                (mlir::Type) nullptr, ValueRange(), res, res0, csz);
            b.setInsertionPointToStart(body);
            auto ld = affine::AffineLoadOp::create(
                b, rewriteLocation(g.getLoc(), options.strip_llvm_debuginfo),
                res, b.getMultiDimIdentityMap(0), ValueRange());
            loads.push_back(ld);
            Value ldVal = ld;
            if (isIndex) {
              ldVal = arith::IndexCastOp::create(
                  b, rewriteLocation(g.getLoc(), options.strip_llvm_debuginfo),
                  b.getIndexType(), ld);

              llvm::SmallSetVector<Operation *, 4> opsToReplace;
              for (OpOperand &use : llvm::make_early_inc_range(arg.getUses())) {
                if (!g->isProperAncestor(use.getOwner())) {
                  continue;
                }
                auto op = use.getOwner();
                if (auto loadOp = dyn_cast<affine::AffineLoadOp>(op)) {
                  opsToReplace.insert(op);
                  continue;
                }
                if (auto storeOp = dyn_cast<affine::AffineStoreOp>(op)) {
                  bool isIndexUse = false;
                  for (auto idx : storeOp.getIndices()) {
                    if (idx == arg) {
                      isIndexUse = true;
                      break;
                    }
                  }
                  if (isIndexUse) {
                    opsToReplace.insert(op);
                    continue;
                  }
                }
                use.set(ldVal);
              }

              for (auto op : opsToReplace) {
                if (auto loadOp = dyn_cast<affine::AffineLoadOp>(op)) {
                  OpBuilder B(loadOp);
                  SmallVector<Value> indices;
                  for (auto idx : loadOp.getIndices()) {
                    if (idx == arg) {
                      indices.push_back(ldVal);
                    } else {
                      indices.push_back(idx);
                    }
                  }
                  auto maybeExpanded = mlir::affine::expandAffineMap(
                      B, loadOp.getLoc(), loadOp.getAffineMap(), indices);
                  assert(maybeExpanded.has_value() &&
                         "failed to expand affine map");
                  auto newLoad = memref::LoadOp::create(
                      B, loadOp.getLoc(), loadOp.getMemref(), *maybeExpanded);
                  loadOp.replaceAllUsesWith(newLoad.getResult());
                  loadOp.erase();
                } else {
                  auto storeOp = cast<affine::AffineStoreOp>(op);

                  OpBuilder B(storeOp);
                  SmallVector<Value> indices;
                  for (auto idx : storeOp.getIndices()) {
                    if (idx == arg) {
                      indices.push_back(ldVal);
                    } else {
                      indices.push_back(idx);
                    }
                  }
                  auto maybeExpanded = mlir::affine::expandAffineMap(
                      B, storeOp.getLoc(), storeOp.getAffineMap(), indices);
                  assert(maybeExpanded.has_value() &&
                         "failed to expand affine map");
                  memref::StoreOp::create(B, storeOp.getLoc(),
                                          storeOp.getValueToStore(),
                                          storeOp.getMemref(), *maybeExpanded);
                  storeOp.erase();
                }
              }
            } else {
              arg.replaceUsesWithIf(ld, [&](OpOperand &opOperand) {
                return g->isProperAncestor(opOperand.getOwner());
              });
            }

            b.setInsertionPointAfter(g);
            gpu::DeallocOp::create(
                b, rewriteLocation(g.getLoc(), options.strip_llvm_debuginfo),
                (mlir::Type) nullptr, ValueRange(), res);
            buffered[arg] = ld;
            operands.insert(res);
            continue;
          }

          if (isa<MemRefType>(arg.getType())) {
            operands.insert(arg);
            continue;
          }

          operands.insert(arg);
        }

        for (auto ld : loads) {
          if (ld != &body->front()) {
            ld->moveBefore(&body->front());
          }
        }
      }

      SmallVector<Type> tensorTypes;
      bool failed = false;
      for (auto arg : operands) {
        auto MT = dyn_cast<MemRefType>(arg.getType());
        if (!MT) {
          failed = true;
          if (err_if_not_fully_raised) {
            llvm::errs() << "failed to raise operand: " << arg << "\n"
                         << " within " << g << "\n";
            signalPassFailure();
          }
          break;
        }
        auto TT = RankedTensorType::get(MT.getShape(), MT.getElementType());
        auto newArg = newBlock->addArgument(
            TT, rewriteLocation(arg.getLoc(), options.strip_llvm_debuginfo));
        mapping.map(arg, newArg);
        tensorTypes.push_back(TT);
      }
      if (failed) {
        delete newBlock;
        continue;
      }

      auto newFuncType =
          FunctionType::get(g->getContext(), tensorTypes, tensorTypes);

      std::string name = "rxla$raised_" + std::to_string(raised_count);
      raised_count++;

      auto newFunc = func::FuncOp::create(
          rewriteLocation(g->getLoc(), options.strip_llvm_debuginfo), name,
          newFuncType);
      newFunc.setVisibility(mlir::SymbolTable::Visibility::Private);
      newFunc.getBody().push_back(newBlock);

      OpBuilder builder(newBlock, newBlock->end());

      bool anyFailed = false;

      ParallelContext emptyPc = ParallelContext::getEmpty(options);
      for (auto &it : body->without_terminator()) {
        anyFailed =
            tryRaisingOpToStableHLO(&it, mapping, builder, maps, emptyPc)
                .failed();
        if (anyFailed) {
          if (err_if_not_fully_raised) {
            llvm::errs() << "failed to raise operation: " << *&it << "\n"
                         << " within " << g << "\n";
            signalPassFailure();
          }
          break;
        }
      }

      if (anyFailed) {
        newFunc->erase();
        continue;
      }

      SmallVector<Value> results;
      for (auto arg : operands) {
        auto val = mapping.lookup(arg);
        results.push_back(val);
      }

      func::ReturnOp::create(
          builder, rewriteLocation(g->getLoc(), options.strip_llvm_debuginfo),
          results);
      modOp.getBody()->push_back(newFunc);
      SymbolTable::setSymbolVisibility(newFunc,
                                       SymbolTable::Visibility::Private);

      {
        OpBuilder builder(g);
        enzymexla::XLAWrapperOp::create(
            builder, g->getLoc(), SymbolRefAttr::get(newFunc),
            llvm::to_vector(operands), nullptr, nullptr);
        if (g->getNumResults() > 0) {
          Value zero = arith::ConstantIndexOp::create(builder, g->getLoc(), 0);
          g->getResult(0).replaceAllUsesWith(zero);
        }
        g->erase();
        anyRaised = true;
      }
    }

    if (!anyRaised) {
      markAllAnalysesPreserved();
    }
  }
};
} // namespace
