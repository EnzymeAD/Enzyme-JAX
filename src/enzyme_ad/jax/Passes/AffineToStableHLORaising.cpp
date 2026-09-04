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

#include "Enzyme/MLIR/Dialect/Ops.h"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

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
#include <deque>
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
    int64_t unrollBudget = 1 << 16;
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
    if (affine::isAffineForInductionVar(iv) && !pc.isParallelIV(iv)) {
      shape.push_back(1);
      continue;
    }

    auto range = getIVRange(iv);
    if (!range.has_value())
      return {};

    shape.push_back(range->getNumIters());
  }

  return shape;
}

static FailureOr<affine::AffineValueMap>
alignMemoryAccess(Value &a, affine::AffineValueMap src, Value *bs,
                  ArrayRef<affine::AffineValueMap> dsts, OpBuilder &builder,
                  ParallelContext pc) {
  // NOTE a default-constructed AffineValueMap holds a null context, so its
  // getAffineMap() cannot even be probed; inputs must be maps the caller
  // actually recorded.
  if (!a)
    return failure();
  for (unsigned qi = 0; qi < dsts.size(); ++qi)
    if (!bs[qi])
      return failure();
  // -> tensor<10x1xf32> loaded from (i) -> (i, 0)
  // -> to tensor<1x10xf32> written as (i) -> (0, i)

  // affineMapShape bails to an empty vector on accesses it cannot size
  // (an IV with no static range); that must reject the alignment, not read
  // past the end below.
  SmallVector<int64_t> shapeA = affineMapShape(src, pc);
  if (shapeA.size() != cast<RankedTensorType>(a.getType()).getShape().size())
    return failure();
  SmallVector<SmallVector<int64_t>> shapeBs;
  for (size_t i = 0; i < dsts.size(); i++) {
    shapeBs.push_back(affineMapShape(dsts[i], pc));
    if (shapeBs[i].size() !=
        cast<RankedTensorType>(bs[i].getType()).getShape().size())
      return failure();
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

    exprs.push_back(
        mlir::getAffineDimExpr(mapOperands.size(), ivA.getContext()));
    mapOperands.push_back(ivA);
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

      exprs.push_back(
          mlir::getAffineDimExpr(mapOperands.size(), ivB.getContext()));
      mapOperands.push_back(ivB);
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

  affine::AffineValueMap outputMap(
      AffineMap::getMultiDimIdentityMap(mapOperands.size(), a.getContext()),
      mapOperands);

  return outputMap;
}

static FailureOr<affine::AffineValueMap>
alignMemoryAccess(Value &a, affine::AffineValueMap src, Value &b,
                  affine::AffineValueMap dst, OpBuilder &builder,
                  ParallelContext pc) {
  Value bs[] = {b};
  affine::AffineValueMap dsts[] = {dst};
  auto res = alignMemoryAccess(a, src, bs, dsts, builder, pc);
  b = bs[0];
  return res;
}

// like affine::expandAffineExpr but with stablehlo ops and returning
// the corresponding AffineValueMap for the produced value.
static FailureOr<std::tuple<Value, affine::AffineValueMap>>
expandAffineExpr(OpBuilder &builder, Location loc, AffineExpr expr,
                 ValueRange operands, IRMapping &mapping, unsigned numDims,
                 ParallelContext pc) {
  using Expanded = std::tuple<Value, affine::AffineValueMap>;
  if (auto constExpr = dyn_cast<AffineConstantExpr>(expr)) {
    auto ET = builder.getI64Type();
    auto TT = RankedTensorType::get({}, ET);
    Value res = stablehlo::ConstantOp::create(
        builder, loc, TT,
        SplatElementsAttr::get(TT, ArrayRef<Attribute>(IntegerAttr::get(
                                       ET, constExpr.getValue()))));
    return Expanded{
        res, affine::AffineValueMap(AffineMap::get(expr.getContext()), {})};
  }

  if (auto binExpr = dyn_cast<AffineBinaryOpExpr>(expr)) {
    AffineExpr lhsExpr = binExpr.getLHS(), rhsExpr = binExpr.getRHS();
    auto lhsExpanded =
        expandAffineExpr(builder, loc, lhsExpr, operands, mapping, numDims, pc);
    if (failed(lhsExpanded))
      return failure();
    auto rhsExpanded =
        expandAffineExpr(builder, loc, rhsExpr, operands, mapping, numDims, pc);
    if (failed(rhsExpanded))
      return failure();
    auto [lhs, lhsMap] = *lhsExpanded;
    auto [rhs, rhsMap] = *rhsExpanded;

    auto aligned = alignMemoryAccess(lhs, lhsMap, rhs, rhsMap, builder, pc);
    if (failed(aligned))
      return failure();
    affine::AffineValueMap outputMap = *aligned;

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
    return Expanded{result, outputMap};
  }

  if (auto symExpr = dyn_cast<AffineSymbolExpr>(expr)) {
    Value sym = operands[symExpr.getPosition() + numDims];
    Value mapped = mapping.lookupOrNull(sym);
    if (!mapped)
      return failure();
    return Expanded{
        mapped, affine::AffineValueMap(AffineMap::get(sym.getContext()), {})};
  }

  if (auto dimExpr = dyn_cast<AffineDimExpr>(expr)) {
    Value dim = operands[dimExpr.getPosition()];
    Value mapped = mapping.lookupOrNull(dim);
    if (!mapped)
      return failure();

    if (!pc.isParallelIV(dim)) {
      return Expanded{
          mapped, affine::AffineValueMap(AffineMap::get(dim.getContext()), {})};
    }

    return Expanded{
        mapped,
        affine::AffineValueMap(
            AffineMap::getMultiDimIdentityMap(1, expr.getContext()), {dim})};
  }

  llvm_unreachable("unreachable");
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

// Blocks guaranteed to reach llvm.unreachable (abort branches), mirroring
// Enzyme's getGuaranteedUnreachable in FunctionUtils.h.
static DenseSet<Block *> getGuaranteedUnreachable(Region &r) {
  DenseSet<Block *> knownUnreachable;
  std::deque<Block *> todo;
  for (Block &b : r)
    todo.push_back(&b);

  while (!todo.empty()) {
    Block *next = todo.front();
    todo.pop_front();

    if (knownUnreachable.contains(next))
      continue;

    bool unreachable = isa<LLVM::UnreachableOp>(next->getTerminator());
    if (!unreachable) {
      auto succs = next->getSuccessors();
      unreachable = !succs.empty();
      for (Block *succ : succs) {
        if (!knownUnreachable.contains(succ)) {
          unreachable = false;
          break;
        }
      }
    }
    if (!unreachable)
      continue;

    knownUnreachable.insert(next);
    for (Block *pred : next->getPredecessors())
      todo.push_back(pred);
  }

  return knownUnreachable;
}

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
      auto newMaskMap =
          alignMemoryAccess(mask, maskMap, cond, condMap, builder, pc);
      if (failed(newMaskMap))
        return Value();

      mask = stablehlo::AndOp::create(
          builder,
          rewriteLocation(ifOp->getLoc(), pc.options.strip_llvm_debuginfo),
          mask, cond);
      maps[mask] = *newMaskMap;
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
            !condTy || condTy.getRank() != 0 || !readOnly)
          return ifOp->emitError(
              "cannot raise a branch choosing between buffers");
        auto sel = stablehlo::SelectOp::create(
            builder,
            rewriteLocation(ifOp->getLoc(), pc.options.strip_llvm_debuginfo),
            cond, thenBuf, elseBuf);
        mapping.map(res, sel.getResult());
        if (auto it = maps.find(thenBuf); it != maps.end())
          maps[sel.getResult()] = it->second;
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
      auto outputMap = alignMemoryAccess(a, mapA, dsts, submaps, builder, pc);
      if (failed(outputMap))
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
      maps[newOp.getResult()] = *outputMap;
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
                   const ParallelContext &pc, bool accumulate = false) {
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

  // The stored value must not vary along a dimension that is absent from the
  // store indices.
  if (llvm::any_of(broadcastDims, [](int64_t dim) { return dim == -1; })) {
    return nullptr;
  }

  // Align update to the store indices grid; the grid axes act as implicit
  // batch dimensions of the scatter.
  update = stablehlo::BroadcastInDimOp::create(
      builder, loc, UTy.clone(gridShape), update, broadcastDims);

  if (pc.mask) {
    SmallVector<int64_t> collapsedDims(scatterDimsToOperandDims.begin(),
                                       scatterDimsToOperandDims.end());
    SmallVector<int64_t> sliceSizes(collapsedDims.size(), 1);
    Value orig = stablehlo::GatherOp::create(
        builder, loc, input, indices,
        stablehlo::GatherDimensionNumbersAttr::get(
            loc.getContext(),
            /*offsetDims*/ {},
            /*collapsedSliceDims*/ collapsedDims,
            /*operandBatchingDims*/ {},
            /*startIndicesBatchingDims*/ {},
            /*startIndexMap*/ collapsedDims,
            /*indexVectorDim*/ (int64_t)gridShape.size()),
        sliceSizes);

    // Broadcast the mask from its IV-space to the update's grid shape.
    Value mask = pc.mask;
    affine::AffineValueMap maskMap = maps.lookup(mask);
    SmallVector<int64_t> maskBroadcastDims;
    for (auto E : maskMap.getAffineMap().getResults()) {
      Value maskIV = getIVForExpr(maskMap, E);
      for (auto [k, iv] : llvm::enumerate(ivs)) {
        if (iv == maskIV) {
          maskBroadcastDims.push_back((int64_t)k);
          break;
        }
      }
    }
    auto gridTy = cast<RankedTensorType>(update.getType());
    auto maskGridTy =
        RankedTensorType::get(gridTy.getShape(), builder.getI1Type());
    Value broadcastedMask = stablehlo::BroadcastInDimOp::create(
        builder, loc, maskGridTy, mask, maskBroadcastDims);

    update = stablehlo::SelectOp::create(builder, loc, broadcastedMask, update,
                                         orig);
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
      // With a mask, masked-out positions scatter orig back at potentially
      // repeated indices — uniqueness can only be claimed without a mask; an
      // accumulating scatter's colliding indices are the whole point.
      /*uniqueIndices*/ !pc.mask && !accumulate);
  Value res = scatter.getResult(0);

  Block *updateBody = new Block();
  scatter.getUpdateComputation().push_back(updateBody);

  auto unrankedTy = RankedTensorType::get(
      {}, cast<RankedTensorType>(update.getType()).getElementType());
  Value currentInBody = updateBody->addArgument(unrankedTy, loc);
  Value updateInBody = updateBody->addArgument(unrankedTy, loc);

  {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(updateBody);
    Value out = updateInBody;
    if (accumulate) {
      // An atomic add raises as a combining scatter: adds commute (up to
      // rounding, exactly like the atomic), so application order is
      // irrelevant.
      out = stablehlo::AddOp::create(builder, loc, currentInBody, updateInBody);
    }
    stablehlo::ReturnOp::create(builder, loc, out);
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
        auto expanded = expandAffineExpr(builder, wloc, expr, operands, mapping,
                                         map.getNumDims(), pc);
        if (failed(expanded))
          return nullptr;
        auto [val, avm] = *expanded;
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

      auto aligned = alignMemoryAccess(reduce_broadcasted, reduce_map, dsts,
                                       submaps, builder, *newPc);
      if (failed(aligned))
        return failure();
      affine::AffineValueMap outputMap = *aligned;

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
      auto aligned = alignMemoryAccess(val, outputMap, dsts.data(), submaps,
                                       builder, *newPc);
      if (failed(aligned))
        return failure();
      affine::AffineValueMap outputMap2 = *aligned;

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

// The op count of `op` once every constant-bound affine.for inside it (and
// `op` itself, if it is one) is unrolled, saturating at 2^40.
static int64_t unrollCost(Operation *op) {
  int64_t body = 1;
  for (Region &r : op->getRegions())
    for (Block &b : r)
      for (Operation &inner : b)
        body = std::min(body + unrollCost(&inner), (int64_t)1 << 40);
  auto forOp = dyn_cast<affine::AffineForOp>(op);
  if (!forOp || !forOp.hasConstantBounds())
    return body;
  int64_t step = forOp.getStepAsInt();
  int64_t trip = (forOp.getConstantUpperBound() -
                  forOp.getConstantLowerBound() + step - 1) /
                 step;
  return std::min(std::max(trip, (int64_t)1) * body, (int64_t)1 << 40);
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
        needsGeneralScatterGather(accessValueMap);

    if (emitAsGather) {
      SmallVector<Value> lIndices;
      for (auto E : accessValueMap.getAffineMap().getResults()) {
        auto expanded = expandAffineExpr(
            builder,
            rewriteLocation(op->getLoc(), pc.options.strip_llvm_debuginfo), E,
            accessValueMap.getOperands(), mapping,
            accessValueMap.getAffineMap().getNumDims(), pc);
        if (failed(expanded))
          return failure();
        auto [idx, idxMap] = *expanded;
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

        auto expanded = expandAffineExpr(
            builder,
            rewriteLocation(op->getLoc(), pc.options.strip_llvm_debuginfo),
            exprToEmit, accessValueMap.getOperands(), mapping,
            accessValueMap.getAffineMap().getNumDims(), pc);
        if (failed(expanded))
          return failure();
        Value startIndex = std::get<0>(*expanded);

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
        needsGeneralScatterGather(accessValueMap);

    if (emitAsScatter) {
      SmallVector<Value> sIndices;
      for (auto E : accessValueMap.getAffineMap().getResults()) {
        auto expanded = expandAffineExpr(
            builder,
            rewriteLocation(op->getLoc(), pc.options.strip_llvm_debuginfo), E,
            accessValueMap.getOperands(), mapping,
            accessValueMap.getAffineMap().getNumDims(), pc);
        if (failed(expanded))
          return failure();
        auto [expandedIndex, indexMap] = *expanded;
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

        auto expanded = expandAffineExpr(
            builder,
            rewriteLocation(iv.getLoc(), pc.options.strip_llvm_debuginfo),
            exprToEmit, accessValueMap.getOperands(), mapping,
            accessValueMap.getAffineMap().getNumDims(), pc);
        if (failed(expanded))
          return failure();
        startIndex = std::get<0>(*expanded);
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
      auto aligned = alignMemoryAccess(mask, maskMap, vals, dsts, builder, pc);
      if (failed(aligned))
        return op->emitError("cannot align masked store");
      storeValueMap = *aligned;

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

  if (auto rmw = dyn_cast<memref::AtomicRMWOp>(op)) {
    // Only accumulation raises (as a combining scatter), and only when the
    // old value is unobserved.
    if ((rmw.getKind() != arith::AtomicRMWKind::addf &&
         rmw.getKind() != arith::AtomicRMWKind::addi) ||
        !rmw.getResult().use_empty())
      return failure();
    Value value = rmw.getValue();
    Value memref = rmw.getMemref();
    SmallVector<Value> sIndices;
    for (auto idx : rmw.getIndices()) {
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
        pc, /*accumulate=*/true);
    if (!res)
      return op->emitError(
                 "atomic add is dependent on less dims than stored value: ")
             << *op;
    mapping.map(memref, res);
    return success();
  }

  if (auto rmw = dyn_cast<enzyme::AtomicRMWOp>(op)) {
    // As for the memref one: only accumulation raises, as a combining
    // scatter, and only when the old value is unobserved. The ordering the op
    // carries says nothing once the accumulation is a scatter over the whole
    // iteration space.
    if ((rmw.getKind() != arith::AtomicRMWKind::addf &&
         rmw.getKind() != arith::AtomicRMWKind::addi) ||
        !rmw.getResult().use_empty())
      return failure();
    Value value = rmw.getValue();
    Value memref = rmw.getMemref();
    SmallVector<Value> sIndices;
    for (auto idx : rmw.getIndices()) {
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
        pc, /*accumulate=*/true);
    if (!res)
      return op->emitError(
                 "atomic add is dependent on less dims than stored value: ")
             << *op;
    mapping.map(memref, res);
    return success();
  }

  if (auto rmw = dyn_cast<enzyme::AffineAtomicRMWOp>(op)) {
    // The affine form of the same accumulation, with the same two
    // conditions -- it accumulates, and its old value is unobserved. Its
    // address is a map over the operands rather than plain indices.
    if ((rmw.getKind() != arith::AtomicRMWKind::addf &&
         rmw.getKind() != arith::AtomicRMWKind::addi) ||
        !rmw.getResult().use_empty())
      return failure();
    Value value = rmw.getValue();
    Value memref = rmw.getMemref();
    if (!mapping.lookupOrNull(value) || !mapping.lookupOrNull(memref))
      return failure();

    affine::AffineValueMap accessValueMap(rmw.getMap(), rmw.getIndices());
    accessValueMap.composeSimplifyAndCanonicalize();

    // A result that names one operand is that operand, taken with the map it
    // was raised under, exactly as the memref form takes its indices.
    // Anything else is expanded the way an affine.store's scatter indices are.
    AffineMap map = accessValueMap.getAffineMap();
    SmallVector<Value> sIndices;
    for (auto E : map.getResults()) {
      unsigned pos = 0;
      bool namesOperand = true;
      if (auto dim = dyn_cast<AffineDimExpr>(E))
        pos = dim.getPosition();
      else if (auto sym = dyn_cast<AffineSymbolExpr>(E))
        pos = map.getNumDims() + sym.getPosition();
      else
        namesOperand = false;

      if (namesOperand) {
        Value mapped = mapping.lookupOrNull(accessValueMap.getOperand(pos));
        if (!mapped || !maps.count(mapped))
          return failure();
        sIndices.push_back(mapped);
        continue;
      }

      auto expanded = expandAffineExpr(
          builder,
          rewriteLocation(op->getLoc(), pc.options.strip_llvm_debuginfo), E,
          accessValueMap.getOperands(), mapping, map.getNumDims(), pc);
      if (failed(expanded))
        return failure();
      auto [expandedIndex, indexMap] = *expanded;
      maps[expandedIndex] = indexMap;
      sIndices.push_back(expandedIndex);
    }

    Value res = emitStoreAsScatter(
        rewriteLocation(op->getLoc(), pc.options.strip_llvm_debuginfo),
        mapping.lookup(value), mapping.lookup(memref), sIndices, builder, maps,
        pc, /*accumulate=*/true);
    if (!res)
      return op->emitError(
                 "atomic add is dependent on less dims than stored value: ")
             << *op;
    mapping.map(memref, res);
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
    auto res = expandAffineExpr(
        builder,
        rewriteLocation(apply.getLoc(), pc.options.strip_llvm_debuginfo),
        avm.getAffineMap().getResult(0), avm.getOperands(), mapping,
        avm.getAffineMap().getNumDims(), pc);
    if (failed(res))
      return failure();
    auto [expanded, expandedMap] = *res;
    mapping.map(apply.getResult(), expanded);
    maps[expanded] = expandedMap;
    return success();
  }

  if (auto p2m = dyn_cast<enzymexla::Pointer2MemrefOp>(op)) {
    Value operand = op->getOperand(0), result = op->getResult(0);
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
    auto outputMap =
        alignMemoryAccess(a, itA->second, b, itB->second, builder, pc);
    if (failed(outputMap))
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
      maps[newRes] = *outputMap;
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
    auto outputMap = alignMemoryAccess(a, mapA, dsts, submaps, builder, pc);
    if (failed(outputMap))
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
      maps[newRes] = *outputMap;
    }

    return success();
  }

  // An alloca scope only delimits stack lifetime, meaningless under value
  // semantics: raise its single-block body and forward the yields.
  if (auto scope = dyn_cast<memref::AllocaScopeOp>(op)) {
    Region &r = scope.getBodyRegion();
    if (!r.hasOneBlock())
      return failure();
    Block *body = &r.front();
    for (auto &innerOp : body->without_terminator()) {
      if (tryRaisingOpToStableHLO(&innerOp, mapping, builder, maps, pc)
              .failed())
        return failure();
    }
    Operation *term = body->getTerminator();
    for (auto [res, yielded] :
         llvm::zip_equal(scope->getResults(), term->getOperands()))
      mapping.map(res, mapping.lookup(yielded));
    return success();
  }

  // An execute region wraps the inliner's cloned callee CFG. It raises when
  // there is a unique live path from the entry to the yield, where a block
  // is dead if every path out of it reaches llvm.unreachable (abort
  // branches); a real diamond or a revisited block fails.
  if (auto exec = dyn_cast<scf::ExecuteRegionOp>(op)) {
    Region &r = exec.getRegion();
    DenseSet<Block *> trapping = getGuaranteedUnreachable(r);
    Block *cur = &r.front();
    DenseSet<Block *> visited;
    while (true) {
      if (!visited.insert(cur).second)
        return failure();
      for (auto &innerOp : cur->without_terminator()) {
        if (tryRaisingOpToStableHLO(&innerOp, mapping, builder, maps, pc)
                .failed())
          return failure();
      }
      Operation *term = cur->getTerminator();
      if (isa<scf::YieldOp>(term)) {
        for (auto [res, yielded] :
             llvm::zip_equal(exec->getResults(), term->getOperands()))
          mapping.map(res, mapping.lookup(yielded));
        return success();
      }
      auto br = dyn_cast<BranchOpInterface>(term);
      if (!br)
        return failure();
      Block *next = nullptr;
      int64_t liveIdx = -1;
      for (auto [i, succ] : llvm::enumerate(term->getSuccessors())) {
        if (trapping.contains(succ))
          continue;
        if (next)
          return failure();
        next = succ;
        liveIdx = i;
      }
      if (!next)
        return failure();
      auto sops = br.getSuccessorOperands(liveIdx);
      for (auto [ba, v] :
           llvm::zip(next->getArguments(), sops.getForwardedOperands()))
        mapping.map(ba, mapping.lookup(v));
      cur = next;
    }
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
      auto expanded = expandAffineExpr(
          builder,
          rewriteLocation(ifOp.getLoc(), pc.options.strip_llvm_debuginfo),
          constraint, constraintMap.getOperands(), mapping,
          constraintMap.getNumDims(), pc);
      if (failed(expanded))
        return failure();
      auto [expandedExpr, outputMap] = *expanded;
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
        auto aligned =
            alignMemoryAccess(cond, map, newCond, outputMap, builder, pc);
        if (failed(aligned))
          return failure();
        map = *aligned;
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
    bool hugeUnroll =
        pc.options.unrollBudget >= 0 && forOp.hasConstantBounds() &&
        unrollCost(forOp.getOperation()) > pc.options.unrollBudget;
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
    if (!MT.hasStaticShape() || !isXLACompatiblePrimitive(MT.getElementType()))
      return op->emitError("cannot raise dynamic or non-primitive alloca")
             << *op;
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

  // Raised execution is ordered over whole tensors: a store over a batched
  // thread axis completes for the entire axis before the next op runs, which
  // is exactly what the barrier guaranteed.
  if (isa<enzymexla::BarrierOp>(op)) {
    // Raised execution is ordered over whole tensors, so a barrier over
    // batched axes is a no-op. That is only sound for the axes the barrier
    // spans when they are batched: an induction variable of a dynamically
    // sized loop raises serialized, with no whole-tensor ordering to lean
    // on, and dropping the barrier would let one lane run ahead of the
    // others' stores.
    for (Value iv : op->getOperands()) {
      auto ba = dyn_cast<BlockArgument>(iv);
      if (!ba)
        continue;
      Operation *owner = ba.getOwner()->getParentOp();
      if (auto par = dyn_cast<affine::AffineParallelOp>(owner)) {
        if (!par.getConstantRanges())
          return op->emitError(
              "barrier over a dynamically sized parallel axis");
        continue;
      }
      if (isa<affine::AffineForOp, scf::ForOp, scf::WhileOp>(owner))
        return op->emitError("barrier over a dynamically sized parallel axis");
    }
    return success();
  }

  // An optimizer hint carries no semantics a tensor program needs.
  if (isa<LLVM::AssumeOp>(op))
    return success();

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
  static std::optional<int64_t> derivedExtentBound(Value v,
                                                   unsigned depth = 0) {
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
      auto l = derivedExtentBound(mn.getLhs(), depth + 1);
      auto r = derivedExtentBound(mn.getRhs(), depth + 1);
      if (l && r)
        return std::min(*l, *r);
      return l ? l : r;
    }
    if (auto mn = v.getDefiningOp<arith::MinUIOp>()) {
      auto l = derivedExtentBound(mn.getLhs(), depth + 1);
      auto r = derivedExtentBound(mn.getRhs(), depth + 1);
      if (l && r)
        return std::min(*l, *r);
      return l ? l : r;
    }
    return std::nullopt;
  }

  // A parallel axis whose extent is dynamic but provably bounded (a block
  // size clamped by a min against a constant) batches at the bound instead
  // of peeling to a serial loop: the axis becomes constant-extent and the
  // body sits behind an `iv < extent` guard, which the masking machinery
  // already understands. Barriers over the axis then stay batched no-ops.
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
      for (unsigned i = 0; i < n; ++i) {
        auto lb = getConstant(par.getLowerBoundMap(i));
        if (!lb || *lb != 0 || par.getSteps()[i] != 1)
          continue;
        if (getConstant(par.getUpperBoundMap(i)))
          continue;
        auto um = par.getUpperBoundMap(i);
        if (um.getNumResults() != 1)
          continue;
        auto se = dyn_cast<AffineSymbolExpr>(um.getResult(0));
        if (!se)
          continue;
        Value ext =
            par.getUpperBoundsOperands()[par.getUpperBoundsMap().getNumDims() +
                                         se.getPosition()];
        if (auto c = derivedExtentBound(ext))
          bounded.push_back({i, *c, ext});
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
          ubounds.push_back(
              getAffineConstantExpr(bit->bound, par.getContext()));
          uboundGroup.push_back(1);
        } else {
          ubounds.append(um.getResults().begin(), um.getResults().end());
          uboundGroup.push_back(um.getNumResults());
        }
        steps.push_back(par.getSteps()[i]);
      }
      // When every bound came out constant, drop the stale symbols and
      // operands entirely: downstream batching expects clean constant maps.
      bool allConstant = llvm::all_of(lbounds,
                                      [](AffineExpr e) {
                                        return isa<AffineConstantExpr>(e);
                                      }) &&
                         llvm::all_of(ubounds, [](AffineExpr e) {
                           return isa<AffineConstantExpr>(e);
                         });
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
      auto ifOp =
          affine::AffineIfOp::create(b, loc, TypeRange(), iset, setOperands,
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

  void runOnOperation() override {
    ParallelContext::Options options{enable_lockstep_for, dump_failed_lockstep,
                                     prefer_while_raising, strip_llvm_debuginfo,
                                     unroll_budget};
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
      stripAccessMemorySpaceCasts(func);
      boundParallelAxes(func);
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
      stripAccessMemorySpaceCasts(g);
      boundParallelAxes(g);
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
            if (arg.getDefiningOp<ub::PoisonOp>()) {
              newVal = cast<mlir::enzyme::AutoDiffTypeInterface>(arg.getType())
                           .createNullValue(
                               builder,
                               rewriteLocation(arg.getLoc(),
                                               options.strip_llvm_debuginfo));
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
            int64_t elemBytes = (cast<MemRefType>(res0.getType())
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
