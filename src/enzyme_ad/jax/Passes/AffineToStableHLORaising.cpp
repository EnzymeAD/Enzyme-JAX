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
#include "src/enzyme_ad/jax/Passes/Passes.h"

#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Affine/Passes.h"
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
#include <optional>

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_AFFINETOSTABLEHLORAISING
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

#define DEBUG_TYPE "raise-affine-to-stablehlo"

using namespace mlir;
using namespace mlir::enzyme;

Type makeIndexToI64(Type ty) {
  if (isa<IndexType>(ty))
    return IntegerType::get(ty.getContext(), 64);

  if (auto tenTy = dyn_cast<RankedTensorType>(ty))
    return RankedTensorType::get(tenTy.getShape(),
                                 makeIndexToI64(tenTy.getElementType()));

  return ty;
}

// This represents the values taken from an induction variable with the
// following syntax: [lb:ub:step]. ub is non-inclusive.
struct InductionVariableRange {
  int64_t lb;
  int64_t ub;
  int64_t step;

  int64_t getNumIters() { return (ub - lb) / step; }
};

// Assumes a single IV per Expr. (i) -> (i * 3 + 2)
static Value getIVForExpr(affine::AffineValueMap map, AffineExpr expr) {
  assert(!expr.isSymbolicOrConstant());
  unsigned pos;

  expr.walk([&pos](AffineExpr expr) {
    if (auto dimExpr = dyn_cast<AffineDimExpr>(expr)) {
      pos = dimExpr.getPosition();
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });

  return map.getOperand(pos);
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
                  llvm::DenseMap<Value, affine::AffineValueMap> &maps) {
  auto ET = builder.getI64Type();
  auto Ty = RankedTensorType::get({range.getNumIters()}, ET);
  Value iota =
      builder.create<stablehlo::IotaOp>(iv.getLoc(), Ty, 0).getResult();
  iota = builder.create<stablehlo::AddOp>(
      iv.getLoc(), Ty, iota,
      builder.create<stablehlo::ConstantOp>(
          iv.getLoc(), Ty,
          SplatElementsAttr::get(
              Ty, ArrayRef<Attribute>(IntegerAttr::get(ET, range.lb)))));
  iota = builder.create<stablehlo::MulOp>(
      iv.getLoc(), Ty, iota,
      builder.create<stablehlo::ConstantOp>(
          iv.getLoc(), Ty,
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
  } options;

  explicit ParallelContext(Options &options) : options(options) {}

  SmallVector<InductionVariableRange, 8> ranges;
  SmallVector<Value, 8> ivs;

  bool isParallelIV(Value iv) { return llvm::is_contained(ivs, iv); }

  RankedTensorType getTensorType(Type elTy) {
    SmallVector<int64_t> shape = llvm::map_to_vector(
        ranges, [&](auto range) { return range.getNumIters(); });
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
    assert(CTT.getElementType() == TT.getElementType());
    if (CTT.getShape() == TT.getShape())
      return Broadcast{v, avm};
    if (llvm::any_of(llvm::zip(CTT.getShape(), TT.getShape()),
                     [](auto p) { return std::get<0>(p) != std::get<1>(p); }))
      return std::nullopt;
    if (CTT.getRank() > TT.getRank())
      return std::nullopt;

    // TODO I haven't thought through how to broadcast non-scalars to the
    // shape we need.
    if (CTT.getRank() != 0)
      return std::nullopt;
    SmallVector<int64_t> dimsToBroadcast;
    auto bc = b.create<stablehlo::BroadcastInDimOp>(v.getLoc(), TT, v,
                                                    dimsToBroadcast);

    AffineMap newMap = AffineMap::getMultiDimIdentityMap(
        TT.getRank() - CTT.getRank(), b.getContext());

    return Broadcast{bc, affine::AffineValueMap(newMap, ivs)};
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
    ParallelContext pc(options);
    return pc;
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

static affine::AffineValueMap
alignMemoryAccess(Value &a, affine::AffineValueMap src, Value *bs,
                  ArrayRef<affine::AffineValueMap> dsts, OpBuilder &builder,
                  ParallelContext pc) {
  // -> tensor<10x1xf32> loaded from (i) -> (i, 0)
  // -> to tensor<1x10xf32> written as (i) -> (0, i)

  SmallVector<int64_t> shapeA = affineMapShape(src, pc);
  assert(shapeA.size() ==
         cast<RankedTensorType>(a.getType()).getShape().size());
  SmallVector<SmallVector<int64_t>> shapeBs;
  for (int i = 0; i < dsts.size(); i++) {
    shapeBs.push_back(affineMapShape(dsts[i], pc));
    assert(shapeBs[i].size() ==
           cast<RankedTensorType>(bs[i].getType()).getShape().size());
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
    a = builder
            .create<stablehlo::BroadcastInDimOp>(
                a.getLoc(), TA.clone(outputShape), a, broadcastDimensionsA)
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
      bs[i] = builder
                  .create<stablehlo::BroadcastInDimOp>(
                      bs[i].getLoc(), TB.clone(outputShape), bs[i],
                      broadcastDimensionsBs[i])
                  .getResult();
  }

  affine::AffineValueMap outputMap(
      AffineMap::getMultiDimIdentityMap(mapOperands.size(), a.getContext()),
      mapOperands);

  return outputMap;
}

static affine::AffineValueMap
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
static std::tuple<Value, affine::AffineValueMap>
expandAffineExpr(OpBuilder &builder, Location loc, AffineExpr expr,
                 ValueRange operands, IRMapping &mapping, unsigned numDims,
                 ParallelContext pc) {
  if (auto constExpr = dyn_cast<AffineConstantExpr>(expr)) {
    auto ET = builder.getI64Type();
    auto TT = RankedTensorType::get({}, ET);
    Value res = builder.create<stablehlo::ConstantOp>(
        loc, TT,
        SplatElementsAttr::get(TT, ArrayRef<Attribute>(IntegerAttr::get(
                                       ET, constExpr.getValue()))));
    return {res, affine::AffineValueMap(AffineMap::get(expr.getContext()), {})};
  }

  if (auto binExpr = dyn_cast<AffineBinaryOpExpr>(expr)) {
    AffineExpr lhsExpr = binExpr.getLHS(), rhsExpr = binExpr.getRHS();
    auto [lhs, lhsMap] =
        expandAffineExpr(builder, loc, lhsExpr, operands, mapping, numDims, pc);
    auto [rhs, rhsMap] =
        expandAffineExpr(builder, loc, rhsExpr, operands, mapping, numDims, pc);

    affine::AffineValueMap outputMap =
        alignMemoryAccess(lhs, lhsMap, rhs, rhsMap, builder, pc);

    auto makeI64Constant = [loc, &builder](ShapedType ty,
                                           int64_t cst) -> Value {
      return builder
          .create<stablehlo::ConstantOp>(
              loc, ty,

              SplatElementsAttr::get(ty, ArrayRef<Attribute>(IntegerAttr::get(
                                             ty.getElementType(), cst))))
          .getResult();
    };

    Value result;
    switch (expr.getKind()) {
    case AffineExprKind::Add:
      result = builder.create<stablehlo::AddOp>(loc, lhs, rhs);
      break;
    case AffineExprKind::Mul:
      result = builder.create<stablehlo::MulOp>(loc, lhs, rhs);
      break;
    case AffineExprKind::Mod:
      // a mod b =
      //     let remainder = srem a, b;
      //         negative = a < 0 in
      //     select negative, remainder + b, remainder.
      {
        Value remainder = builder.create<stablehlo::RemOp>(loc, lhs, rhs);
        Value negative = builder.create<stablehlo::CompareOp>(
            loc, lhs, makeI64Constant(cast<ShapedType>(lhs.getType()), 0),
            stablehlo::ComparisonDirection::LT);
        result = builder.create<stablehlo::SelectOp>(
            loc, negative,
            builder.create<stablehlo::AddOp>(loc, remainder, rhs), remainder);
      };
      break;
    case AffineExprKind::FloorDiv:
      // a floordiv b =
      //     let negative = a < 0 in
      //     let absolute = negative ? -a - 1 : a in
      //     let quotient = absolute / b in
      //         negative ? -quotient - 1 : quotient
      {
        Value negative = builder.create<stablehlo::CompareOp>(
            loc, lhs, makeI64Constant(cast<ShapedType>(lhs.getType()), 0),
            stablehlo::ComparisonDirection::LE);
        Value one = makeI64Constant(cast<ShapedType>(lhs.getType()), 1);
        Value absolute = builder.create<stablehlo::SelectOp>(
            loc, negative,
            builder.create<stablehlo::SubtractOp>(
                loc, builder.create<stablehlo::NegOp>(loc, lhs), one),
            lhs);
        Value quotient = builder.create<stablehlo::DivOp>(loc, absolute, rhs);
        result = builder.create<stablehlo::SelectOp>(
            loc, negative,
            builder.create<stablehlo::SubtractOp>(
                loc, builder.create<stablehlo::NegOp>(loc, quotient), one),
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
        Value negative = builder.create<stablehlo::CompareOp>(
            loc, lhs, makeI64Constant(cast<ShapedType>(lhs.getType()), 0),
            stablehlo::ComparisonDirection::LE);
        Value one = makeI64Constant(cast<ShapedType>(lhs.getType()), 1);
        Value absolute = builder.create<stablehlo::SelectOp>(
            loc, negative, builder.create<stablehlo::NegOp>(loc, lhs),
            builder.create<stablehlo::AddOp>(loc, lhs, one));
        Value quotient = builder.create<stablehlo::DivOp>(loc, absolute, rhs);
        result = builder.create<stablehlo::SelectOp>(
            loc, negative, builder.create<stablehlo::NegOp>(loc, quotient),
            builder.create<stablehlo::AddOp>(loc, quotient, one));
      };
      break;
    default:
      llvm_unreachable("unsupported expansion of expr");
    }
    return {result, outputMap};
  }

  if (auto symExpr = dyn_cast<AffineSymbolExpr>(expr)) {
    Value sym = operands[symExpr.getPosition() + numDims];
    return {mapping.lookup(sym),
            affine::AffineValueMap(AffineMap::get(sym.getContext()), {})};
  }

  if (auto dimExpr = dyn_cast<AffineDimExpr>(expr)) {
    Value dim = operands[dimExpr.getPosition()];

    if (affine::isAffineForInductionVar(dim)) {
      return {mapping.lookup(dim),
              affine::AffineValueMap(AffineMap::get(dim.getContext()), {})};
    }

    return {
        mapping.lookup(dim),
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

static LogicalResult
emitIfAsSelect(Operation *ifOp, Value cond, affine::AffineValueMap map,
               OpBuilder &builder, IRMapping &mapping,
               DenseMap<Value, affine::AffineValueMap> &maps,
               ParallelContext pc) {
  Block *thenBlock = &ifOp->getRegion(0).front(),
        *elseBlock = &ifOp->getRegion(1).front();

  for (auto &innerOp : thenBlock->without_terminator()) {
    if (tryRaisingOpToStableHLO(&innerOp, mapping, builder, maps, pc).failed())
      return failure();
  }

  for (auto &innerOp : elseBlock->without_terminator()) {
    if (tryRaisingOpToStableHLO(&innerOp, mapping, builder, maps, pc).failed())
      return failure();
  }

  Operation *thenTerm = thenBlock->getTerminator(),
            *elseTerm = elseBlock->getTerminator();

  for (auto [thenVal, elseVal, res] :
       llvm::zip_equal(thenTerm->getOperands(), elseTerm->getOperands(),
                       ifOp->getResults())) {

    Value a = cond;
    Value b = mapping.lookup(thenVal);
    Value c = mapping.lookup(elseVal);

    auto mapA = map, mapB = maps.lookup(b), mapC = maps.lookup(c);

    Value dsts[] = {b, c};
    affine::AffineValueMap submaps[] = {mapB, mapC};
    auto outputMap = alignMemoryAccess(a, mapA, dsts, submaps, builder, pc);
    b = dsts[0];
    c = dsts[1];
    assert(b.getType() == c.getType());

    auto newOp = builder.create<stablehlo::SelectOp>(ifOp->getLoc(), a, b, c);
    mapping.map(res, newOp.getResult());
    maps[newOp.getResult()] = outputMap;
  }
  return success();
}

static Value
emitLoadAsGather(Location loc, Value mappedMemref, ValueRange lIndices,
                 OpBuilder &builder,
                 llvm::DenseMap<Value, affine::AffineValueMap> &maps) {
  Value indices = nullptr;

  SmallVector<int64_t> startIndexMap;
  SmallVector<int64_t> sliceSizes;
  SmallVector<int64_t> outputShape;
  SmallVector<Value> ivs;
  for (auto raisedIdx : lIndices) {
    startIndexMap.push_back(startIndexMap.size());

    auto Ty = cast<RankedTensorType>(raisedIdx.getType());

    SmallVector<int64_t> indicesShape(Ty.getShape().begin(),
                                      Ty.getShape().end());
    indicesShape.push_back(1);

    auto rank = Ty.getShape().size();
    // if (rank > 1) {
    //   LLVM_DEBUG(llvm::dbgs()
    //              << "failed to raised load (indices with rank > 1)\n");
    //   return nullptr;
    // }

    sliceSizes.push_back(1);

    SmallVector<int64_t> dimsToBroadcast;
    if (rank == 0) {
      raisedIdx = builder.create<stablehlo::ReshapeOp>(
          loc, Ty.clone({1}), raisedIdx); // tensor<1xi64>
      dimsToBroadcast.push_back(0);
      indicesShape.push_back(1);
    } else {
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
          outputShape.push_back(Ty.getShape()[i]);
          dimsToBroadcast.push_back(ivs.size());
          ivs.push_back(iv);
        } else {
          // this dim is already present
          dimsToBroadcast.push_back((int64_t)ivPos);
        }
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

      raisedIdx = builder.create<stablehlo::BroadcastInDimOp>(
          loc, Ty.clone(raisedIdxShape), raisedIdx, dimsToBroadcast);

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

      indices = builder.create<stablehlo::BroadcastInDimOp>(
          loc, Ty.clone(shape), indices, bDims);

      indicesTy = cast<RankedTensorType>(indices.getType());
      SmallVector<int64_t> newIndicesShape(
          indicesTy.getShape().drop_back().begin(),
          indicesTy.getShape().drop_back().end());
      newIndicesShape.push_back(
          indicesTy.getShape()[indicesTy.getShape().size() - 1] + 1);

      indices = builder.create<stablehlo::ConcatenateOp>(
          loc, Ty.clone(newIndicesShape), ValueRange{indices, raisedIdx},
          (int64_t)newIndicesShape.size() - 1);
    } else {

      auto S = cast<RankedTensorType>(raisedIdx.getType()).getShape();
      SmallVector<int64_t> shape(S.begin(), S.end());
      shape.push_back(1);

      indices =
          builder.create<stablehlo::ReshapeOp>(loc, Ty.clone(shape), raisedIdx);
    }
  }

  auto Ty = cast<RankedTensorType>(indices.getType());
  SmallVector<int64_t> productOfIndices = {1, (int64_t)lIndices.size()};

  for (auto s : Ty.getShape().drop_back()) {
    productOfIndices[0] *= s;
  }

  indices = builder.create<stablehlo::ReshapeOp>(
      loc, Ty.clone(productOfIndices), indices);

  Value res = builder.create<stablehlo::GatherOp>(
      loc, mappedMemref, indices,
      stablehlo::GatherDimensionNumbersAttr::get(
          loc.getContext(),
          /*offsetDims*/ {},
          /*collapsedSliceDims*/ startIndexMap,
          /*operandBatchingDims*/ {},
          /*startIndicesBatchingDims*/ {},
          /*startIndexMap*/ startIndexMap,
          /*indexVectorDim*/ 1),
      sliceSizes);

  auto OT = cast<RankedTensorType>(res.getType());
  res = builder.create<stablehlo::ReshapeOp>(loc, OT.clone(outputShape), res);

  affine::AffineValueMap outputMap(
      AffineMap::getMultiDimIdentityMap(ivs.size(), loc.getContext()), ivs);

  maps[res] = outputMap;

  return res;
}

static Value
emitStoreAsScatter(Location loc, Value update, Value input, ValueRange sIndices,
                   OpBuilder &builder,
                   llvm::DenseMap<Value, affine::AffineValueMap> &maps) {
  Value indices = nullptr;

  affine::AffineValueMap updateValueMap = maps.lookup(update);

  auto UTy = cast<RankedTensorType>(update.getType());
  SmallVector<int64_t> broadcastDims(UTy.getShape().size(), -1);
  SmallVector<int64_t> updateShape;
  SmallVector<int64_t> scatterDimsToOperandDims;

  for (auto [i, raisedIdx] : llvm::enumerate(sIndices)) {
    auto idxMap = maps.lookup(raisedIdx);

    auto Ty = cast<RankedTensorType>(raisedIdx.getType());
    SmallVector<int64_t> indicesShape(Ty.getShape().begin(),
                                      Ty.getShape().end());
    indicesShape.push_back(1);

    int64_t rank = Ty.getShape().size();
    if (rank > 1)
      return nullptr;

    scatterDimsToOperandDims.push_back(i);

    if (rank == 0) {
      indicesShape.push_back(1);
    } else {
      auto iv = getIVForExpr(idxMap, idxMap.getAffineMap().getResult(0));

      updateShape.push_back(
          cast<RankedTensorType>(raisedIdx.getType()).getShape()[0]);

      for (auto [updateIdx, E] :
           llvm::enumerate(updateValueMap.getAffineMap().getResults())) {
        Value updateIv = getIVForExpr(updateValueMap, E);
        if (updateIv == iv) {
          if (broadcastDims[updateIdx] != -1) {
            LLVM_DEBUG(llvm::dbgs()
                       << "todo: multiple ivs in different indices for load\n");
            return nullptr;
          }

          broadcastDims[updateIdx] = (updateShape.size() - 1);
        }
      }
    }

    raisedIdx = builder.create<stablehlo::ReshapeOp>(
        loc, Ty.clone(indicesShape), raisedIdx); // tensor<?x1xi64>

    if (indices) {
      int64_t indicesSize =
                  cast<RankedTensorType>(indices.getType()).getShape()[0],
              numDims = cast<RankedTensorType>(indices.getType()).getShape()[1],
              newSize =
                  cast<RankedTensorType>(raisedIdx.getType()).getShape()[0];

      indices = builder.create<stablehlo::BroadcastInDimOp>(
          loc, Ty.clone({indicesSize, newSize, numDims}), indices,
          llvm::ArrayRef<int64_t>({0, 2}));
      indices = builder.create<stablehlo::ReshapeOp>(
          loc, Ty.clone({indicesSize * newSize, numDims}), indices);
      raisedIdx = builder.create<stablehlo::BroadcastInDimOp>(
          loc, Ty.clone({indicesSize, newSize}), raisedIdx,
          llvm::ArrayRef<int64_t>({1, 0}));
      raisedIdx = builder.create<stablehlo::ReshapeOp>(
          loc, Ty.clone({indicesSize * newSize, 1}), raisedIdx);

      indices = builder.create<stablehlo::ConcatenateOp>(
          loc, Ty.clone({indicesSize * newSize, numDims + 1}),
          ValueRange{indices, raisedIdx}, 1);
    } else {
      indices = raisedIdx;
    }
  }

  if (llvm::any_of(broadcastDims, [](int64_t dim) { return dim == -1; })) {
    return nullptr;
  }

  update = builder.create<stablehlo::BroadcastInDimOp>(
      loc, cast<RankedTensorType>(update.getType()).clone(updateShape), update,
      broadcastDims);

  update = builder.create<stablehlo::ReshapeOp>(
      loc,
      RankedTensorType::get(
          {cast<RankedTensorType>(indices.getType()).getShape()[0]},
          cast<RankedTensorType>(update.getType()).getElementType()),
      update);

  auto Ty = cast<RankedTensorType>(input.getType());
  stablehlo::ScatterOp scatter = builder.create<stablehlo::ScatterOp>(
      loc, llvm::ArrayRef<Type>{Ty}, ValueRange{input}, indices,
      ValueRange{update},
      stablehlo::ScatterDimensionNumbersAttr::get(
          loc.getContext(),
          /*updateWindowDims*/ {},
          /*insertedWindowDims*/ scatterDimsToOperandDims,
          /*inputBatchingDims*/ {},
          /*scatterIndicesBatchingDims*/ {},
          /*scatterDimsToOperandDims*/ scatterDimsToOperandDims,
          /*indexVectorDim*/ 1),
      /*indicesAreSorted*/ false,
      /*uniqueIndices*/ false);
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
    builder.create<stablehlo::ReturnOp>(loc, updateInBody);
  }

  return res;
}

static LogicalResult tryRaisingForOpToStableHLOUnroll(
    affine::AffineForOp forOp, IRMapping &mapping, OpBuilder &builder,
    llvm::DenseMap<Value, affine::AffineValueMap> &maps, ParallelContext pc) {
  // Materialize an unrolled version of the loop in a temporary block and
  // generate the raised version of that. The unrolled version will be deleted
  // afterwards and the results of the original for loop will be mapped to it.

  // There arises a problem with the affine maps contained in the for loop which
  // until now correctly identified the loop iv as an affine dim, but will now
  // take constants as inputs. We need to canonicalize those maps before raising
  // the operations because we assume dim inputs to be loop ivs. This is the
  // reason we need to canonicalize all affine maps before we raise them in the
  // other parts of the code.
  auto tmpBlock = std::make_unique<Block>();
  OpBuilder oldFuncBuilder(builder.getContext());
  oldFuncBuilder.setInsertionPointToStart(tmpBlock.get());
  auto clonedFor = cast<affine::AffineForOp>(oldFuncBuilder.clone(*forOp));
  auto yield = oldFuncBuilder.create<affine::AffineYieldOp>(
      clonedFor.getLoc(), clonedFor.getResults());
  if (failed(affine::loopUnrollFull(clonedFor)))
    return failure();

  // Make a temporary new mapping because we will map values from the temporary
  // block which we will delete later.
  IRMapping forMapping = mapping;
  for (auto &innerOp : tmpBlock->without_terminator()) {
    if (tryRaisingOpToStableHLO(&innerOp, forMapping, builder, maps, pc)
            .failed())
      return failure();
  }
  // Remap the results of the loop in the main mapping which will be needed for
  // raising subsequent ops.
  for (auto [yielded, res] :
       llvm::zip_equal(yield.getOperands(), forOp.getResults())) {
    auto mapped = forMapping.lookupOrNull(yielded);
    assert(mapped);
    mapping.map(res, mapped);
  }
  for (auto [from, to] : forMapping.getValueMap()) {
    Block *b = from.getParentBlock();
    // This checks whether `tmpBlock` is an ancestor of `from`. If it is not,
    // then we need to reflect any change in the `forMapping` in the global
    // `mapping`. We need to do this because memref arguments to the function we
    // are raising get remapped as the raising process goes on.
    bool shouldRemap;
    while (true) {
      if (!b) {
        shouldRemap = true;
        break;
      }
      if (b == tmpBlock.get()) {
        shouldRemap = false;
        break;
      }
      Operation *op = b->getParentOp();
      if (op) {
        b = op->getBlock();
      } else {
        shouldRemap = true;
        break;
      }
    }
    if (shouldRemap)
      mapping.map(from, to);
  }
  return success();
}

static LogicalResult tryRaisingForOpToStableHLOWhile(
    affine::AffineForOp forOp, IRMapping &parentMapping, OpBuilder &builder,
    llvm::DenseMap<Value, affine::AffineValueMap> &maps, ParallelContext pc) {
  IRMapping mapping = parentMapping;
  if (!forOp.hasConstantBounds()) {
    LLVM_DEBUG(llvm::dbgs() << "ForOp does not have constant bounds\n");
    return failure();
  }

  Value iv = forOp.getInductionVar();
  InductionVariableRange range{forOp.getConstantLowerBound(),
                               forOp.getConstantUpperBound(),
                               forOp.getStepAsInt()};

  auto ET = builder.getI64Type();
  auto TT = RankedTensorType::get({}, ET);

  Value lb = builder.create<stablehlo::ConstantOp>(
      forOp.getLoc(), TT,
      SplatElementsAttr::get(
          TT, ArrayRef<Attribute>(IntegerAttr::get(ET, range.lb))));
  Value ub = builder.create<stablehlo::ConstantOp>(
      forOp.getLoc(), TT,
      SplatElementsAttr::get(
          TT, ArrayRef<Attribute>(IntegerAttr::get(ET, range.ub))));
  Value step = builder.create<stablehlo::ConstantOp>(
      forOp.getLoc(), TT,
      SplatElementsAttr::get(
          TT, ArrayRef<Attribute>(IntegerAttr::get(ET, range.step))));

  Block *entryBlock = &forOp->getParentOfType<func::FuncOp>().getBody().front();

  Block *cond = new Block(), *body = new Block();
  Value ivInCond = cond->addArgument(TT, iv.getLoc());
  Value ivInBody = body->addArgument(TT, iv.getLoc());

  SmallVector<Value> inits;
  inits.push_back(lb);

  for (auto [init, iterArg] :
       llvm::zip(forOp.getInits(), forOp.getRegionIterArgs())) {
    auto TT = pc.getTensorType(init.getType());
    cond->addArgument(TT, iterArg.getLoc());
    Value iterArgInBody = body->addArgument(TT, iterArg.getLoc());
    auto tensorInit = mapping.lookup(init);
    auto broadcastInit =
        pc.getBroadcast(builder, maps.lookup(tensorInit), tensorInit);
    if (!broadcastInit) {
      LLVM_DEBUG(llvm::dbgs() << "Could not broadcast an init\n"
                              << init << "\n");
      return failure();
    }
    inits.push_back(broadcastInit->v);
    mapping.map(iterArg, iterArgInBody);
    maps[iterArgInBody] = broadcastInit->avm;
  }

  for (auto memref : entryBlock->getArguments()) {
    Value mappedMemref = mapping.lookup(memref);
    inits.push_back(mappedMemref);

    cond->addArgument(mappedMemref.getType(), mappedMemref.getLoc());
    Value memrefInBody =
        body->addArgument(mappedMemref.getType(), mappedMemref.getLoc());
    mapping.map(memref, memrefInBody);
  }

  auto whileOp = builder.create<stablehlo::WhileOp>(forOp.getLoc(), inits);

  whileOp->getRegion(0).push_back(cond);
  whileOp->getRegion(1).push_back(body);

  {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(cond);
    Value cond = builder.create<stablehlo::CompareOp>(
        forOp.getLoc(), ivInCond, ub, stablehlo::ComparisonDirection::LT);
    builder.create<stablehlo::ReturnOp>(forOp.getLoc(), cond);
  }
  {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(body);

    mapping.map(iv, ivInBody);
    maps[ivInBody] =
        affine::AffineValueMap(AffineMap::get(forOp->getContext()), {});

    for (auto &innerOp : forOp.getBody()->without_terminator()) {
      if (tryRaisingOpToStableHLO(&innerOp, mapping, builder, maps, pc)
              .failed()) {
        LLVM_DEBUG(llvm::dbgs() << "Failed to raise inner op\n"
                                << innerOp << "\n");
        return failure();
      }
    }

    Value newIvInBody =
        builder.create<stablehlo::AddOp>(forOp.getLoc(), ivInBody, step);

    SmallVector<Value> loopCarried = {newIvInBody};
    for (auto [iterArg, yieldedIterArgs] :
         llvm::zip(forOp.getRegionIterArgs(),
                   forOp.getBody()->getTerminator()->getOperands())) {
      if (maps.lookup(mapping.lookup(iterArg)) !=
          maps.lookup(mapping.lookup(yieldedIterArgs))) {
        LLVM_DEBUG(llvm::dbgs() << "invalid init for iterArg: ";
                   iterArg.printAsOperand(llvm::dbgs(), OpPrintingFlags());
                   llvm::dbgs() << "\n");
        whileOp->erase();
        return failure();
      }
      loopCarried.push_back(mapping.lookup(yieldedIterArgs));
    }

    for (auto memref : entryBlock->getArguments())
      loopCarried.push_back(mapping.lookup(memref));
    builder.create<stablehlo::ReturnOp>(forOp.getLoc(), loopCarried);
  }

  for (auto [i, memref] : llvm::enumerate(entryBlock->getArguments()))
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
    if (!range.has_value())
      return failure();
    emitIVToStableHLO(builder, iv, *range, mapping, maps);
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
      auto reduce_map = maps.lookup(reduce_broadcasted);

      auto forOp = cast<affine::AffineForOp>(
          iters[reduced_idx].getOwner()->getParentOp());

      Value idx_broadcasted = mapping.lookup(forOp.getInductionVar());
      auto idx_map = maps.lookup(idx_broadcasted);

      Value dsts[] = {idx_broadcasted, mapping.lookup(init_val)};
      affine::AffineValueMap submaps[] = {idx_map, maps.lookup(dsts[1])};

      auto outputMap = alignMemoryAccess(reduce_broadcasted, reduce_map, dsts,
                                         submaps, builder, *newPc);

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
      Value init_values[1] = {builder.create<stablehlo::ConstantOp>(
          innerOp.getLoc(), builder.getZeroAttr(unrankedTensorType))};

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
      auto redwin = builder.create<stablehlo::ReduceWindowOp>(
          innerOp.getLoc(), restys, operands, init_values,
          builder.getDenseI64ArrayAttr(win_dim),
          builder.getDenseI64ArrayAttr(win_strides),
          builder.getDenseI64ArrayAttr(base_dialations),
          builder.getDenseI64ArrayAttr(win_dialations),
          DenseIntElementsAttr::get(
              RankedTensorType::get(padding_shape, builder.getIntegerType(64)),
              padding_dialations));

      auto block = new Block();
      redwin.getBody().push_back(block);

      auto a = block->addArgument(unrankedTensorType, innerOp.getLoc());
      auto b = block->addArgument(unrankedTensorType, innerOp.getLoc());

      {
        OpBuilder builder(block, block->end());
        auto addOp = builder.create<stablehlo::AddOp>(innerOp.getLoc(), a, b);
        builder.create<stablehlo::ReturnOp>(innerOp.getLoc(),
                                            addOp.getResult());
      }

      Value result = redwin->getResult(0);
      if (isa<arith::AddIOp, arith::AddFOp>(&innerOp)) {
        result =
            builder.create<stablehlo::AddOp>(innerOp.getLoc(), result, dsts[1]);
      } else if (isa<arith::SubIOp, arith::SubFOp>(&innerOp)) {
        result = builder.create<stablehlo::SubtractOp>(innerOp.getLoc(),
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
    auto val = mapping.lookup(yval);
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
            v = builder.create<arith::ConstantIndexOp>(res.getLoc(), 0);
          }
          vals.push_back(v);
        }

        auto newVal = builder.create<stablehlo::SliceOp>(
            res.getLoc(), val, startIndices, limitIndices, strides);

        SmallVector<int64_t> newShape;
        for (auto &&[i, sz] : llvm::enumerate(newVal.getType().getShape())) {
          if (i != idx_to_reduce) {
            newShape.push_back(sz);
          }
        }
        auto newVal2 = builder.create<stablehlo::ReshapeOp>(
            res.getLoc(),
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
        submaps.push_back(maps.lookup(dst));
      }
      auto outputMap2 = alignMemoryAccess(val, outputMap, dsts.data(), submaps,
                                          builder, *newPc);

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

      switch (kind) {
      case arith::AtomicRMWKind::addf:
      case arith::AtomicRMWKind::addi:
        break;
      default:
        return failure();
      }

      Value inputs[] = {val};
      Type types[] = {RankedTensorType::get(redshape, res.getType())};

      auto unrankedTensorType = RankedTensorType::get(
          {}, cast<RankedTensorType>(val.getType()).getElementType());
      Value inits[1] = {builder.create<stablehlo::ConstantOp>(
          res.getLoc(), builder.getZeroAttr(unrankedTensorType))};

      auto red = builder.create<stablehlo::ReduceOp>(
          val.getLoc(), types, inputs, inits,
          builder.getDenseI64ArrayAttr(idxs_to_reduce));

      auto block = new Block();
      red.getBody().push_back(block);

      auto a = block->addArgument(unrankedTensorType, res.getLoc());
      auto b = block->addArgument(unrankedTensorType, res.getLoc());

      {
        OpBuilder builder(block, block->end());
        auto addOp = builder.create<stablehlo::AddOp>(res.getLoc(), a, b);
        builder.create<stablehlo::ReturnOp>(res.getLoc(), addOp.getResult());
      }

      SmallVector<Value> vals;
      for (auto v : outputMap.getOperands()) {
        auto operand = dyn_cast<BlockArgument>(v);
        if (operand && operand.getOwner()->getParentOp() == pforOp) {
          v = builder.create<arith::ConstantIndexOp>(res.getLoc(), 0);
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

    auto inputTen = mapping.lookup(access.memref);

    SmallVector<int64_t> outputShape = affineMapShape(accessValueMap, pc);

    SmallVector<int64_t> strides;
    SmallVector<int64_t> reverseDims;

    if (affineMapToSlice(accessValueMap, strides, reverseDims, pc).failed()) {
      SmallVector<Value> lIndices;
      for (auto E : accessValueMap.getAffineMap().getResults()) {
        auto [idx, idxMap] = expandAffineExpr(
            builder, op->getLoc(), E, accessValueMap.getOperands(), mapping,
            accessValueMap.getAffineMap().getNumDims(), pc);
        maps[idx] = idxMap;
        lIndices.push_back(idx);
      }

      Value res =
          emitLoadAsGather(op->getLoc(), inputTen, lIndices, builder, maps);
      if (!res) {
        LLVM_DEBUG(llvm::dbgs()
                   << "failed to raise load (indices of rank > 1): " << *op
                   << "\n");
        return failure();
      }
      mapping.map(loadOp.getResult(), res);
      return success();
    }

    bool dynIndices = llvm::any_of(accessValueMap.getOperands(), [](Value iv) {
      return affine::isAffineForInductionVar(iv);
    });
    bool emitAsGather = dynIndices && llvm::any_of(strides, [](int64_t stride) {
                          return stride != 1;
                        });

    if (emitAsGather) {
      SmallVector<Value> lIndices;
      for (auto E : accessValueMap.getAffineMap().getResults()) {
        auto [idx, idxMap] = expandAffineExpr(
            builder, op->getLoc(), E, accessValueMap.getOperands(), mapping,
            accessValueMap.getAffineMap().getNumDims(), pc);
        maps[idx] = idxMap;
        lIndices.push_back(idx);
      }

      Value res =
          emitLoadAsGather(op->getLoc(), inputTen, lIndices, builder, maps);
      if (!res) {
        LLVM_DEBUG(llvm::dbgs()
                   << "failed to raise load (indices of rank > 1): " << *op
                   << "\n");
        return failure();
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
            builder, op->getLoc(), exprToEmit, accessValueMap.getOperands(),
            mapping, accessValueMap.getAffineMap().getNumDims(), pc);

        startIndices.push_back(startIndex);
      }

      newVal = builder.create<stablehlo::DynamicSliceOp>(
          op->getLoc(), T, inputTen, startIndices, outputShape);
    } else {
      bool needSlice = false;

      SmallVector<int64_t> startIndices;
      SmallVector<int64_t> limitIndices;

      for (auto [E, stride, sz] : llvm::zip_equal(
               accessValueMap.getAffineMap().getResults(), strides,
               cast<RankedTensorType>(inputTen.getType()).getShape())) {
        int64_t start, limit;
        if (auto constOp = dyn_cast<AffineConstantExpr>(E)) {
          start = constOp.getValue();
          limit = constOp.getValue() + 1;
          stride = 1;
        } else {
          auto range = computeExprRange(accessValueMap, E);
          start = range->step < 0 ? range->ub - range->step : range->lb;
          limit = range->step < 0 ? range->lb - range->step : range->ub;
        }

        needSlice |= sz != (limit - start) / stride;

        startIndices.push_back(start);
        limitIndices.push_back(limit);
      }

      if (needSlice)
        newVal = builder.create<stablehlo::SliceOp>(
            op->getLoc(), T, inputTen, startIndices, limitIndices, strides);
      else
        newVal = inputTen;
    }

    if (reverseDims.size())
      newVal = builder.create<stablehlo::ReverseOp>(inputTen.getLoc(), newVal,
                                                    reverseDims);

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

    newVal = builder
                 .create<stablehlo::ReshapeOp>(
                     newVal.getLoc(),
                     cast<RankedTensorType>(newVal.getType()).clone(dynShape),
                     newVal)
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
    auto operand = mapping.lookup(storeOp.getMemref());
    auto update = mapping.lookup(storeOp.getValue());

    affine::MemRefAccess access(storeOp);

    affine::AffineValueMap accessValueMap;
    access.getAccessMap(&accessValueMap);
    // See tryRaisingForOpToStableHLOUnroll
    accessValueMap.composeSimplifyAndCanonicalize();

    SmallVector<int64_t> strides;
    SmallVector<int64_t> reverseDims;

    if (affineMapToSlice(accessValueMap, strides, reverseDims, pc).failed()) {
      SmallVector<Value> sIndices;
      for (auto E : accessValueMap.getAffineMap().getResults()) {
        auto [expandedIndex, indexMap] = expandAffineExpr(
            builder, op->getLoc(), E, accessValueMap.getOperands(), mapping,
            accessValueMap.getAffineMap().getNumDims(), pc);
        maps[expandedIndex] = indexMap;
        sIndices.push_back(expandedIndex);
      }

      Value res = emitStoreAsScatter(op->getLoc(), update, operand, sIndices,
                                     builder, maps);
      if (!res) {
        LLVM_DEBUG(llvm::dbgs() << "affine.store (scatter) is dependent on "
                                   "less dims than stored value: "
                                << *op << "\n";
                   auto flags = OpPrintingFlags();
                   for (auto iv
                        : accessValueMap.getOperands()) {
                     iv.printAsOperand(llvm::dbgs(), flags);
                     llvm::dbgs() << ", ";
                   } llvm::dbgs()
                   << "\n";
                   accessValueMap.getAffineMap().dump();

        );
        return failure();
      }
      mapping.map(storeOp.getMemref(), res);
      return success();
    }

    bool emitAsScatter =
        llvm::any_of(strides, [](int64_t stride) { return stride != 1; });

    if (emitAsScatter) {
      // Cannot emit as a dynamic_update_slice, emit as scatter instead

      SmallVector<Value> sIndices;
      for (auto E : accessValueMap.getAffineMap().getResults()) {
        auto [expandedIndex, indexMap] = expandAffineExpr(
            builder, op->getLoc(), E, accessValueMap.getOperands(), mapping,
            accessValueMap.getAffineMap().getNumDims(), pc);
        maps[expandedIndex] = indexMap;
        sIndices.push_back(expandedIndex);
      }

      Value res = emitStoreAsScatter(op->getLoc(), update, operand, sIndices,
                                     builder, maps);
      if (!res) {
        LLVM_DEBUG(llvm::dbgs() << "affine.store (scatter) is dependent on "
                                   "less dims than stored value: "
                                << *op << "\n";
                   auto flags = OpPrintingFlags();
                   for (auto iv
                        : accessValueMap.getOperands()) {
                     iv.printAsOperand(llvm::dbgs(), flags);
                     llvm::dbgs() << ", ";
                   } llvm::dbgs()
                   << "\n";
                   accessValueMap.getAffineMap().dump();

        );
        return failure();
      }
      mapping.map(storeOp.getMemref(), res);
      return success();
    }

    auto rank = access.getRank();
    SmallVector<Value> startIndicesValues;
    startIndicesValues.reserve(rank);

    auto Ty = builder.getI64Type();
    auto unrankedTensorType = RankedTensorType::get({}, Ty);

    assert(maps.contains(update));
    affine::AffineValueMap updateValueMap = maps.lookup(update);

    // for each dim in update, where it will
    // be located in broadcastedupdate
    SmallVector<int64_t> broadcastDims(
        cast<RankedTensorType>(update.getType()).getShape().size(), -1);
    SmallVector<int64_t> updateShape;

    for (auto [E, stride] :
         llvm::zip_equal(accessValueMap.getAffineMap().getResults(), strides)) {

      Value startIndex;
      if (E.isSymbolicOrConstant()) {
        startIndex =
            builder
                .create<stablehlo::ConstantOp>(
                    op->getLoc(), unrankedTensorType,
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
            builder, iv.getLoc(), exprToEmit, accessValueMap.getOperands(),
            mapping, accessValueMap.getAffineMap().getNumDims(), pc);
        startIndex = startIndex_;
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
      LLVM_DEBUG(
          llvm::dbgs()
              << "affine.store is dependent on less dims than stored value: "
              << *op << "\n";
          auto flags = OpPrintingFlags(); for (auto iv
                                               : accessValueMap.getOperands()) {
            iv.printAsOperand(llvm::dbgs(), flags);
            llvm::dbgs() << ", ";
          } llvm::dbgs() << "\n";
          accessValueMap.getAffineMap().dump();
          for (auto iv
               : updateValueMap.getOperands()) {
            iv.printAsOperand(llvm::dbgs(), flags);
            llvm::dbgs() << ", ";
          } llvm::dbgs()
          << "\n";
          updateValueMap.getAffineMap().dump(););
      return failure();
    }

    update = builder.create<stablehlo::BroadcastInDimOp>(
        op->getLoc(),
        cast<RankedTensorType>(update.getType()).clone(updateShape), update,
        broadcastDims);

    if (!update)
      return failure();

    if (reverseDims.size())
      update = builder.create<stablehlo::ReverseOp>(storeOp.getLoc(), update,
                                                    reverseDims);

    auto newOperand = builder.create<stablehlo::DynamicUpdateSliceOp>(
        op->getLoc(), operand, update, startIndicesValues);

    mapping.map(storeOp.getMemref(), newOperand.getResult());
    return success();
  }

  if (auto loadOp = dyn_cast<memref::LoadOp>(op)) {
    auto memref = loadOp.getMemref();

    SmallVector<Value> lIndices;
    for (auto idx : loadOp.getIndices())
      lIndices.push_back(mapping.lookup(idx));

    Value res = emitLoadAsGather(op->getLoc(), mapping.lookup(memref), lIndices,
                                 builder, maps);
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
    for (auto idx : storeOp.getIndices())
      sIndices.push_back(mapping.lookup(idx));

    Value res =
        emitStoreAsScatter(op->getLoc(), mapping.lookup(value),
                           mapping.lookup(memref), sIndices, builder, maps);
    if (!res) {
      LLVM_DEBUG(llvm::dbgs()
                 << "memref.store is dependent on less dims than stored value: "
                 << *op << "\n");
      return failure();
    }

    mapping.map(memref, res);

    return success();
  }

  if (isa<ub::PoisonOp>(op)) {
    affine::AffineValueMap accessMap(AffineMap::get(op->getContext()), {});

    Type ET = op->getResult(0).getType();
    auto unrankedTensorType = RankedTensorType::get({}, ET);

    if (!ET.isInteger() && !isa<FloatType>(ET))
      return failure();

    auto newConst = builder.create<stablehlo::ConstantOp>(
        op->getLoc(), unrankedTensorType,
        SplatElementsAttr::get(
            unrankedTensorType,
            ArrayRef<Attribute>(
                ET.isInteger() ? (Attribute)IntegerAttr::get(ET, 0)
                               : (Attribute)FloatAttr::get(ET, APFloat(0.0)))));

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
    auto newConst = builder.create<stablehlo::ConstantOp>(
        op->getLoc(), unrankedTensorType,
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
  if (isa<arith::IndexCastUIOp, arith::IndexCastOp,
          enzymexla::Memref2PointerOp>(op)) {
    Value operand = op->getOperand(0), result = op->getResult(0);
    mapping.map(result, mapping.lookup(operand));
    return success();
  }

  if (auto apply = dyn_cast<affine::AffineApplyOp>(op)) {
    auto avm = apply.getAffineValueMap();
    // See tryRaisingForOpToStableHLOUnroll
    avm.composeSimplifyAndCanonicalize();
    auto [expanded, expandedMap] = expandAffineExpr(
        builder, apply.getLoc(), avm.getAffineMap().getResult(0),
        avm.getOperands(), mapping, avm.getAffineMap().getNumDims(), pc);
    mapping.map(apply.getResult(), expanded);
    maps[expanded] = expandedMap;
    return success();
  }

  if (auto p2m = dyn_cast<enzymexla::Pointer2MemrefOp>(op)) {
    Value operand = op->getOperand(0), result = op->getResult(0);
    auto input = mapping.lookup(operand);
    auto MT = p2m.getType();
    auto TT = RankedTensorType::get(MT.getShape(), MT.getElementType());
    auto res =
        builder.create<stablehlo::BitcastConvertOp>(p2m.getLoc(), TT, input);
    mapping.map(result, res);
    return success();
  }

  // unary ops
  if (isa<math::SinOp, math::SinhOp, math::CosOp, math::CoshOp, arith::NegFOp,
          arith::ExtUIOp, arith::SIToFPOp, arith::UIToFPOp, arith::FPToSIOp,
          arith::FPToUIOp, arith::TruncFOp, arith::ExtFOp, math::SqrtOp,
          math::RsqrtOp, math::CbrtOp, math::LogOp, math::ExpOp, math::AbsFOp,
          math::AbsIOp, math::IsNaNOp, math::AtanOp, arith::BitcastOp>(op)) {
    assert(op->getNumOperands() == 1 && op->getNumResults() == 1);

    auto operand = op->getOperand(0);
    auto newOperand = mapping.lookup(operand);

    auto IT = cast<RankedTensorType>(newOperand.getType());
    auto T = RankedTensorType::get(IT.getShape(),
                                   makeIndexToI64(op->getResult(0).getType()));

    auto newOp =
        Operation::create(op->getLoc(), op->getName(), {T}, {newOperand},
                          op->getAttrs(), OpaqueProperties(nullptr), {}, 0);
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
          math::CopySignOp, math::PowFOp>(op)) {
    assert(op->getNumOperands() == 2 && op->getNumResults() == 1);

    Value a = mapping.lookup(op->getOperand(0)),
          b = mapping.lookup(op->getOperand(1));

    auto mapA = maps.lookup(a), mapB = maps.lookup(b);
    auto outputMap = alignMemoryAccess(a, mapA, b, mapB, builder, pc);
    assert(a.getType() == b.getType());

    auto IT = cast<RankedTensorType>(a.getType());
    Type result = RankedTensorType::get(
        IT.getShape(), makeIndexToI64(op->getResult(0).getType()));

    auto newOp =
        Operation::create(op->getLoc(), op->getName(), {result}, {a, b},
                          op->getAttrs(), OpaqueProperties(nullptr), {}, 0);

    builder.insert(newOp);

    for (auto [oldRes, newRes] :
         llvm::zip_equal(op->getResults(), newOp->getResults())) {
      mapping.map(oldRes, newRes);
      maps[newRes] = outputMap;
    }

    return success();
  }

  // ternary ops
  if (isa<arith::SelectOp, math::FmaOp>(op)) {
    assert(op->getNumOperands() == 3 && op->getNumResults() == 1);

    Value a = mapping.lookup(op->getOperand(0)),
          b = mapping.lookup(op->getOperand(1)),
          c = mapping.lookup(op->getOperand(2));

    auto mapA = maps.lookup(a), mapB = maps.lookup(b), mapC = maps.lookup(c);

    Value dsts[] = {b, c};
    affine::AffineValueMap submaps[] = {mapB, mapC};
    auto outputMap = alignMemoryAccess(a, mapA, dsts, submaps, builder, pc);
    b = dsts[0];
    c = dsts[1];
    assert(b.getType() == c.getType());

    Type result = b.getType();

    auto newOp =
        Operation::create(op->getLoc(), op->getName(), {result}, {a, b, c},
                          op->getAttrs(), OpaqueProperties(nullptr), {}, 0);

    builder.insert(newOp);

    for (auto [oldRes, newRes] :
         llvm::zip_equal(op->getResults(), newOp->getResults())) {
      mapping.map(oldRes, newRes);
      maps[newRes] = outputMap;
    }

    return success();
  }

  if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
    if (!ifOp.elseBlock() || ifOp->getNumResults() == 0 ||
        llvm::any_of(*ifOp.thenBlock(),
                     [ifOp](Operation &op) {
                       return !isSafeToSpeculativelyExecuteAtScope(ifOp, &op);
                     }) ||
        llvm::any_of(*ifOp.elseBlock(), [ifOp](Operation &op) {
          return !isSafeToSpeculativelyExecuteAtScope(ifOp, &op);
        })) {
      LLVM_DEBUG(llvm::dbgs()
                 << "cannot raise if yet (non-pure or yielded values): " << *op
                 << "\n");
      return failure();
    }

    Value cond = mapping.lookup(ifOp.getCondition());
    if (emitIfAsSelect(op, cond, maps.lookup(cond), builder, mapping, maps, pc)
            .failed())
      return failure();

    return success();
  }

  if (auto ifOp = dyn_cast<affine::AffineIfOp>(op)) {
    if (!ifOp.hasElse() || ifOp->getNumResults() == 0 ||
        llvm::any_of(*ifOp.getThenBlock(),
                     [ifOp](Operation &op) {
                       return !isSafeToSpeculativelyExecuteAtScope(ifOp, &op);
                     }) ||
        llvm::any_of(*ifOp.getElseBlock(), [ifOp](Operation &op) {
          return !isSafeToSpeculativelyExecuteAtScope(ifOp, &op);
        })) {
      LLVM_DEBUG(llvm::dbgs()
                 << "cannot raise if yet (non-pure or yielded values): " << *op
                 << "\n");
      return failure();
    }

    auto is = ifOp.getIntegerSet();
    if (is.getNumSymbols() != 0) {
      LLVM_DEBUG(llvm::dbgs() << "cannot raise integer set with symbols yet\n");
      return failure(); // TODO
    }

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
          builder, ifOp.getLoc(), constraint, constraintMap.getOperands(),
          mapping, constraintMap.getNumDims(), pc);
      Value zero = builder.create<stablehlo::ConstantOp>(
          ifOp.getLoc(), cast<ShapedType>(expandedExpr.getType()),
          SplatElementsAttr::get(
              cast<ShapedType>(expandedExpr.getType()),
              ArrayRef<Attribute>(IntegerAttr::get(builder.getI64Type(), 0))));
      Value newCond = builder.create<stablehlo::CompareOp>(
          ifOp.getLoc(), expandedExpr, zero,
          eq ? stablehlo::ComparisonDirection::EQ
             : stablehlo::ComparisonDirection::GE);
      if (cond) {
        map = alignMemoryAccess(cond, map, newCond, outputMap, builder, pc);
        cond = builder.create<stablehlo::AndOp>(ifOp.getLoc(), cond, newCond);
      } else {
        cond = newCond;
        map = outputMap;
      }
    }

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
    if (pc.options.enableLockstepFor &&
        tryRaisingLockStepForOpToStableHLO(forOp, mapping, builder, maps, pc)
            .succeeded()) {
      return success();
    }
    if (pc.options.preferWhileRaising &&
        tryRaisingForOpToStableHLOWhile(forOp, mapping, builder, maps, pc)
            .succeeded()) {
      return success();
    }
    if (tryRaisingForOpToStableHLOUnroll(forOp, mapping, builder, maps, pc)
            .succeeded()) {
      return success();
    }
  }

  if (isa<LLVM::NoAliasScopeDeclOp>(op)) {
    return success();
  }

  LLVM_DEBUG(llvm::dbgs() << "cannot raise op to stablehlo: " << *op << "\n";);

  return failure();
}

static void
replaceAffineFuncWithStableHLOFunc(func::FuncOp oldFunc, func::FuncOp newFunc,
                                   llvm::ArrayRef<Operation *> users) {
  for (auto op : users) {
    auto user = dyn_cast<enzymexla::JITCallOp>(op);

    OpBuilder builder(user);
    auto newCall = builder.create<func::CallOp>(user->getLoc(), newFunc,
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
    auto TT = RankedTensorType::get(MT.getShape(), MT.getElementType());
    auto newArg = newBlock->addArgument(TT, arg.getLoc());
    mapping.map(arg, newArg);
    tensorTypes.push_back(TT);
  }

  auto modOp = func->getParentOfType<ModuleOp>();

  auto newFuncType =
      FunctionType::get(func->getContext(), tensorTypes, tensorTypes);

  std::string name = func.getSymName().str();
  name += "_raised";

  auto newFunc = func::FuncOp::create(func->getLoc(), name, newFuncType);
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

  builder.create<func::ReturnOp>(func->getLoc(), results);
  modOp.getBody()->push_back(newFunc);

  replaceAffineFuncWithStableHLOFunc(func, newFunc, users);

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
        auto add2 = rewriter.create<arith::AddFOp>(op.getLoc(), rhs,
                                                   lhs->getOperand(1 - j));
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

  void runOnOperation() override {
    ParallelContext::Options options{enable_lockstep_for, dump_failed_lockstep,
                                     prefer_while_raising};
    std::vector<func::FuncOp> funcs;

    auto context = getOperation()->getContext();

    if (enable_lockstep_for) {

      RewritePatternSet patterns(context);
      patterns.add<PushReductionsDown>(context);
      GreedyRewriteConfig config;
      if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                              std::move(patterns), config))) {
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

          if (matchPattern(arg, m_Constant(&attr))) {
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
                           .createNullValue(builder, arg.getLoc());
            } else {
              auto newConst = builder.create<stablehlo::ConstantOp>(
                  arg.getLoc(), unrankedTensorType,
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
              auto cl =
                  b.create<enzymexla::Pointer2MemrefOp>(arg.getLoc(), T, arg);
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
                  U->setOperand(0, B.create<enzymexla::Memref2PointerOp>(
                                       arg.getLoc(), arg.getType(), cl));
                }
              }
              operands.insert(cl);
              continue;
            }
          }

          if (buffered.find(arg) != buffered.end()) {
            continue;
          }

          if (isa<IntegerType, FloatType>(arg.getType())) {
            OpBuilder b(g);
            b.setInsertionPoint(g);
            auto MT0 =
                MemRefType::get({}, arg.getType(), MemRefLayoutAttrInterface{},
                                b.getI64IntegerAttr(0));
            auto MT =
                MemRefType::get({}, arg.getType(), MemRefLayoutAttrInterface{},
                                b.getI64IntegerAttr(1));

            auto res =
                b.create<gpu::AllocOp>(g.getLoc(), MT, (mlir::Type) nullptr,
                                       ValueRange(), ValueRange(), ValueRange())
                    ->getResult(0);

            auto res0 = b.create<memref::AllocaOp>(g.getLoc(), MT0);
            b.create<affine::AffineStoreOp>(g.getLoc(), arg, res0,
                                            b.getMultiDimIdentityMap(0),
                                            ValueRange());
            auto c1 = b.create<arith::ConstantIndexOp>(g.getLoc(), 1);
            b.create<enzymexla::MemcpyOp>(g.getLoc(), (mlir::Type) nullptr,
                                          ValueRange(), res, res0, c1);
            b.setInsertionPointToStart(body);
            auto ld = b.create<affine::AffineLoadOp>(
                g.getLoc(), res, b.getMultiDimIdentityMap(0), ValueRange());
            loads.push_back(ld);
            arg.replaceUsesWithIf(ld, [&](OpOperand &opOperand) {
              return g->isProperAncestor(opOperand.getOwner());
            });

            b.setInsertionPointAfter(g);
            b.create<gpu::DeallocOp>(g.getLoc(), (mlir::Type) nullptr,
                                     ValueRange(), res);
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
            ;
            signalPassFailure();
          }
          break;
        }
        auto TT = RankedTensorType::get(MT.getShape(), MT.getElementType());
        auto newArg = newBlock->addArgument(TT, arg.getLoc());
        mapping.map(arg, newArg);
        tensorTypes.push_back(TT);
      }
      if (failed) {
        delete newBlock;
        continue;
      }

      auto newFuncType =
          FunctionType::get(g->getContext(), tensorTypes, tensorTypes);

      std::string name = "raised";

      auto newFunc = func::FuncOp::create(g->getLoc(), name, newFuncType);
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

      builder.create<func::ReturnOp>(g->getLoc(), results);
      modOp.getBody()->push_back(newFunc);
      SymbolTable::setSymbolVisibility(newFunc,
                                       SymbolTable::Visibility::Private);

      {
        OpBuilder builder(g);
        builder.create<enzymexla::XLAWrapperOp>(
            g->getLoc(), SymbolRefAttr::get(newFunc), llvm::to_vector(operands),
            nullptr, nullptr);
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
