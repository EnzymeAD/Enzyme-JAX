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
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"

#include "src/enzyme_ad/jax/Dialect/Ops.h"
#include "stablehlo/dialect/StablehloOps.h"
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
  if (ty.isa<IndexType>())
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

static std::optional<InductionVariableRange> getIVRange(Value iv) {
  assert(affine::isAffineInductionVar(iv));

  auto owner = affine::getAffineParallelInductionVarOwner(iv);
  if (!owner) {
    auto forOp = affine::getForInductionVarOwner(iv);
    if (!forOp.hasConstantBounds())
      return std::nullopt;

    int64_t lb = forOp.getConstantLowerBound(),
            ub = forOp.getConstantUpperBound(), step = forOp.getStepAsInt();

    return InductionVariableRange{lb, ub, step};
  }

  if (owner.hasMinMaxBounds()) // Non-constant ranges.
    return std::nullopt;

  auto ivPos = cast<BlockArgument>(iv).getArgNumber();
  auto lb = owner.getLowerBoundMap(ivPos)
                .getResult(0)
                .cast<AffineConstantExpr>()
                .getValue();
  auto ub = owner.getUpperBoundMap(ivPos)
                .getResult(0)
                .cast<AffineConstantExpr>()
                .getValue();
  auto step = owner.getSteps()[ivPos];

  return InductionVariableRange{lb, ub, step};
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

    auto rhsConst = rhs.dyn_cast<AffineConstantExpr>();
    auto constantSide =
        rhsConst ? rhsConst : lhs.dyn_cast<AffineConstantExpr>();
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
      // %i = (0) to (180) step 1
      // -%i = (-0) to (-180) (step -1)
      if (const_ < 0) {
        // (i) in (lb) to (ub)
        // (-i) in (-ub) to (-lb) (step -1)
        range.lb = rangeDyn->lb * const_;
        range.ub = rangeDyn->ub * const_;
        range.step = rangeDyn->step * const_;
      } else {
        range.lb = rangeDyn->lb * const_;
        range.ub = rangeDyn->ub * const_;
        range.step = rangeDyn->step * const_;
      }
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
                                      SmallVectorImpl<int64_t> &startIndices,
                                      SmallVectorImpl<int64_t> &limitIndices,
                                      SmallVectorImpl<int64_t> &strides,
                                      SmallVectorImpl<int64_t> &reverseDims) {
  auto rank = accessValueMap.getNumResults();

  startIndices.reserve(rank);
  limitIndices.reserve(rank);
  strides.reserve(rank);

  for (unsigned i = 0; i < rank; i++) {
    auto expr = accessValueMap.getResult(i);

    if (auto constExpr = dyn_cast<AffineConstantExpr>(expr)) {
      auto const_ = constExpr.getValue();
      startIndices.push_back(const_);
      limitIndices.push_back(const_ + 1);
      strides.push_back(1);
      continue;
    }

    auto range = computeExprRange(accessValueMap, expr);

    if (!range.has_value())
      return failure();

    if (range->step < 0) {
      // 0:-1:-180 -> -179:1:1
      startIndices.push_back(range->ub - range->step);
      limitIndices.push_back(range->lb - range->step);
      strides.push_back(-range->step);
      reverseDims.push_back(i);
    } else {
      startIndices.push_back(range->lb);
      limitIndices.push_back(range->ub);
      strides.push_back(range->step);
    }
  }

  return success();
}

static SmallVector<int64_t>
affineMapShape(affine::AffineValueMap accessValueMap) {
  SmallVector<int64_t> startIndices;
  SmallVector<int64_t> limitIndices;
  SmallVector<int64_t> strides;
  SmallVector<int64_t> reverseDims;

  if (affineMapToSlice(accessValueMap, startIndices, limitIndices, strides,
                       reverseDims)
          .failed())
    return {};

  SmallVector<int64_t> shape;
  shape.reserve(startIndices.size());

  for (auto [lb, ub, step] :
       llvm::zip_equal(startIndices, limitIndices, strides)) {
    shape.push_back((ub - lb) / step);
  }

  return shape;
}

static affine::AffineValueMap
alignMemoryAccess(Value &a, affine::AffineValueMap src, Value *bs,
                  ArrayRef<affine::AffineValueMap> dsts, OpBuilder &builder) {
  // -> tensor<10x1xf32> loaded from (i) -> (i, 0)
  // -> to tensor<1x10xf32> written as (i) -> (0, i)

  SmallVector<int64_t> shapeA = affineMapShape(src);
  assert(shapeA.size() ==
         cast<RankedTensorType>(a.getType()).getShape().size());
  SmallVector<SmallVector<int64_t>> shapeBs;
  for (int i = 0; i < dsts.size(); i++) {
    shapeBs.push_back(affineMapShape(dsts[i]));
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

  unsigned idxA = 0;
  unsigned rankA = src.getNumResults();

  SetVector<Value> ivs;

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

  auto TA = a.getType().cast<RankedTensorType>();

  a = builder
          .create<stablehlo::BroadcastInDimOp>(
              a.getLoc(), TA.clone(outputShape), a, broadcastDimensionsA)
          .getResult();

  for (size_t i = 0; i < dsts.size(); i++) {
    auto TB = bs[i].getType().cast<RankedTensorType>();
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
                  affine::AffineValueMap dst, OpBuilder &builder) {
  Value bs[] = {b};
  affine::AffineValueMap dsts[] = {dst};
  auto res = alignMemoryAccess(a, src, bs, dsts, builder);
  b = bs[0];
  return res;
}

// like affine::expandAffineExpr but with stablehlo ops and returning
// the corresponding AffineValueMap for the produced value.
static std::tuple<Value, affine::AffineValueMap>
expandAffineExpr(OpBuilder &builder, Location loc, AffineExpr expr,
                 ValueRange operands, IRMapping &mapping) {
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
        expandAffineExpr(builder, loc, lhsExpr, operands, mapping);
    auto [rhs, rhsMap] =
        expandAffineExpr(builder, loc, rhsExpr, operands, mapping);

    affine::AffineValueMap outputMap =
        alignMemoryAccess(lhs, lhsMap, rhs, rhsMap, builder);

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
            loc, lhs, makeI64Constant(lhs.getType().cast<ShapedType>(), 0),
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
            loc, lhs, makeI64Constant(lhs.getType().cast<ShapedType>(), 0),
            stablehlo::ComparisonDirection::LE);
        Value one = makeI64Constant(lhs.getType().cast<ShapedType>(), 1);
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
            loc, lhs, makeI64Constant(lhs.getType().cast<ShapedType>(), 0),
            stablehlo::ComparisonDirection::LE);
        Value one = makeI64Constant(lhs.getType().cast<ShapedType>(), 1);
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

  if (auto dimExpr = dyn_cast<AffineDimExpr>(expr)) {
    Value dim = operands[dimExpr.getPosition()];
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
                        llvm::DenseMap<Value, affine::AffineValueMap> &maps);

static LogicalResult
emitIfAsSelect(Operation *ifOp, Value cond, affine::AffineValueMap map,
               OpBuilder &builder, IRMapping &mapping,
               DenseMap<Value, affine::AffineValueMap> &maps) {
  Block *thenBlock = &ifOp->getRegion(0).front(),
        *elseBlock = &ifOp->getRegion(1).front();

  for (auto &innerOp : thenBlock->without_terminator()) {
    if (tryRaisingOpToStableHLO(&innerOp, mapping, builder, maps).failed())
      return failure();
  }

  for (auto &innerOp : elseBlock->without_terminator()) {
    if (tryRaisingOpToStableHLO(&innerOp, mapping, builder, maps).failed())
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

    auto mapA = map, mapB = maps[b], mapC = maps[c];

    Value dsts[] = {b, c};
    affine::AffineValueMap submaps[] = {mapB, mapC};
    auto outputMap = alignMemoryAccess(a, mapA, dsts, submaps, builder);
    b = dsts[0];
    c = dsts[1];
    assert(b.getType() == c.getType());

    auto IT = b.getType().cast<RankedTensorType>();
    Type result = b.getType();

    auto newOp = builder.create<stablehlo::SelectOp>(ifOp->getLoc(), a, b, c);
    mapping.map(res, newOp.getResult());
    maps[newOp.getResult()] = outputMap;
  }
  return success();
}

static LogicalResult
tryRaisingOpToStableHLO(Operation *op, IRMapping &mapping, OpBuilder &builder,
                        llvm::DenseMap<Value, affine::AffineValueMap> &maps) {

  // Affine load inside a loop becomes a slice
  if (auto loadOp = dyn_cast<affine::AffineLoadOp>(op)) {
    affine::MemRefAccess access(loadOp);

    affine::AffineValueMap accessValueMap;
    access.getAccessMap(&accessValueMap);

    auto inputTen = mapping.lookup(access.memref);

    auto rank = access.getRank();
    SmallVector<int64_t> outputShape;
    outputShape.reserve(rank);

    SmallVector<int64_t> startIndices;
    SmallVector<int64_t> limitIndices;
    SmallVector<int64_t> strides;
    SmallVector<int64_t> reverseDims;

    if (affineMapToSlice(accessValueMap, startIndices, limitIndices, strides,
                         reverseDims)
            .failed())
      return failure();

    for (auto [lb, ub, step] :
         llvm::zip_equal(startIndices, limitIndices, strides)) {
      outputShape.push_back((ub - lb) / step);
    }

    auto T = RankedTensorType::get(
        outputShape,
        inputTen.getType().cast<RankedTensorType>().getElementType());

    auto sliceOp = builder.create<stablehlo::SliceOp>(
        op->getLoc(), T, inputTen, startIndices, limitIndices, strides);

    Value newVal = sliceOp.getResult();

    newVal = builder.create<stablehlo::ReverseOp>(inputTen.getLoc(), newVal,
                                                  reverseDims);

    SmallVector<AffineExpr> dynExprs;
    SmallVector<int64_t> dynShape;

    AffineMap affineMap = accessValueMap.getAffineMap();
    for (auto [S, E] : llvm::zip_equal(outputShape, affineMap.getResults())) {
      if (!E.isSymbolicOrConstant()) {
        dynExprs.push_back(E);
        dynShape.push_back(S);
      }
    }

    auto val = loadOp.getResult();

    newVal = builder
                 .create<stablehlo::ReshapeOp>(
                     newVal.getLoc(),
                     newVal.getType().cast<RankedTensorType>().clone(dynShape),
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

    SmallVector<int64_t> startIndices;
    SmallVector<int64_t> limitIndices;
    SmallVector<int64_t> strides;
    SmallVector<int64_t> reverseDims;

    if (affineMapToSlice(accessValueMap, startIndices, limitIndices, strides,
                         reverseDims)
            .failed())
      return failure();

    auto rank = access.getRank();
    SmallVector<Value> startIndicesValues;
    startIndicesValues.reserve(rank);

    auto Ty = builder.getI64Type();
    auto unrankedTensorType = RankedTensorType::get({}, Ty);

    affine::AffineValueMap updateValueMap = maps[update];

    // for each dim in update, where it will
    // be located in broadcastedupdate
    SmallVector<int64_t> broadcastDims(
        update.getType().cast<RankedTensorType>().getShape().size(), -1);
    SmallVector<int64_t> updateShape;

    for (auto [E, lb, ub, stride] :
         llvm::zip_equal(accessValueMap.getAffineMap().getResults(),
                         startIndices, limitIndices, strides)) {
      updateShape.push_back((ub - lb) / stride);

      startIndicesValues.push_back(
          builder
              .create<stablehlo::ConstantOp>(
                  op->getLoc(), unrankedTensorType,
                  SplatElementsAttr::get(
                      unrankedTensorType,
                      ArrayRef<Attribute>(IntegerAttr::get(Ty, lb))))
              .getResult());

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
        update.getType().cast<RankedTensorType>().clone(updateShape), update,
        broadcastDims);

    if (!update)
      return failure();

    update = builder.create<stablehlo::ReverseOp>(storeOp.getLoc(), update,
                                                  reverseDims);

    auto newOperand = builder.create<stablehlo::DynamicUpdateSliceOp>(
        op->getLoc(), operand, update, startIndicesValues);

    mapping.map(storeOp.getMemref(), newOperand.getResult());
    return success();
  }

  if (auto loadOp = dyn_cast<memref::LoadOp>(op)) {
    auto memref = loadOp.getMemref();

    Value indices = nullptr;

    SmallVector<int64_t> startIndexMap;
    SmallVector<int64_t> sliceSizes;
    SmallVector<int64_t> outputShape;
    SmallVector<Value> ivs;
    for (auto idx : loadOp.getIndices()) {
      startIndexMap.push_back(startIndexMap.size());

      auto raisedIdx = mapping.lookup(idx);
      auto Ty = raisedIdx.getType().cast<RankedTensorType>();

      SmallVector<int64_t> indicesShape(Ty.getShape().begin(),
                                        Ty.getShape().end());
      indicesShape.push_back(1);

      auto rank = Ty.getShape().size();
      assert(rank <= 1);

      sliceSizes.push_back(1);

      if (rank == 0) {
        raisedIdx = builder.create<stablehlo::ReshapeOp>(
            op->getLoc(), Ty.clone({1}), raisedIdx);
        indicesShape.push_back(1);
      } else {
        outputShape.push_back(Ty.getShape()[0]);

        auto map = maps[raisedIdx];
        assert(map.getNumResults() == 1);
        auto iv = getIVForExpr(map, map.getAffineMap().getResult(0));
        ivs.push_back(iv);
      }

      raisedIdx = builder.create<stablehlo::ReshapeOp>(
          op->getLoc(), Ty.clone(indicesShape), raisedIdx); // tensor<?x1xi64>

      if (indices) {
        int64_t indicesSize =
                    indices.getType().cast<RankedTensorType>().getShape()[0],
                numDims =
                    indices.getType().cast<RankedTensorType>().getShape()[1],
                newSize =
                    raisedIdx.getType().cast<RankedTensorType>().getShape()[0];

        indices = builder.create<stablehlo::BroadcastInDimOp>(
            op->getLoc(), Ty.clone({indicesSize, newSize, numDims}), indices,
            llvm::ArrayRef<int64_t>({0, 2}));
        indices = builder.create<stablehlo::ReshapeOp>(
            op->getLoc(), Ty.clone({indicesSize * newSize, numDims}), indices);
        raisedIdx = builder.create<stablehlo::BroadcastInDimOp>(
            op->getLoc(), Ty.clone({indicesSize, newSize}), raisedIdx,
            llvm::ArrayRef<int64_t>({1, 0}));
        raisedIdx = builder.create<stablehlo::ReshapeOp>(
            op->getLoc(), Ty.clone({indicesSize * newSize, 1}), raisedIdx);

        indices = builder.create<stablehlo::ConcatenateOp>(
            op->getLoc(), Ty.clone({indicesSize * newSize, numDims + 1}),
            ValueRange{indices, raisedIdx}, 1);
      } else {
        indices = raisedIdx;
      }
    }

    Value res = builder.create<stablehlo::GatherOp>(
        op->getLoc(), mapping.lookup(memref), indices,
        stablehlo::GatherDimensionNumbersAttr::get(
            loadOp.getContext(),
            /*offsetDims*/ {},
            /*collapsedSliceDims*/ startIndexMap,
            /*operandBatchingDims*/ {},
            /*startIndicesBatchingDims*/ {},
            /*startIndexMap*/ startIndexMap,
            /*indexVectorDim*/ 1),
        sliceSizes);

    auto OT = res.getType().cast<RankedTensorType>();
    res = builder.create<stablehlo::ReshapeOp>(op->getLoc(),
                                               OT.clone(outputShape), res);

    mapping.map(loadOp.getResult(), res);

    affine::AffineValueMap outputMap(
        AffineMap::getMultiDimIdentityMap(ivs.size(), op->getContext()), ivs);

    maps[res] = outputMap;

    return success();
  }

  if (auto constOp = dyn_cast<arith::ConstantOp>(op)) {
    affine::AffineValueMap accessMap(AffineMap::get(op->getContext()), {});

    auto isIndex = constOp.getType().isa<IndexType>();
    auto ET = isIndex ? builder.getI64Type() : constOp.getType();
    auto unrankedTensorType = RankedTensorType::get({}, ET);
    auto newConst = builder.create<stablehlo::ConstantOp>(
        op->getLoc(), unrankedTensorType,
        SplatElementsAttr::get(
            unrankedTensorType,
            ArrayRef<Attribute>(
                isIndex
                    ? IntegerAttr::get(
                          ET, constOp.getValue().cast<IntegerAttr>().getValue())
                    : constOp.getValueAttr())));
    auto newVal = newConst.getResult();
    mapping.map(constOp.getResult(), newVal);
    maps[newVal] = accessMap;

    return success();
  }

  // Identity
  if (isa<arith::IndexCastUIOp, arith::IndexCastOp>(op)) {
    Value operand = op->getOperand(0), result = op->getResult(0);
    mapping.map(result, mapping.lookup(operand));
    return success();
  }

  // unary ops
  if (isa<math::SinOp, math::SinhOp, math::CosOp, math::CoshOp, arith::NegFOp,
          arith::ExtUIOp, arith::SIToFPOp, math::SqrtOp, math::RsqrtOp,
          math::CbrtOp, math::LogOp, math::ExpOp, math::AbsFOp, math::AbsIOp,
          math::IsNaNOp>(op)) {
    assert(op->getNumOperands() == 1 && op->getNumResults() == 1);

    auto operand = op->getOperand(0);
    auto newOperand = mapping.lookup(operand);

    auto IT = newOperand.getType().cast<RankedTensorType>();
    auto T = RankedTensorType::get(IT.getShape(),
                                   makeIndexToI64(op->getResult(0).getType()));

    auto newOp =
        Operation::create(op->getLoc(), op->getName(), {T}, {newOperand},
                          op->getAttrs(), OpaqueProperties(nullptr), {}, 0);
    mapping.map(op->getResult(0), newOp->getResult(0));
    maps[newOp->getResult(0)] = maps[newOperand];

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
          arith::MaxSIOp, arith::RemSIOp, arith::RemUIOp, math::CopySignOp,
          math::PowFOp>(op)) {
    assert(op->getNumOperands() == 2 && op->getNumResults() == 1);

    Value a = mapping.lookup(op->getOperand(0)),
          b = mapping.lookup(op->getOperand(1));

    auto mapA = maps[a], mapB = maps[b];
    auto outputMap = alignMemoryAccess(a, mapA, b, mapB, builder);
    assert(a.getType() == b.getType());

    auto IT = a.getType().cast<RankedTensorType>();
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
  if (isa<arith::SelectOp>(op)) {
    assert(op->getNumOperands() == 3 && op->getNumResults() == 1);

    Value a = mapping.lookup(op->getOperand(0)),
          b = mapping.lookup(op->getOperand(1)),
          c = mapping.lookup(op->getOperand(2));

    auto mapA = maps[a], mapB = maps[b], mapC = maps[c];

    Value dsts[] = {b, c};
    affine::AffineValueMap submaps[] = {mapB, mapC};
    auto outputMap = alignMemoryAccess(a, mapA, dsts, submaps, builder);
    b = dsts[0];
    c = dsts[1];
    assert(b.getType() == c.getType());

    auto IT = b.getType().cast<RankedTensorType>();
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
                     [](Operation &op) { return !mlir::isPure(&op); }) ||
        llvm::any_of(*ifOp.elseBlock(),
                     [](Operation &op) { return !mlir::isPure(&op); })) {
      LLVM_DEBUG(llvm::dbgs()
                 << "cannot raise if yet (non-pure or yielded values): " << *op
                 << "\n");
      return failure();
    }

    Value cond = mapping.lookup(ifOp.getCondition());
    if (emitIfAsSelect(op, cond, maps[cond], builder, mapping, maps).failed())
      return failure();

    return success();
  }

  // Affine if (only pure ops with yield is currently supported)
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
    for (auto [constraint, eq] :
         llvm::zip_equal(is.getConstraints(), is.getEqFlags())) {
      auto [expandedExpr, outputMap] = expandAffineExpr(
          builder, ifOp.getLoc(), constraint, ifOp.getOperands(), mapping);
      Value zero = builder.create<stablehlo::ConstantOp>(
          ifOp.getLoc(), expandedExpr.getType().cast<ShapedType>(),
          SplatElementsAttr::get(
              expandedExpr.getType().cast<ShapedType>(),
              ArrayRef<Attribute>(IntegerAttr::get(builder.getI64Type(), 0))));
      Value newCond = builder.create<stablehlo::CompareOp>(
          ifOp.getLoc(), expandedExpr, zero,
          eq ? stablehlo::ComparisonDirection::EQ
             : stablehlo::ComparisonDirection::GE);
      if (cond) {
        map = alignMemoryAccess(cond, map, newCond, outputMap, builder);
        cond = builder.create<stablehlo::AndOp>(ifOp.getLoc(), cond, newCond);
      } else {
        cond = newCond;
        map = outputMap;
      }
    }

    if (emitIfAsSelect(op, cond, map, builder, mapping, maps).failed())
      return failure();

    return success();
  }

  // Inner parallel for loops
  if (auto parallelOp = dyn_cast<affine::AffineParallelOp>(op)) {
    if (parallelOp.hasMinMaxBounds() || !parallelOp.getReductions().empty())
      return failure();

    for (auto iv : parallelOp.getIVs()) {
      auto range = getIVRange(iv);
      if (!range.has_value())
        return failure();
      emitIVToStableHLO(builder, iv, *range, mapping, maps);
    }

    for (auto &innerOp : parallelOp.getBody()->without_terminator()) {
      if (tryRaisingOpToStableHLO(&innerOp, mapping, builder, maps).failed())
        return failure();
    }

    return success();
  }

  // // Inner for op
  // if (auto forOp = dyn_cast<affine::AffineForOp>(op)) {
  //   if (!forOp.hasConstantBounds())
  //     return failure();
  //
  //   Value iv = forOp.getInductionVar();
  //   InductionVariableRange range = *getIVRange(iv);
  //
  //   auto ET = builder.getI64Type();
  //   auto TT = RankedTensorType::get({}, ET);
  //
  //   Value lb = builder.create<stablehlo::ConstantOp>(
  //       forOp.getLoc(), TT,
  //       SplatElementsAttr::get(
  //           TT, ArrayRef<Attribute>(IntegerAttr::get(ET, range.lb))));
  //   Value ub = builder.create<stablehlo::ConstantOp>(
  //       forOp.getLoc(), TT,
  //       SplatElementsAttr::get(
  //           TT, ArrayRef<Attribute>(IntegerAttr::get(ET, range.ub))));
  //   Value step = builder.create<stablehlo::ConstantOp>(
  //       forOp.getLoc(), TT,
  //       SplatElementsAttr::get(
  //           TT, ArrayRef<Attribute>(IntegerAttr::get(ET, range.step))));
  //
  //   auto whileOp =
  //       builder.create<stablehlo::WhileOp>(forOp.getLoc(), ValueRange{lb});
  //
  //   Block *cond = new Block(), *body = new Block();
  //   whileOp->getRegion(0).push_back(cond);
  //   whileOp->getRegion(1).push_back(body);
  //
  //   Value ivInCond = cond->addArgument(TT, iv.getLoc());
  //   {
  //     OpBuilder::InsertionGuard guard(builder);
  //     builder.setInsertionPointToStart(cond);
  //     Value cond = builder.create<stablehlo::CompareOp>(
  //         forOp.getLoc(), ivInCond, step,
  //         stablehlo::ComparisonDirection::LT);
  //     builder.create<stablehlo::ReturnOp>(forOp.getLoc(), cond);
  //   }
  //   Value ivInBody = body->addArgument(TT, iv.getLoc());
  //   {
  //     OpBuilder::InsertionGuard guard(builder);
  //     builder.setInsertionPointToStart(body);
  //
  //     mapping.map(iv, ivInBody);
  //     maps[ivInBody] =
  //         affine::AffineValueMap(AffineMap::get(op->getContext()), {});
  //
  //     // Is this the only condition?
  //     SetVector<Value> mutatedMemrefs;
  //     op->walk([&](affine::AffineStoreOp AS) {
  //       mutatedMemrefs.insert(AS.getMemref());
  //     });
  //
  //     // for (auto m : mutatedMemrefs) {
  //     //   auto T = mapping.lookup(m).getType();
  //     //   whileOp->insertOperands();
  //     //   cond->addArgument(T, m.getLoc());
  //     //   body->addArgument(T, m.getLoc());
  //     // }
  //
  //     for (auto &innerOp : forOp.getBody()->without_terminator()) {
  //
  //       if (tryRaisingOpToStableHLO(&innerOp, mapping, builder,
  //       maps).failed())
  //         return failure();
  //     }
  //
  //     Value newIvInBody =
  //         builder.create<stablehlo::AddOp>(forOp.getLoc(), ivInBody, step);
  //
  //     SmallVector<Value> loopCarried = {newIvInBody};
  //     for (auto m : mutatedMemrefs)
  //       loopCarried.push_back(mapping.lookup(m));
  //     builder.create<stablehlo::ReturnOp>(forOp.getLoc(), loopCarried);
  //   }
  //
  //   return success();
  // }

  LLVM_DEBUG(llvm::dbgs() << "cannot raise op to stablehlo: " << *op << "\n";);

  return failure();
}

static void replaceAffineFuncWithStableHLOFunc(func::FuncOp oldFunc,
                                               func::FuncOp newFunc) {
  auto modOp = oldFunc->getParentOfType<ModuleOp>();

  SymbolTableCollection symbolTable;
  symbolTable.getSymbolTable(modOp);

  auto use_opt =
      symbolTable.getSymbolTable(modOp).getSymbolUses(oldFunc, modOp);
  for (auto use : *use_opt) {
    auto user = dyn_cast<enzymexla::JITCallOp>(use.getUser());

    OpBuilder builder(user);
    auto newCall = builder.create<func::CallOp>(user->getLoc(), newFunc,
                                                user->getOperands());

    auto operand_aliases = user.getOutputOperandAliases();
    assert(operand_aliases.size() == user.getNumResults());

    SmallVector<Value> replacements;
    size_t outputs = user.getNumResults();
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

  symbolTable.getSymbolTable(modOp).erase(oldFunc);
}

static bool tryRaisingToStableHLO(func::FuncOp func) {
  Block *body = &func->getRegion(0).front();
  Block *newBlock = new Block();

  IRMapping mapping;
  mapping.map(body, newBlock);

  SmallVector<Type> tensorTypes;
  for (auto arg : body->getArguments()) {
    auto MT = arg.getType().cast<MemRefType>();
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

  for (auto &it : body->without_terminator()) {
    Operation *bodyOp = &it;
    if (auto loopRoot = dyn_cast<affine::AffineParallelOp>(bodyOp)) {
      // TODO: support multiple nested loops

      // TODO: support multiple dimensions
      // if (loopRoot.getNumDims() != 1) {
      //   anyFailed = true;
      // };

      // TODO: handle reductions
      if (!loopRoot.getReductions().empty()) {
        anyFailed = true;
      }

      for (auto iv : loopRoot.getIVs()) {
        auto range = getIVRange(iv);
        if (!range.has_value()) {
          anyFailed = true;
          break;
        }

        emitIVToStableHLO(builder, iv, *range, mapping, maps);
      }

      if (!anyFailed) {
        auto loopBody = loopRoot.getBody();
        for (auto &it : loopBody->without_terminator()) {
          Operation *op = &it;
          anyFailed |=
              tryRaisingOpToStableHLO(op, mapping, builder, maps).failed();

          if (anyFailed)
            break;
        }
      }
    } else if (auto constOp = dyn_cast<arith::ConstantOp>(bodyOp)) {
      anyFailed =
          tryRaisingOpToStableHLO(bodyOp, mapping, builder, maps).failed();
      if (anyFailed)
        break;
    } else {
      anyFailed = true;
      break;
    }
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

  replaceAffineFuncWithStableHLOFunc(func, newFunc);

  return true;
}

namespace {
struct AffineToStableHLORaisingPass
    : public enzyme::impl::AffineToStableHLORaisingBase<
          AffineToStableHLORaisingPass> {
  using AffineToStableHLORaisingBase::AffineToStableHLORaisingBase;

  void runOnOperation() override {
    std::vector<func::FuncOp> funcs;

    auto op = getOperation();

    op->walk([&](func::FuncOp func) {
      auto FT = func.getFunctionType().dyn_cast<FunctionType>();

      // Identify raised kernels which takes in memrefs instead of tensors
      if (FT &&
          llvm::all_of(FT.getInputs(),
                       [](Type argTy) { return argTy.isa<MemRefType>(); }) &&
          FT.getNumResults() == 0) {
        funcs.push_back(func);
      }
    });

    bool anyRaised = false;
    while (!funcs.empty()) {
      auto kernelFunc = funcs.back();
      anyRaised |= tryRaisingToStableHLO(kernelFunc);
      funcs.pop_back();
    }

    if (!anyRaised) {
      markAllAnalysesPreserved();
    }
  }
};
} // namespace
