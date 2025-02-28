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
#include "mlir/IR/IRMapping.h"

#include "src/enzyme_ad/jax/Dialect/Ops.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_AFFINETOSTABLEHLORAISING
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

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

  InductionVariableRange range;
  range.lb = lb;
  range.ub = ub;
  range.step = step;

  return std::optional<InductionVariableRange>{range};
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
                                      SmallVectorImpl<int64_t> &strides) {
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

    startIndices.push_back(range->lb);
    limitIndices.push_back(range->ub);
    strides.push_back(range->step);
  }

  return success();
}

static SmallVector<int64_t>
affineMapShape(affine::AffineValueMap accessValueMap) {
  SmallVector<int64_t> startIndices;
  SmallVector<int64_t> limitIndices;
  SmallVector<int64_t> strides;

  if (affineMapToSlice(accessValueMap, startIndices, limitIndices, strides)
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
alignMemoryAccess(Value &a, affine::AffineValueMap src, Value &b,
                  affine::AffineValueMap dst, OpBuilder &builder) {
  // -> tensor<10x1xf32> loaded from (i) -> (i, 0)
  // -> to tensor<1x10xf32> written as (i) -> (0, i)
  SmallVector<int64_t> shapeA = affineMapShape(src);
  SmallVector<int64_t> shapeB = affineMapShape(dst);

  SmallVector<int64_t> outputShape;

  SmallVector<int64_t> broadcastDimensionsA(shapeA.size(), -1);
  SmallVector<int64_t> broadcastDimensionsB(shapeB.size(), -1);

  SmallVector<AffineExpr> exprs;
  SmallVector<Value> mapOperands;

  unsigned idxA = 0, idxB = 0;
  unsigned rankA = src.getNumResults(), rankB = dst.getNumResults();

  SetVector<Value> ivs;

  for (auto [i, EA] : llvm::enumerate(src.getAffineMap().getResults())) {
    broadcastDimensionsA[i] = outputShape.size();

    Value ivA = getIVForExpr(src, EA);

    for (unsigned j = 0, e = dst.getNumResults(); j < e; ++j) {
      auto EB = dst.getResult(j);
      if (getIVForExpr(dst, EB) == ivA) {
        broadcastDimensionsB[j] = outputShape.size();
      }
    }

    outputShape.push_back(shapeA[i]);

    exprs.push_back(
        mlir::getAffineDimExpr(mapOperands.size(), ivA.getContext()));
    mapOperands.push_back(ivA);
  }

  for (auto [i, EB] : llvm::enumerate(dst.getAffineMap().getResults())) {
    if (broadcastDimensionsB[i] != -1)
      continue; // dim already set in A

    Value ivB = getIVForExpr(dst, EB);

    broadcastDimensionsB[i] = outputShape.size();
    outputShape.push_back(shapeB[i]);

    exprs.push_back(
        mlir::getAffineDimExpr(mapOperands.size(), ivB.getContext()));
    mapOperands.push_back(ivB);
  }

  auto TA = a.getType().cast<RankedTensorType>(),
       TB = b.getType().cast<RankedTensorType>();

  a = builder
          .create<stablehlo::BroadcastInDimOp>(
              a.getLoc(), TA.clone(outputShape), a, broadcastDimensionsA)
          .getResult();

  b = builder
          .create<stablehlo::BroadcastInDimOp>(
              b.getLoc(), TB.clone(outputShape), b, broadcastDimensionsB)
          .getResult();

  affine::AffineValueMap outputMap(
      AffineMap::getMultiDimIdentityMap(mapOperands.size(), a.getContext()),
      mapOperands);

  return outputMap;
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

    if (affineMapToSlice(accessValueMap, startIndices, limitIndices, strides)
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

    Value newVal = sliceOp.getResult();
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

    if (affineMapToSlice(accessValueMap, startIndices, limitIndices, strides)
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

    // Store has less ivs than load.
    if (llvm::any_of(broadcastDims, [](int64_t dim) { return dim == -1; })) {
      return failure();
    }

    update = builder.create<stablehlo::BroadcastInDimOp>(
        op->getLoc(),
        update.getType().cast<RankedTensorType>().clone(updateShape), update,
        broadcastDims);

    if (!update)
      return failure();

    auto newOperand = builder.create<stablehlo::DynamicUpdateSliceOp>(
        op->getLoc(), operand, update, startIndicesValues);

    mapping.map(storeOp.getMemref(), newOperand.getResult());
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
  if (isa<arith::IndexCastUIOp>(op)) {
    Value operand = op->getOperand(0), result = op->getResult(0);
    mapping.map(result, mapping.lookup(operand));
    return success();
  }

  // unary ops
  if (isa<math::SinOp, math::SinhOp, math::CosOp, math::CoshOp, arith::NegFOp,
          arith::ExtUIOp, arith::SIToFPOp, math::SqrtOp, math::RsqrtOp,
          math::LogOp, math::ExpOp, math::AbsFOp, math::AbsIOp>(op)) {
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
          arith::DivUIOp, arith::OrIOp, arith::AndIOp, arith::CmpIOp,
          arith::CmpFOp, arith::ShRUIOp, arith::ShRSIOp, arith::ShLIOp,
          arith::MinimumFOp, arith::MaximumFOp, arith::MinUIOp, arith::MinSIOp,
          arith::MaxUIOp, arith::MaxSIOp, arith::RemSIOp, arith::RemUIOp>(op)) {
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
    auto outputMap = alignMemoryAccess(b, mapB, c, mapC, builder);
    outputMap = alignMemoryAccess(a, mapA, b, outputMap, builder);

    auto IT = b.getType().cast<RankedTensorType>();
    Type result = RankedTensorType::get(
        IT.getShape(), makeIndexToI64(op->getResult(0).getType()));

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
        if (!range.has_value())
          anyFailed = true;

        auto ET = builder.getI64Type();
        auto Ty = RankedTensorType::get({range->getNumIters()}, ET);
        Value iota =
            builder.create<stablehlo::IotaOp>(iv.getLoc(), Ty, 0).getResult();
        iota = builder.create<stablehlo::AddOp>(
            iv.getLoc(), Ty, iota,
            builder.create<stablehlo::ConstantOp>(
                iv.getLoc(), Ty,
                SplatElementsAttr::get(
                    Ty, ArrayRef<Attribute>(IntegerAttr::get(ET, range->lb)))));
        iota = builder.create<stablehlo::MulOp>(
            iv.getLoc(), Ty, iota,
            builder.create<stablehlo::ConstantOp>(
                iv.getLoc(), Ty,
                SplatElementsAttr::get(Ty, ArrayRef<Attribute>(IntegerAttr::get(
                                               ET, range->step)))));
        mapping.map(iv, iota);

        // contiguous with respect to itself: (d0) -> (d0)
        affine::AffineValueMap accessMap(
            AffineMap::getMultiDimIdentityMap(1, iv.getContext()), {iv});
        maps[iota] = accessMap;
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
