//===- AffineRaising.cpp - Raise affine loops to stablehlo ----------------===//
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
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/IRMapping.h"

#include "src/enzyme_ad/jax/Dialect/Ops.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_AFFINERAISE
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

using namespace mlir;
using namespace mlir::enzyme;

// Given an affine map for a load/store operation, compute the startIndices,
// limitIndices and strides corresponding in the memref based on the loop
// induction variables. (i) -> (0, i, 10) will give [0:1:1, 0:end:1, 10:11:1]
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
    } else if (auto dimExpr = dyn_cast<AffineDimExpr>(expr)) {
      Value iv = accessValueMap.getOperand(dimExpr.getPosition());
      assert(affine::isAffineInductionVar(iv));

      auto owner = affine::getAffineParallelInductionVarOwner(iv);

      if (!owner.getConstantRanges().has_value()) // Non-constant ranges.
        return failure();

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

      startIndices.push_back(lb);
      limitIndices.push_back(ub);
      strides.push_back(step);
    } else {
      return failure();
    }
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

  for (size_t i = 0, e = startIndices.size(); i < e; ++i) {
    int64_t lb = startIndices[i], ub = limitIndices[i], step = strides[i];
    shape.push_back((ub - lb) / step);
  }

  return shape;
}

static Value alignMemoryAccess(Value val, affine::AffineValueMap src,
                               affine::AffineValueMap dst, OpBuilder &builder) {
  // val -> tensor<10x1xf32> loaded from (i) -> (i, 0)
  // -> to tensor<1x10xf32> written as (i) -> (0, i)

  auto rank = src.getNumResults();
  if (rank > dst.getNumResults())
    return val;

  if (rank < dst.getNumResults()) {
    assert(rank == 0); // not sure this is valid otherwise. (i.e. if there are
                       // less moving dims)
    auto T = val.getType().cast<RankedTensorType>();

    val = builder
              .create<stablehlo::BroadcastInDimOp>(val.getLoc(),
                                                   T.clone(affineMapShape(dst)),
                                                   val, ArrayRef<int64_t>())
              .getResult();

    return val;
  }

  // Needs transpose
  SmallVector<int64_t> perm;
  perm.reserve(rank);

  AffineMap srcMap = src.getAffineMap(), dstMap = dst.getAffineMap();

  for (unsigned i = 0; i < rank; ++i) {
    auto srcExpr = src.getResult(i);
    if (srcExpr.getKind() == AffineExprKind::Constant) {
      perm.push_back(i);
      continue;
    }

    auto iv = src.getOperand(srcExpr.cast<AffineDimExpr>().getPosition());
    for (unsigned j = 0, e = dst.getNumResults(); j < e; ++j) {
      auto dstExpr = dst.getResult(j);
      if (auto dstDim = dstExpr.dyn_cast<AffineDimExpr>()) {
        auto dstIv = dst.getOperand(dstDim.getPosition());
        if (iv == dstIv) {
          perm.push_back(j);
          break;
        }
      }
    }
  }

  auto transposeOp =
      builder.create<stablehlo::TransposeOp>(val.getLoc(), val, perm);

  return transposeOp.getResult();
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

    auto val = loadOp.getResult();
    auto newVal = sliceOp.getResult();
    mapping.map(val, newVal);
    maps[newVal] = accessValueMap;

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
    for (auto idx : startIndices) {
      startIndicesValues.push_back(
          builder
              .create<stablehlo::ConstantOp>(
                  op->getLoc(), unrankedTensorType,
                  SplatElementsAttr::get(
                      unrankedTensorType,
                      ArrayRef<Attribute>(IntegerAttr::get(Ty, idx))))
              .getResult());
    }

    // TODO: here we need to make sure that the actual saved value has the right
    // transpose. consider the following kernel:
    //
    // affine.parallel (%arg1, %arg2) = (0, 0) to (100, 100) {
    //   %0 = affine.load %arg0[%arg2, %arg1] : memref<100x100xf32, 1>
    //   affine.store %0, %arg0[%arg1, %arg2] : memref<100x100xf32, 1>
    // }
    //
    // in this case, we want to transpose on the store (or read) to emit a
    // stablehlo.transpose.
    update = alignMemoryAccess(update, maps[update], accessValueMap, builder);

    auto newOperand = builder.create<stablehlo::DynamicUpdateSliceOp>(
        op->getLoc(), operand, update, startIndicesValues);

    mapping.map(storeOp.getMemref(), newOperand.getResult());
    return success();
  }

  if (auto constOp = dyn_cast<arith::ConstantOp>(op)) {
    auto unrankedTensorType =
        RankedTensorType::get({}, constOp.getResult().getType());

    auto newConst = builder.create<stablehlo::ConstantOp>(
        op->getLoc(), unrankedTensorType,
        SplatElementsAttr::get(unrankedTensorType,
                               ArrayRef<Attribute>(constOp.getValueAttr())));

    mapping.map(constOp.getResult(), newConst.getResult());
    return success();
  }

  // TODO: needs broadcasting for arguments
  if (isa<arith::MulIOp, arith::MulFOp, arith::AddIOp, arith::AddFOp>(op)) {
    assert(op->getNumOperands() == 2 && op->getNumResults() == 1);

    Value a = mapping.lookup(op->getOperand(0)),
          b = mapping.lookup(op->getOperand(1));

    auto mapA = maps[a], mapB = maps[b];

    a = alignMemoryAccess(a, mapA, mapB, builder);
    b = alignMemoryAccess(b, mapB, mapA, builder);

    assert(a.getType() == b.getType());

    Type result = a.getType();

    auto newOp =
        Operation::create(op->getLoc(), op->getName(), {result}, {a, b},
                          op->getAttrs(), OpaqueProperties(nullptr), {}, 0);

    builder.insert(newOp);

    for (auto [oldRes, newRes] :
         llvm::zip_equal(op->getResults(), newOp->getResults())) {
      mapping.map(oldRes, newRes);
      maps[newRes] = maps[mapping.lookup(op->getOperand(0))];
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
    auto user = use.getUser();

    assert(isa<enzymexla::JITCallOp>(user));

    OpBuilder builder(user);
    auto newCall = builder.create<func::CallOp>(user->getLoc(), newFunc,
                                                user->getOperands());

    for (auto [oldRes, newRes] :
         llvm::zip_equal(user->getResults(), newCall->getResults())) {
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

      if (!anyFailed) {
        auto loopBody = loopRoot.getBody();
        for (auto &it : loopBody->without_terminator()) {
          Operation *op = &it;
          anyFailed |=
              tryRaisingOpToStableHLO(op, mapping, builder, maps).failed();
        }
      }
    } else if (auto constOp = dyn_cast<arith::ConstantOp>(bodyOp)) {
      affine::AffineValueMap accessMap(AffineMap::get(bodyOp->getContext()),
                                       {});

      auto unrankedTensorType = RankedTensorType::get({}, constOp.getType());
      auto newConst = builder.create<stablehlo::ConstantOp>(
          bodyOp->getLoc(), unrankedTensorType,
          SplatElementsAttr::get(unrankedTensorType,
                                 ArrayRef<Attribute>(constOp.getValueAttr())));
      auto newVal = newConst.getResult();
      mapping.map(constOp.getResult(), newVal);
      maps[newVal] = accessMap;

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
struct AffineRaisePass : public enzyme::impl::AffineRaiseBase<AffineRaisePass> {
  using AffineRaiseBase::AffineRaiseBase;

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

    while (!funcs.empty()) {
      auto kernelFunc = funcs.back();
      tryRaisingToStableHLO(kernelFunc);
      funcs.pop_back();
    }
  }
};
} // namespace
