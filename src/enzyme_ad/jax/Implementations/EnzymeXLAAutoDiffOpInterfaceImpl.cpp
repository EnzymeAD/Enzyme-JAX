//===- EnzymeXLAAutoDiffOpInterfaceImpl.cpp - Interface external model ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the external model implementation of the automatic
// differentiation op interfaces for the EnzymeXLA dialect.
//
//===----------------------------------------------------------------------===//

#include "Enzyme/MLIR/Implementations/CoreDialectsAutoDiffImplementations.h"
#include "Enzyme/MLIR/Interfaces/AutoDiffOpInterface.h"
#include "Enzyme/MLIR/Interfaces/GradientUtils.h"
#include "Enzyme/MLIR/Interfaces/GradientUtilsReverse.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/RegionUtils.h"
#include "src/enzyme_ad/jax/Implementations/SHLOGenericBatchOpInterface.h"
#include "src/enzyme_ad/jax/Utils.h"

#include "Dialect/Ops.h"
#include "mlir/IR/TypeSupport.h"

#include "stablehlo/dialect/ChloOps.h"
#include "stablehlo/dialect/StablehloOps.h"

#include "src/enzyme_ad/jax/Dialect/Dialect.h"
#include "src/enzyme_ad/jax/Dialect/Ops.h"
#include "src/enzyme_ad/jax/Implementations/XLADerivatives.h"

using namespace mlir;
using namespace mlir::enzyme;
using namespace mlir::enzymexla;
using namespace mlir::stablehlo;

static int64_t to_i64(int64_t x) { return x; }
static int64_t to_i64(llvm::APInt x) { return x.getSExtValue(); }

static mlir::DenseI64ArrayAttr getI64Attr(OpBuilder &builder,
                                          llvm::ArrayRef<int64_t> vals) {
  return builder.getDenseI64ArrayAttr(vals);
}

static Type updateMemorySpace(Type typ, Attribute globalMemSpace) {
  return llvm::TypeSwitch<Type, Type>(typ)
      .Case<enzyme::CacheType>([&](auto cacheType) {
        return enzyme::CacheType::get(
            typ.getContext(),
            updateMemorySpace(cacheType.getType(), globalMemSpace));
      })
      .Case<MemRefType>([&](auto memrefType) {
        return *memrefType.clonePtrWith(globalMemSpace, std::nullopt);
      })
      .Default(typ);
};

namespace {
#include "src/enzyme_ad/jax/Implementations/EnzymeXLADerivatives.inc"

void traverseDownDefUseChains(
    SmallVectorImpl<Value> &frontier,
    function_ref<void(Operation *)> processOperation) {
  DenseSet<Value> visited;
  while (!frontier.empty()) {
    Value v = frontier.back();
    Operation *definingOp = v.getDefiningOp();
    frontier.pop_back();

    if (!definingOp)
      continue;

    processOperation(definingOp);

    for (Operation *user : v.getUsers())
      for (auto result : user->getResults())
        if (!visited.contains(result)) {
          frontier.push_back(result);
          visited.insert(result);
        }
  }
}

struct GPUWrapperOpEnzymeOpsRemover
    : public EnzymeOpsRemoverOpInterface::ExternalModel<
          GPUWrapperOpEnzymeOpsRemover, GPUWrapperOp> {

  LogicalResult removeEnzymeOps(Operation *op,
                                PatternRewriter &rewriter) const {

    auto wrapOp = cast<GPUWrapperOp>(op);

    // Gradients whose value is set in either branches.
    llvm::SetVector<Value> gradients;

    // We assume pushes are exclusive.
    llvm::MapVector<Value, CacheInfo> pushedCaches;

    // Grad to value
    IRMapping mapping;

    removalBlockExplore(&wrapOp.getRegion().front(), mapping, rewriter,
                        gradients, pushedCaches);

    if (gradients.empty() && pushedCaches.empty())
      return success();

    llvm::MapVector<Value, CacheInfo> cachesMap;
    for (auto &it : *wrapOp.getBody()) {
      Operation *op = &it;
      if (auto pushOp = dyn_cast<enzyme::PushOp>(op)) {
        CacheInfo info(pushOp.getCache());
        if (cachesMap.contains(pushOp.getValue()))
          info = info.merge(cachesMap.lookup(pushOp.getValue()), rewriter);
        cachesMap[pushOp.getValue()] = info;
      }
    }
    SmallVector<CacheInfo> caches =
        llvm::map_to_vector(cachesMap, [](auto p) { return std::get<1>(p); });

    if (caches.empty())
      return success();

    SetVector<Value> visited;
    getUsedValuesDefinedAbove(wrapOp.getBodyRegion(), visited);
    SmallVector<Value> frontier = llvm::map_to_vector(
        caches, [](CacheInfo info) { return info.pushedValue(); });
    SetVector<Operation *> opsToMove;
    // Traverse backward from pushed values to find operations that the pushed
    // value depends on
    while (!frontier.empty()) {
      Value v = frontier.back();
      Operation *definingOp = v.getDefiningOp();
      frontier.pop_back();

      if (!definingOp)
        continue;
      if (definingOp->getBlock() != &wrapOp.getBodyRegion().front())
        continue;

      // Assume allocations and frees are legal to move
      if (hasEffect<MemoryEffects::Read>(definingOp) ||
          hasEffect<MemoryEffects::Write>(definingOp)) {
        definingOp->emitError() << "cannot move op with side effects";
        return failure();
      }
      opsToMove.insert(definingOp);

      for (Value operand : definingOp->getOperands()) {
        if (visited.contains(operand))
          continue;

        frontier.push_back(operand);
        visited.insert(operand);
      }
    }

    // Move the push and dependent values outside of the wrapper
    OpBuilder::InsertionGuard guard(rewriter);
    IRMapping map;
    rewriter.setInsertionPoint(wrapOp);
    // Assume caches are in global memory (address space 1)
    auto globalMemSpace = rewriter.getI64IntegerAttr(1);
    for (Operation *toMove : llvm::reverse(opsToMove)) {
      Operation *cloned = rewriter.clone(*toMove, map);
      toMove->replaceAllUsesWith(cloned->getResults());

      if (auto allocOp = dyn_cast<memref::AllocOp>(cloned)) {
        auto gpuAlloc = gpu::AllocOp::create(
            rewriter, allocOp.getLoc(),
            *allocOp.getType().clonePtrWith(globalMemSpace, std::nullopt),
            /*asyncDependencies=*/ValueRange(), allocOp.getDynamicSizes(),
            /*symbolOperands=*/ValueRange());
        allocOp.replaceAllUsesWith(gpuAlloc.getResult(0));
        rewriter.eraseOp(allocOp);

        // Update the memory space of any users
        SmallVector<Value> frontier{gpuAlloc.getResult(0)};
        traverseDownDefUseChains(frontier, [globalMemSpace](Operation *op) {
          if (auto pushOp = dyn_cast<enzyme::PushOp>(op)) {
            Type newType =
                updateMemorySpace(pushOp.getCache().getType(), globalMemSpace);
            pushOp.getCache().setType(newType);
          }
          if (auto subviewOp = dyn_cast<memref::SubViewOp>(op)) {
            auto newType = cast<MemRefType>(
                updateMemorySpace(subviewOp.getType(), globalMemSpace));
            subviewOp.getResult().setType(newType);
          }
        });
      }
    }
    for (auto &info : caches) {
      rewriter.moveOpBefore(info.pushOp, wrapOp);
      auto revWrapper = info.popOp->getParentOfType<enzymexla::GPUWrapperOp>();
      // The pop has to be in the reverse kernel for the cache to be moved out
      // of this one. Without it there is nothing to move the pop before, and
      // the cache stays in a shape nothing further down can lower.
      if (!revWrapper)
        return info.popOp->emitError()
               << "cache pushed in this gpu_wrapper is popped outside of any "
                  "gpu_wrapper (in "
               << info.popOp->getParentOp()->getName() << ")";
      rewriter.moveOpBefore(info.popOp, revWrapper);

      SmallVector<Value> frontier{info.popOp.getResult()};
      traverseDownDefUseChains(frontier, [globalMemSpace](Operation *op) {
        if (auto subviewOp = dyn_cast<memref::SubViewOp>(op)) {
          auto newType = cast<MemRefType>(
              updateMemorySpace(subviewOp.getType(), globalMemSpace));
          subviewOp.getResult().setType(newType);
        }
      });

      for (auto user : info.popOp.getResult().getUsers()) {
        if (hasSingleEffect<MemoryEffects::Free>(user)) {
          rewriter.setInsertionPointAfter(revWrapper);
          gpu::DeallocOp::create(rewriter, wrapOp.getLoc(), TypeRange(),
                                 info.popOp.getResult());
          rewriter.eraseOp(user);
        }
      }
    }

    return success();
  }
};

// Reverse-mode adjoint for pure view-cast ops (Pointer2Memref /
// Memref2Pointer). We only need to materialize the corresponding shadow view
// in the augmented primal and register it via setInvertedPointer so downstream
// memref.load / memref.store consumers can locate it through invertPointerM.
// This mirrors the pattern used by LLVM::GEPOp and memref::SubViewOp.
template <typename OpTy>
struct ViewCastOpInterfaceReverse
    : public ReverseAutoDiffOpInterface::ExternalModel<
          ViewCastOpInterfaceReverse<OpTy>, OpTy> {
  LogicalResult createReverseModeAdjoint(Operation *op, OpBuilder &builder,
                                         MGradientUtilsReverse *gutils,
                                         SmallVector<Value> caches) const {
    return success();
  }

  SmallVector<Value> cacheValues(Operation *op,
                                 MGradientUtilsReverse *gutils) const {
    return {};
  }

  LogicalResult createShadowValues(Operation *op, OpBuilder &builder,
                                   MGradientUtilsReverse *gutils) const {
    auto castOp = cast<OpTy>(op);
    auto newCastOp = cast<OpTy>(gutils->getNewFromOriginal(op));
    Value source = castOp.getSource();
    if (!gutils->isConstantValue(source)) {
      Value sourceShadow = gutils->invertPointerM(source, builder);
      auto shadowCast = cast<OpTy>(builder.clone(*newCastOp));
      shadowCast.getSourceMutable().assign(sourceShadow);
      gutils->setInvertedPointer(castOp.getResult(), shadowCast->getResult(0));
    }
    return success();
  }
};

struct GPUWrapperOpInterfaceReverse
    : public ReverseAutoDiffOpInterface::ExternalModel<
          GPUWrapperOpInterfaceReverse, GPUWrapperOp> {
  LogicalResult createReverseModeAdjoint(Operation *op, OpBuilder &builder,
                                         MGradientUtilsReverse *gutils,
                                         SmallVector<Value> caches) const {

    auto wrapOp = cast<GPUWrapperOp>(op);

    SmallVector<Value> operands;
    for (auto v : caches) {
      operands.push_back(gutils->popCache(v, builder));
    }

    auto repFor = GPUWrapperOp::create(builder, wrapOp.getLoc(), operands);

    bool valid = true;
    for (auto &&[oldReg, newReg] :
         llvm::zip(op->getRegions(), repFor->getRegions())) {
      for (auto &&[oBB, revBB] : llvm::zip(oldReg, newReg)) {
        OpBuilder bodyBuilder(&revBB, revBB.end());

        // Create implicit terminator if not present (when num results > 0)
        if (revBB.empty()) {
          YieldOp::create(bodyBuilder, repFor->getLoc(), ValueRange());
        }
        bodyBuilder.setInsertionPoint(revBB.getTerminator());

        auto first = oBB.rbegin();
        first++; // skip terminator

        auto last = oBB.rend();

        for (auto it = first; it != last; ++it) {
          Operation *op = &*it;
          valid &=
              gutils->Logic.visitChild(op, bodyBuilder, gutils).succeeded();
        }
      }
    }

    return success(valid);
  }

  SmallVector<Value> cacheValues(Operation *op,
                                 MGradientUtilsReverse *gutils) const {
    auto wrapOp = cast<GPUWrapperOp>(op);

    Operation *newOp = gutils->getNewFromOriginal(op);
    OpBuilder cacheBuilder(newOp);
    SmallVector<Value> caches;

    for (auto val : wrapOp.getBlockDims()) {
      Value cacheLB = gutils->initAndPushCache(gutils->getNewFromOriginal(val),
                                               cacheBuilder);
      caches.push_back(cacheLB);
    }

    return caches;
  }

  LogicalResult createShadowValues(Operation *op, OpBuilder &builder,
                                   MGradientUtilsReverse *gutils) const {
    return success();
  }
};

} // namespace

// Batched, the ops that act along one dimension of their operand act along
// the same dimension past the batch dimensions.
template <typename OpTy>
struct DimensionedOpBatchInterface
    : public BatchOpInterface::ExternalModel<DimensionedOpBatchInterface<OpTy>,
                                             OpTy> {
  mlir::LogicalResult createBatch(Operation *src, OpBuilder &builder,
                                  IRMapping &mapper,
                                  ArrayRef<int64_t> batchSizes) const {
    auto op = cast<OpTy>(src);
    auto n = OpTy::create(builder, op.getLoc(), mapper.lookup(op.getOperand()),
                          op.getLhs(), op.getRhs(),
                          op.getDimension() + batchSizes.size());
    mapper.map(src->getResult(0), n.getResult());
    return success();
  }
};

struct RotateOpBatchInterface
    : public BatchOpInterface::ExternalModel<RotateOpBatchInterface, RotateOp> {
  mlir::LogicalResult createBatch(Operation *src, OpBuilder &builder,
                                  IRMapping &mapper,
                                  ArrayRef<int64_t> batchSizes) const {
    auto op = cast<RotateOp>(src);
    auto n =
        RotateOp::create(builder, op.getLoc(), mapper.lookup(op.getOperand()),
                         op.getAmount(), op.getDimension() + batchSizes.size());
    mapper.map(src->getResult(0), n.getResult());
    return success();
  }
};

static Value createScalarConst(OpBuilder &builder, Location loc, Type elTy,
                               double val) {
  auto scalarTy = RankedTensorType::get({}, elTy);
  return stablehlo::ConstantOp::create(
      builder, loc, scalarTy, cast<ElementsAttr>(makeAttr(scalarTy, val)));
}

static Value createSplatConst(OpBuilder &builder, Location loc,
                              RankedTensorType ty, double val) {
  auto scalarTy = RankedTensorType::get({}, ty.getElementType());
  auto attr = SplatElementsAttr::get(
      ty,
      cast<ElementsAttr>(makeAttr(scalarTy, val)).getValues<Attribute>()[0]);
  return stablehlo::ConstantOp::create(builder, loc, ty, attr);
}

static Value createNegInfConst(OpBuilder &builder, Location loc,
                               RankedTensorType ty) {
  auto fltTy = cast<FloatType>(ty.getElementType());
  APFloat negInf =
      APFloat::getInf(fltTy.getFloatSemantics(), /*Negative=*/true);
  auto attr = SplatElementsAttr::get(ty, builder.getFloatAttr(fltTy, negInf));
  return stablehlo::ConstantOp::create(builder, loc, ty, attr);
}

static Value convertElementType(OpBuilder &builder, Location loc, Value v,
                                Type targetElTy) {
  auto ty = cast<RankedTensorType>(v.getType());
  if (ty.getElementType() == targetElTy)
    return v;
  return stablehlo::ConvertOp::create(
      builder, loc, RankedTensorType::get(ty.getShape(), targetElTy), v);
}

static Value reduceSumAlongDims(OpBuilder &builder, Location loc, Value input,
                                ArrayRef<int64_t> dims) {
  if (dims.empty())
    return input;
  auto inTy = cast<RankedTensorType>(input.getType());
  Type elTy = inTy.getElementType();
  SmallVector<int64_t> sortedDims(dims.begin(), dims.end());
  llvm::sort(sortedDims);
  SmallVector<int64_t> outShape;
  for (int64_t d = 0; d < inTy.getRank(); ++d) {
    if (!llvm::is_contained(sortedDims, d))
      outShape.push_back(inTy.getDimSize(d));
  }
  auto outTy = RankedTensorType::get(outShape, elTy);
  Value initVal = createScalarConst(builder, loc, elTy, 0.0);
  auto reduceOp = stablehlo::ReduceOp::create(
      builder, loc, TypeRange{outTy}, ValueRange{input}, ValueRange{initVal},
      builder.getDenseI64ArrayAttr(sortedDims));
  Block *block = new Block();
  reduceOp.getBody().push_back(block);
  auto argTy = RankedTensorType::get({}, elTy);
  block->addArgument(argTy, loc);
  block->addArgument(argTy, loc);
  {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(block);
    Value sum = stablehlo::AddOp::create(builder, loc, block->getArgument(0),
                                         block->getArgument(1));
    stablehlo::ReturnOp::create(builder, loc, ValueRange{sum});
  }
  return reduceOp.getResult(0);
}

static Value flattenBatchOperand(OpBuilder &builder, Location loc, Value val,
                                 ArrayRef<int64_t> batchSizes, int64_t B) {
  if (!val)
    return Value();
  auto ty = cast<RankedTensorType>(val.getType());
  int64_t nBatch = batchSizes.size();
  int64_t origRank = ty.getRank() - nBatch;
  int64_t d0 = ty.getDimSize(nBatch);
  Value cur = val;
  if (d0 == 1 && B != 1) {
    SmallVector<int64_t> bcastShape(batchSizes.begin(), batchSizes.end());
    bcastShape.push_back(B);
    for (int64_t i = 1; i < origRank; ++i)
      bcastShape.push_back(ty.getDimSize(nBatch + i));
    SmallVector<int64_t> bcastDims;
    for (int64_t i = 0; i < ty.getRank(); ++i)
      bcastDims.push_back(i);
    cur = stablehlo::BroadcastInDimOp::create(
        builder, loc, RankedTensorType::get(bcastShape, ty.getElementType()),
        cur, builder.getDenseI64ArrayAttr(bcastDims));
  }
  int64_t totalB = B;
  for (int64_t bs : batchSizes)
    totalB *= bs;
  SmallVector<int64_t> flatShape = {totalB};
  for (int64_t i = 1; i < origRank; ++i)
    flatShape.push_back(ty.getDimSize(nBatch + i));
  return stablehlo::ReshapeOp::create(
      builder, loc, RankedTensorType::get(flatShape, ty.getElementType()), cur);
}

static Value unflattenBatchResult(OpBuilder &builder, Location loc, Value val,
                                  ArrayRef<int64_t> batchSizes, int64_t B,
                                  ArrayRef<int64_t> origShape,
                                  bool sumOverSharedBatch = false) {
  auto ty = cast<RankedTensorType>(val.getType());
  SmallVector<int64_t> fullShape(batchSizes.begin(), batchSizes.end());
  fullShape.push_back(B);
  for (size_t i = 1; i < origShape.size(); ++i)
    fullShape.push_back(origShape[i]);
  Value unflat = stablehlo::ReshapeOp::create(
      builder, loc, RankedTensorType::get(fullShape, ty.getElementType()), val);
  if (sumOverSharedBatch && origShape[0] == 1 && B != 1) {
    Value reduced = reduceSumAlongDims(
        builder, loc, unflat, {static_cast<int64_t>(batchSizes.size())});
    SmallVector<int64_t> finalShape(batchSizes.begin(), batchSizes.end());
    finalShape.append(origShape.begin(), origShape.end());
    return stablehlo::ReshapeOp::create(
        builder, loc, RankedTensorType::get(finalShape, ty.getElementType()),
        reduced);
  }
  return unflat;
}

template <typename OpTy>
struct AttentionActivity
    : public ActivityOpInterface::ExternalModel<AttentionActivity<OpTy>, OpTy> {
  bool isInactive(Operation *op) const { return false; }
  bool isArgInactive(Operation *op, size_t idx) const {
    auto ty = dyn_cast<ShapedType>(op->getOperand(idx).getType());
    return !ty || !isa<FloatType>(ty.getElementType());
  }
};

static Value expandAttentionKVHeads(OpBuilder &builder, Location loc, Value kv,
                                    int64_t N_q, int64_t W) {
  auto ty = cast<RankedTensorType>(kv.getType());
  if (W == 1) {
    int64_t B = ty.getDimSize(0);
    int64_t S = ty.getDimSize(1);
    int64_t N_kv = ty.getDimSize(2);
    int64_t D = ty.getDimSize(3);
    int64_t G = N_q / N_kv;
    if (G == 1)
      return kv;
    auto ty5D = RankedTensorType::get({B, S, N_kv, G, D}, ty.getElementType());
    Value bcast = stablehlo::BroadcastInDimOp::create(
        builder, loc, ty5D, kv, builder.getDenseI64ArrayAttr({0, 1, 2, 4}));
    return stablehlo::ReshapeOp::create(
        builder, loc,
        RankedTensorType::get({B, S, N_q, D}, ty.getElementType()), bcast);
  }
  int64_t B = ty.getDimSize(1);
  int64_t S = ty.getDimSize(2);
  int64_t N_kv = ty.getDimSize(3);
  int64_t D = ty.getDimSize(4);
  int64_t G = N_q / N_kv;
  if (G == 1)
    return kv;
  auto ty6D = RankedTensorType::get({W, B, S, N_kv, G, D}, ty.getElementType());
  Value bcast = stablehlo::BroadcastInDimOp::create(
      builder, loc, ty6D, kv, builder.getDenseI64ArrayAttr({0, 1, 2, 3, 5}));
  return stablehlo::ReshapeOp::create(
      builder, loc,
      RankedTensorType::get({W, B, S, N_q, D}, ty.getElementType()), bcast);
}

static Value reduceAttentionKVHeads(OpBuilder &builder, Location loc,
                                    Value kvExp, int64_t N_kv, int64_t W) {
  auto ty = cast<RankedTensorType>(kvExp.getType());
  if (W == 1) {
    int64_t B = ty.getDimSize(0);
    int64_t S = ty.getDimSize(1);
    int64_t N_q = ty.getDimSize(2);
    int64_t D = ty.getDimSize(3);
    int64_t G = N_q / N_kv;
    if (G == 1)
      return kvExp;
    Value kv5D = stablehlo::ReshapeOp::create(
        builder, loc,
        RankedTensorType::get({B, S, N_kv, G, D}, ty.getElementType()), kvExp);
    return reduceSumAlongDims(builder, loc, kv5D, {3});
  }
  int64_t B = ty.getDimSize(1);
  int64_t S = ty.getDimSize(2);
  int64_t N_q = ty.getDimSize(3);
  int64_t D = ty.getDimSize(4);
  int64_t G = N_q / N_kv;
  if (G == 1)
    return kvExp;
  Value kv6D = stablehlo::ReshapeOp::create(
      builder, loc,
      RankedTensorType::get({W, B, S, N_kv, G, D}, ty.getElementType()), kvExp);
  return reduceSumAlongDims(builder, loc, kv6D, {4});
}

static Value prependAttentionWidth(OpBuilder &builder, Location loc, Value val,
                                   int64_t W) {
  if (W == 1)
    return val;
  auto ty = cast<RankedTensorType>(val.getType());
  SmallVector<int64_t> newShape = {W};
  newShape.append(ty.getShape().begin(), ty.getShape().end());
  SmallVector<int64_t> bcastDims;
  for (int64_t i = 0; i < ty.getRank(); ++i)
    bcastDims.push_back(i + 1);
  return stablehlo::BroadcastInDimOp::create(
      builder, loc, RankedTensorType::get(newShape, ty.getElementType()), val,
      builder.getDenseI64ArrayAttr(bcastDims));
}

static Value broadcastAttentionBias(OpBuilder &builder, Location loc,
                                    Value biasVal, RankedTensorType targetTy,
                                    int64_t W) {
  if (biasVal.getType() == targetTy)
    return biasVal;
  SmallVector<int64_t> dims = (W == 1) ? SmallVector<int64_t>{0, 1, 2, 3}
                                       : SmallVector<int64_t>{0, 1, 2, 3, 4};
  return stablehlo::BroadcastInDimOp::create(
      builder, loc, targetTy, biasVal, builder.getDenseI64ArrayAttr(dims));
}

static Value reduceAttentionBias(OpBuilder &builder, Location loc, Value val,
                                 ArrayRef<int64_t> biasShape, int64_t W) {
  auto valTy = cast<RankedTensorType>(val.getType());
  Type elTy = valTy.getElementType();
  int64_t offset = (W == 1) ? 0 : 1;
  SmallVector<int64_t> reduceDims;
  for (int64_t d = 0; d < 4; ++d) {
    if (biasShape[d] == 1 && valTy.getDimSize(offset + d) != 1)
      reduceDims.push_back(offset + d);
  }
  Value reduced = reduceSumAlongDims(builder, loc, val, reduceDims);
  SmallVector<int64_t> outShape;
  if (W > 1)
    outShape.push_back(W);
  outShape.append(biasShape.begin(), biasShape.end());
  auto outTy = RankedTensorType::get(outShape, elTy);
  if (reduced.getType() != outTy)
    reduced = stablehlo::ReshapeOp::create(builder, loc, outTy, reduced);
  return reduced;
}

// ([W,] B, T, N, D) x ([W,] B, S, N, D) -> ([W,] B, N, T, S)
static Value attnDotQK(OpBuilder &builder, Location loc, Value lhsBtnd,
                       Value rhsBsnd, int64_t W) {
  auto lhsTy = cast<RankedTensorType>(lhsBtnd.getType());
  auto rhsTy = cast<RankedTensorType>(rhsBsnd.getType());
  Type elTy = lhsTy.getElementType();
  if (W == 1) {
    int64_t B = lhsTy.getDimSize(0);
    int64_t T = lhsTy.getDimSize(1);
    int64_t N = lhsTy.getDimSize(2);
    int64_t S = rhsTy.getDimSize(1);
    auto dims = stablehlo::DotDimensionNumbersAttr::get(
        builder.getContext(), /*lhsBatch=*/{0, 2}, /*rhsBatch=*/{0, 2},
        /*lhsContract=*/{3}, /*rhsContract=*/{3});
    return stablehlo::DotGeneralOp::create(
        builder, loc, RankedTensorType::get({B, N, T, S}, elTy), lhsBtnd,
        rhsBsnd, dims, nullptr, nullptr);
  }
  int64_t B = lhsTy.getDimSize(1);
  int64_t T = lhsTy.getDimSize(2);
  int64_t N = lhsTy.getDimSize(3);
  int64_t S = rhsTy.getDimSize(2);
  auto dims = stablehlo::DotDimensionNumbersAttr::get(
      builder.getContext(), /*lhsBatch=*/{0, 1, 3}, /*rhsBatch=*/{0, 1, 3},
      /*lhsContract=*/{4}, /*rhsContract=*/{4});
  return stablehlo::DotGeneralOp::create(
      builder, loc, RankedTensorType::get({W, B, N, T, S}, elTy), lhsBtnd,
      rhsBsnd, dims, nullptr, nullptr);
}

// ([W,] B, N, T, S) x ([W,] B, S, N, D) -> ([W,] B, T, N, D)
static Value attnDotPV(OpBuilder &builder, Location loc, Value pBnts,
                       Value vBsnd, int64_t W) {
  auto pTy = cast<RankedTensorType>(pBnts.getType());
  auto vTy = cast<RankedTensorType>(vBsnd.getType());
  Type elTy = pTy.getElementType();
  if (W == 1) {
    int64_t B = pTy.getDimSize(0);
    int64_t N = pTy.getDimSize(1);
    int64_t T = pTy.getDimSize(2);
    int64_t D = vTy.getDimSize(3);
    auto dims = stablehlo::DotDimensionNumbersAttr::get(
        builder.getContext(), /*lhsBatch=*/{0, 1}, /*rhsBatch=*/{0, 2},
        /*lhsContract=*/{3}, /*rhsContract=*/{1});
    Value bntd = stablehlo::DotGeneralOp::create(
        builder, loc, RankedTensorType::get({B, N, T, D}, elTy), pBnts, vBsnd,
        dims, nullptr, nullptr);
    return stablehlo::TransposeOp::create(
        builder, loc, bntd, builder.getDenseI64ArrayAttr({0, 2, 1, 3}));
  }
  int64_t B = pTy.getDimSize(1);
  int64_t N = pTy.getDimSize(2);
  int64_t T = pTy.getDimSize(3);
  int64_t D = vTy.getDimSize(4);
  auto dims = stablehlo::DotDimensionNumbersAttr::get(
      builder.getContext(), /*lhsBatch=*/{0, 1, 2}, /*rhsBatch=*/{0, 1, 3},
      /*lhsContract=*/{4}, /*rhsContract=*/{2});
  Value wbntd = stablehlo::DotGeneralOp::create(
      builder, loc, RankedTensorType::get({W, B, N, T, D}, elTy), pBnts, vBsnd,
      dims, nullptr, nullptr);
  return stablehlo::TransposeOp::create(
      builder, loc, wbntd, builder.getDenseI64ArrayAttr({0, 1, 3, 2, 4}));
}

// ([W,] B, N, T, S) x ([W,] B, T, N, D) -> ([W,] B, S, N, D)
static Value attnDotPTDOut(OpBuilder &builder, Location loc, Value pBnts,
                           Value dOutBtnd, int64_t W) {
  auto pTy = cast<RankedTensorType>(pBnts.getType());
  auto dTy = cast<RankedTensorType>(dOutBtnd.getType());
  Type elTy = pTy.getElementType();
  if (W == 1) {
    int64_t B = pTy.getDimSize(0);
    int64_t N = pTy.getDimSize(1);
    int64_t S = pTy.getDimSize(3);
    int64_t D = dTy.getDimSize(3);
    auto dims = stablehlo::DotDimensionNumbersAttr::get(
        builder.getContext(), /*lhsBatch=*/{0, 1}, /*rhsBatch=*/{0, 2},
        /*lhsContract=*/{2}, /*rhsContract=*/{1});
    Value bnsd = stablehlo::DotGeneralOp::create(
        builder, loc, RankedTensorType::get({B, N, S, D}, elTy), pBnts,
        dOutBtnd, dims, nullptr, nullptr);
    return stablehlo::TransposeOp::create(
        builder, loc, bnsd, builder.getDenseI64ArrayAttr({0, 2, 1, 3}));
  }
  int64_t B = pTy.getDimSize(1);
  int64_t N = pTy.getDimSize(2);
  int64_t S = pTy.getDimSize(4);
  int64_t D = dTy.getDimSize(4);
  auto dims = stablehlo::DotDimensionNumbersAttr::get(
      builder.getContext(), /*lhsBatch=*/{0, 1, 2}, /*rhsBatch=*/{0, 1, 3},
      /*lhsContract=*/{3}, /*rhsContract=*/{2});
  Value wbnsd = stablehlo::DotGeneralOp::create(
      builder, loc, RankedTensorType::get({W, B, N, S, D}, elTy), pBnts,
      dOutBtnd, dims, nullptr, nullptr);
  return stablehlo::TransposeOp::create(
      builder, loc, wbnsd, builder.getDenseI64ArrayAttr({0, 1, 3, 2, 4}));
}

struct AttentionRecomputedState {
  Value probs;
  Value qValidBnt;
  Value validAll;
};

static AttentionRecomputedState
recomputeAttentionState(OpBuilder &builder, Location loc, Value qC, Value kC,
                        Value lseC, Value bias, Value mask, Value qLengths,
                        Value kvLengths, double scaleVal, bool isCausal) {
  auto qTy = cast<RankedTensorType>(qC.getType());
  auto kTy = cast<RankedTensorType>(kC.getType());
  Type computeTy = qTy.getElementType();
  int64_t B = qTy.getDimSize(0);
  int64_t T = qTy.getDimSize(1);
  int64_t N_q = qTy.getDimSize(2);
  int64_t S = kTy.getDimSize(1);

  auto bntsTy = RankedTensorType::get({B, N_q, T, S}, computeTy);
  auto bntTy = RankedTensorType::get({B, N_q, T}, computeTy);

  Value scores = attnDotQK(builder, loc, qC, kC, /*W=*/1);
  if (scaleVal != 1.0) {
    Value scaleConst = createSplatConst(builder, loc, bntsTy, scaleVal);
    scores = stablehlo::MulOp::create(builder, loc, scores, scaleConst);
  }
  if (bias) {
    Value biasC = convertElementType(builder, loc, bias, computeTy);
    biasC = broadcastAttentionBias(builder, loc, biasC, bntsTy, /*W=*/1);
    scores = stablehlo::AddOp::create(builder, loc, scores, biasC);
  }

  auto bntsI64Ty = RankedTensorType::get({B, N_q, T, S}, builder.getI64Type());
  auto bntsI1Ty = RankedTensorType::get({B, N_q, T, S}, builder.getI1Type());
  Value validMask;
  auto combineMask = [&](Value m) {
    validMask =
        validMask
            ? stablehlo::AndOp::create(builder, loc, validMask, m).getResult()
            : m;
  };

  if (isCausal) {
    Value tIdx = stablehlo::IotaOp::create(builder, loc, bntsI64Ty,
                                           builder.getI64IntegerAttr(2));
    Value sIdx = stablehlo::IotaOp::create(builder, loc, bntsI64Ty,
                                           builder.getI64IntegerAttr(3));
    Value causal = stablehlo::CompareOp::create(
        builder, loc, tIdx, sIdx, stablehlo::ComparisonDirection::GE);
    combineMask(causal);
  }
  if (mask) {
    Value m = broadcastAttentionBias(builder, loc, mask, bntsI1Ty, /*W=*/1);
    combineMask(m);
  }
  if (kvLengths) {
    Value kvLenI64 =
        convertElementType(builder, loc, kvLengths, builder.getI64Type());
    Value kvLenBcast = stablehlo::BroadcastInDimOp::create(
        builder, loc, bntsI64Ty, kvLenI64, builder.getDenseI64ArrayAttr({0}));
    Value sIdx = stablehlo::IotaOp::create(builder, loc, bntsI64Ty,
                                           builder.getI64IntegerAttr(3));
    Value kvMask = stablehlo::CompareOp::create(
        builder, loc, sIdx, kvLenBcast, stablehlo::ComparisonDirection::LT);
    combineMask(kvMask);
  }

  Value qValidBnt = stablehlo::IsFiniteOp::create(builder, loc, lseC);
  if (qLengths) {
    auto bntI64Ty = RankedTensorType::get({B, N_q, T}, builder.getI64Type());
    Value qLenI64 =
        convertElementType(builder, loc, qLengths, builder.getI64Type());
    Value qLenBcast = stablehlo::BroadcastInDimOp::create(
        builder, loc, bntI64Ty, qLenI64, builder.getDenseI64ArrayAttr({0}));
    Value tIdx = stablehlo::IotaOp::create(builder, loc, bntI64Ty,
                                           builder.getI64IntegerAttr(2));
    Value qMask = stablehlo::CompareOp::create(
        builder, loc, tIdx, qLenBcast, stablehlo::ComparisonDirection::LT);
    qValidBnt = stablehlo::AndOp::create(builder, loc, qValidBnt, qMask);
  }

  Value zeroBnt = createSplatConst(builder, loc, bntTy, 0.0);
  Value safeLse =
      stablehlo::SelectOp::create(builder, loc, qValidBnt, lseC, zeroBnt);
  Value safeLseBcast = stablehlo::BroadcastInDimOp::create(
      builder, loc, bntsTy, safeLse, builder.getDenseI64ArrayAttr({0, 1, 2}));
  Value pRaw = stablehlo::ExpOp::create(
      builder, loc,
      stablehlo::SubtractOp::create(builder, loc, scores, safeLseBcast));

  Value validAll = stablehlo::BroadcastInDimOp::create(
      builder, loc, bntsI1Ty, qValidBnt,
      builder.getDenseI64ArrayAttr({0, 1, 2}));
  if (validMask)
    validAll = stablehlo::AndOp::create(builder, loc, validAll, validMask);

  Value zeroBnts = createSplatConst(builder, loc, bntsTy, 0.0);
  Value probs =
      stablehlo::SelectOp::create(builder, loc, validAll, pRaw, zeroBnts);
  return {probs, qValidBnt, validAll};
}

struct AttentionOpInterfaceRev
    : public ReverseAutoDiffOpInterface::ExternalModel<AttentionOpInterfaceRev,
                                                       AttentionOp> {
  LogicalResult createReverseModeAdjoint(Operation *orig, OpBuilder &builder,
                                         MGradientUtilsReverse *gutils,
                                         SmallVector<Value> caches) const {
    auto op = cast<AttentionOp>(orig);
    if (caches.empty())
      return success();

    Location loc = op.getLoc();
    size_t idx = 0;
    Value q = gutils->popCache(caches[idx++], builder);
    Value k = gutils->popCache(caches[idx++], builder);
    Value v = gutils->popCache(caches[idx++], builder);
    Value bias =
        op.getBias() ? gutils->popCache(caches[idx++], builder) : Value();
    Value mask =
        op.getMask() ? gutils->popCache(caches[idx++], builder) : Value();
    Value qLengths =
        op.getQLengths() ? gutils->popCache(caches[idx++], builder) : Value();
    Value kvLengths =
        op.getKvLengths() ? gutils->popCache(caches[idx++], builder) : Value();
    Value output = gutils->popCache(caches[idx++], builder);
    Value lse = gutils->popCache(caches[idx++], builder);

    bool qActive = !gutils->isConstantValue(op.getQuery());
    bool kActive = !gutils->isConstantValue(op.getKey());
    bool vActive = !gutils->isConstantValue(op.getValue());
    bool biasActive = op.getBias() && !gutils->isConstantValue(op.getBias());
    bool outActive = !gutils->isConstantValue(op.getOutput());
    bool lseActive = !gutils->isConstantValue(op.getLse());
    int64_t W = gutils->width;

    if (outActive) {
      Value dOutput = gutils->diffe(op.getOutput(), builder);
      gutils->zeroDiffe(op.getOutput(), builder);

      if (W == 1) {
        Type dBiasTy = biasActive ? bias.getType() : Type();
        auto gradOp = AttentionGradOp::create(
            builder, loc, q.getType(), k.getType(), v.getType(), dBiasTy, q, k,
            v, output, lse, dOutput, bias, mask, qLengths, kvLengths,
            op.getScaleAttr(), op.getIsCausalAttr());

        if (qActive)
          gutils->addToDiffe(op.getQuery(), gradOp.getDQuery(), builder);
        if (kActive)
          gutils->addToDiffe(op.getKey(), gradOp.getDKey(), builder);
        if (vActive)
          gutils->addToDiffe(op.getValue(), gradOp.getDValue(), builder);
        if (biasActive)
          gutils->addToDiffe(op.getBias(), gradOp.getDBias(), builder);
      } else {
        auto qTy = cast<RankedTensorType>(q.getType());
        int64_t B = qTy.getDimSize(0);
        auto bcastAndFlatten = [&](Value val) -> Value {
          if (!val)
            return Value();
          Value bcast = prependAttentionWidth(builder, loc, val, W);
          return flattenBatchOperand(builder, loc, bcast, {W}, B);
        };

        Value qFlat = bcastAndFlatten(q);
        Value kFlat = bcastAndFlatten(k);
        Value vFlat = bcastAndFlatten(v);
        Value outFlat = bcastAndFlatten(output);
        Value lseFlat = bcastAndFlatten(lse);
        Value dOutFlat = flattenBatchOperand(builder, loc, dOutput, {W}, B);
        Value biasFlat = bcastAndFlatten(bias);
        Value maskFlat = bcastAndFlatten(mask);
        Value qLenFlat = bcastAndFlatten(qLengths);
        Value kvLenFlat = bcastAndFlatten(kvLengths);

        Type dBiasFlatTy = biasActive ? biasFlat.getType() : Type();
        auto gradOp = AttentionGradOp::create(
            builder, loc, qFlat.getType(), kFlat.getType(), vFlat.getType(),
            dBiasFlatTy, qFlat, kFlat, vFlat, outFlat, lseFlat, dOutFlat,
            biasFlat, maskFlat, qLenFlat, kvLenFlat, op.getScaleAttr(),
            op.getIsCausalAttr());

        if (qActive) {
          Value dq = unflattenBatchResult(builder, loc, gradOp.getDQuery(), {W},
                                          B, qTy.getShape());
          gutils->addToDiffe(op.getQuery(), dq, builder);
        }
        if (kActive) {
          auto kTy = cast<RankedTensorType>(k.getType());
          Value dk = unflattenBatchResult(builder, loc, gradOp.getDKey(), {W},
                                          B, kTy.getShape());
          gutils->addToDiffe(op.getKey(), dk, builder);
        }
        if (vActive) {
          auto vTy = cast<RankedTensorType>(v.getType());
          Value dv = unflattenBatchResult(builder, loc, gradOp.getDValue(), {W},
                                          B, vTy.getShape());
          gutils->addToDiffe(op.getValue(), dv, builder);
        }
        if (biasActive) {
          auto biasTy = cast<RankedTensorType>(bias.getType());
          Value db = unflattenBatchResult(builder, loc, gradOp.getDBias(), {W},
                                          B, biasTy.getShape(),
                                          /*sumOverSharedBatch=*/true);
          gutils->addToDiffe(op.getBias(), db, builder);
        }
      }
    }

    if (lseActive) {
      Value dLseRaw = gutils->diffe(op.getLse(), builder);
      gutils->zeroDiffe(op.getLse(), builder);

      if (qActive || kActive || biasActive) {
        auto qTy = cast<RankedTensorType>(q.getType());
        auto kTy = cast<RankedTensorType>(k.getType());
        int64_t N_q = qTy.getDimSize(2);
        int64_t N_kv = kTy.getDimSize(2);
        Type elTy = qTy.getElementType();
        Type computeTy =
            (elTy.isF16() || elTy.isBF16()) ? (Type)builder.getF32Type() : elTy;
        double scaleVal = op.getScale().convertToDouble();

        Value qC = convertElementType(builder, loc, q, computeTy);
        Value kC = expandAttentionKVHeads(
            builder, loc, convertElementType(builder, loc, k, computeTy), N_q,
            /*W=*/1);
        Value lseC = convertElementType(builder, loc, lse, computeTy);

        auto state = recomputeAttentionState(builder, loc, qC, kC, lseC, bias,
                                             mask, qLengths, kvLengths,
                                             scaleVal, op.getIsCausal());
        Value PW = prependAttentionWidth(builder, loc, state.probs, W);
        Value qValidBntW =
            prependAttentionWidth(builder, loc, state.qValidBnt, W);

        Value dLseC = convertElementType(builder, loc, dLseRaw, computeTy);
        Value zeroLseW = createSplatConst(
            builder, loc, cast<RankedTensorType>(dLseC.getType()), 0.0);
        Value safeDLse = stablehlo::SelectOp::create(builder, loc, qValidBntW,
                                                     dLseC, zeroLseW);

        auto scoreWTy = cast<RankedTensorType>(PW.getType());
        SmallVector<int64_t> lseBcastDims =
            (W == 1) ? SmallVector<int64_t>{0, 1, 2}
                     : SmallVector<int64_t>{0, 1, 2, 3};
        Value safeDLseBcast = stablehlo::BroadcastInDimOp::create(
            builder, loc, scoreWTy, safeDLse,
            builder.getDenseI64ArrayAttr(lseBcastDims));
        Value dS = stablehlo::MulOp::create(builder, loc, PW, safeDLseBcast);

        if (biasActive) {
          auto biasTy = cast<RankedTensorType>(bias.getType());
          Value dbC =
              reduceAttentionBias(builder, loc, dS, biasTy.getShape(), W);
          gutils->addToDiffe(op.getBias(),
                             convertElementType(builder, loc, dbC, elTy),
                             builder);
        }

        Value dSScaled = dS;
        if (scaleVal != 1.0) {
          Value sConst = createSplatConst(builder, loc, scoreWTy, scaleVal);
          dSScaled = stablehlo::MulOp::create(builder, loc, dS, sConst);
        }

        if (qActive) {
          Value kCW = prependAttentionWidth(builder, loc, kC, W);
          Value dqC = attnDotPV(builder, loc, dSScaled, kCW, W);
          gutils->addToDiffe(op.getQuery(),
                             convertElementType(builder, loc, dqC, elTy),
                             builder);
        }
        if (kActive) {
          Value qCW = prependAttentionWidth(builder, loc, qC, W);
          Value dkExp = attnDotPTDOut(builder, loc, dSScaled, qCW, W);
          Value dkC = reduceAttentionKVHeads(builder, loc, dkExp, N_kv, W);
          gutils->addToDiffe(op.getKey(),
                             convertElementType(builder, loc, dkC, elTy),
                             builder);
        }
      }
    }

    return success();
  }

  SmallVector<Value> cacheValues(Operation *orig,
                                 MGradientUtilsReverse *gutils) const {
    auto op = cast<AttentionOp>(orig);
    bool anyInputActive =
        !gutils->isConstantValue(op.getQuery()) ||
        !gutils->isConstantValue(op.getKey()) ||
        !gutils->isConstantValue(op.getValue()) ||
        (op.getBias() && !gutils->isConstantValue(op.getBias()));
    bool anyOutputActive = !gutils->isConstantValue(op.getOutput()) ||
                           !gutils->isConstantValue(op.getLse());
    if (!anyInputActive || !anyOutputActive)
      return {};

    SmallVector<Value> caches;
    Operation *newOp = gutils->getNewFromOriginal(op);
    OpBuilder cacheBuilder(newOp);

    caches.push_back(gutils->initAndPushCache(
        gutils->getNewFromOriginal(op.getQuery()), cacheBuilder));
    caches.push_back(gutils->initAndPushCache(
        gutils->getNewFromOriginal(op.getKey()), cacheBuilder));
    caches.push_back(gutils->initAndPushCache(
        gutils->getNewFromOriginal(op.getValue()), cacheBuilder));
    if (op.getBias())
      caches.push_back(gutils->initAndPushCache(
          gutils->getNewFromOriginal(op.getBias()), cacheBuilder));
    if (op.getMask())
      caches.push_back(gutils->initAndPushCache(
          gutils->getNewFromOriginal(op.getMask()), cacheBuilder));
    if (op.getQLengths())
      caches.push_back(gutils->initAndPushCache(
          gutils->getNewFromOriginal(op.getQLengths()), cacheBuilder));
    if (op.getKvLengths())
      caches.push_back(gutils->initAndPushCache(
          gutils->getNewFromOriginal(op.getKvLengths()), cacheBuilder));

    cacheBuilder.setInsertionPointAfter(newOp);
    caches.push_back(gutils->initAndPushCache(
        gutils->getNewFromOriginal(op.getOutput()), cacheBuilder));
    caches.push_back(gutils->initAndPushCache(
        gutils->getNewFromOriginal(op.getLse()), cacheBuilder));

    return caches;
  }

  LogicalResult createShadowValues(Operation *op, OpBuilder &builder,
                                   MGradientUtilsReverse *gutils) const {
    return success();
  }
};

struct AttentionOpInterfaceFwd
    : public AutoDiffOpInterface::ExternalModel<AttentionOpInterfaceFwd,
                                                AttentionOp> {
  LogicalResult createForwardModeTangent(Operation *orig, OpBuilder &builder,
                                         MGradientUtils *gutils) const {
    auto op = cast<AttentionOp>(orig);
    if (gutils->isConstantValue(op.getOutput()) &&
        gutils->isConstantValue(op.getLse())) {
      gutils->eraseIfUnused(orig);
      return success();
    }

    auto newOp = cast<AttentionOp>(gutils->getNewFromOriginal(orig));
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointAfter(newOp);
    Location loc = op.getLoc();

    Value q = newOp.getQuery();
    Value k = newOp.getKey();
    Value v = newOp.getValue();
    Value lse = newOp.getLse();

    auto qTy = cast<RankedTensorType>(q.getType());
    auto lseTy = cast<RankedTensorType>(lse.getType());

    int64_t N_q = qTy.getDimSize(2);
    int64_t W = gutils->width;

    Type elTy = qTy.getElementType();
    Type computeTy =
        (elTy.isF16() || elTy.isBF16()) ? (Type)builder.getF32Type() : elTy;
    double scaleVal = op.getScale().convertToDouble();

    Value qC = convertElementType(builder, loc, q, computeTy);
    Value kC = expandAttentionKVHeads(
        builder, loc, convertElementType(builder, loc, k, computeTy), N_q,
        /*W=*/1);
    Value vC = expandAttentionKVHeads(
        builder, loc, convertElementType(builder, loc, v, computeTy), N_q,
        /*W=*/1);
    Value lseC = convertElementType(builder, loc, lse, computeTy);

    auto state = recomputeAttentionState(
        builder, loc, qC, kC, lseC, newOp.getBias(), newOp.getMask(),
        newOp.getQLengths(), newOp.getKvLengths(), scaleVal, op.getIsCausal());

    Value qCW = prependAttentionWidth(builder, loc, qC, W);
    Value kCW = prependAttentionWidth(builder, loc, kC, W);
    Value vCW = prependAttentionWidth(builder, loc, vC, W);
    Value PW = prependAttentionWidth(builder, loc, state.probs, W);
    Value validAllW = prependAttentionWidth(builder, loc, state.validAll, W);

    auto dSTy = cast<RankedTensorType>(PW.getType());
    Value zeroDS = createSplatConst(builder, loc, dSTy, 0.0);
    Value dS = zeroDS;

    if (!gutils->isConstantValue(op.getQuery())) {
      Value dq = convertElementType(
          builder, loc, gutils->invertPointerM(op.getQuery(), builder),
          computeTy);
      Value dqK = attnDotQK(builder, loc, dq, kCW, W);
      if (scaleVal != 1.0) {
        Value sConst = createSplatConst(builder, loc, dSTy, scaleVal);
        dqK = stablehlo::MulOp::create(builder, loc, dqK, sConst);
      }
      dS = stablehlo::AddOp::create(builder, loc, dS, dqK);
    }

    if (!gutils->isConstantValue(op.getKey())) {
      Value dkRaw = convertElementType(
          builder, loc, gutils->invertPointerM(op.getKey(), builder),
          computeTy);
      Value dk = expandAttentionKVHeads(builder, loc, dkRaw, N_q, W);
      Value qDk = attnDotQK(builder, loc, qCW, dk, W);
      if (scaleVal != 1.0) {
        Value sConst = createSplatConst(builder, loc, dSTy, scaleVal);
        qDk = stablehlo::MulOp::create(builder, loc, qDk, sConst);
      }
      dS = stablehlo::AddOp::create(builder, loc, dS, qDk);
    }

    if (op.getBias() && !gutils->isConstantValue(op.getBias())) {
      Value db = convertElementType(
          builder, loc, gutils->invertPointerM(op.getBias(), builder),
          computeTy);
      db = broadcastAttentionBias(builder, loc, db, dSTy, W);
      dS = stablehlo::AddOp::create(builder, loc, dS, db);
    }

    dS = stablehlo::SelectOp::create(builder, loc, validAllW, dS, zeroDS);

    int64_t sDim = (W == 1) ? 3 : 4;
    Value pdS = stablehlo::MulOp::create(builder, loc, PW, dS);
    Value dLse = reduceSumAlongDims(builder, loc, pdS, {sDim});

    SmallVector<int64_t> lseBcastDims = (W == 1)
                                            ? SmallVector<int64_t>{0, 1, 2}
                                            : SmallVector<int64_t>{0, 1, 2, 3};
    Value dLseBcast = stablehlo::BroadcastInDimOp::create(
        builder, loc, dSTy, dLse, builder.getDenseI64ArrayAttr(lseBcastDims));
    Value dP = stablehlo::MulOp::create(
        builder, loc, PW,
        stablehlo::SubtractOp::create(builder, loc, dS, dLseBcast));

    if (!gutils->isConstantValue(op.getOutput())) {
      Value dOut = attnDotPV(builder, loc, dP, vCW, W);
      if (!gutils->isConstantValue(op.getValue())) {
        Value dvRaw = convertElementType(
            builder, loc, gutils->invertPointerM(op.getValue(), builder),
            computeTy);
        Value dv = expandAttentionKVHeads(builder, loc, dvRaw, N_q, W);
        Value pDv = attnDotPV(builder, loc, PW, dv, W);
        dOut = stablehlo::AddOp::create(builder, loc, dOut, pDv);
      }
      Value dOutFinal = convertElementType(builder, loc, dOut, elTy);
      gutils->setDiffe(op.getOutput(), dOutFinal, builder);
    }
    if (!gutils->isConstantValue(op.getLse())) {
      Value dLseFinal =
          convertElementType(builder, loc, dLse, lseTy.getElementType());
      gutils->setDiffe(op.getLse(), dLseFinal, builder);
    }

    gutils->eraseIfUnused(orig);
    return success();
  }
};

struct AttentionGradOpInterfaceFwd
    : public AutoDiffOpInterface::ExternalModel<AttentionGradOpInterfaceFwd,
                                                AttentionGradOp> {
  LogicalResult createForwardModeTangent(Operation *orig, OpBuilder &builder,
                                         MGradientUtils *gutils) const {
    auto op = cast<AttentionGradOp>(orig);
    bool dqResActive = !gutils->isConstantValue(op.getDQuery());
    bool dkResActive = !gutils->isConstantValue(op.getDKey());
    bool dvResActive = !gutils->isConstantValue(op.getDValue());
    bool dbResActive = op.getDBias() && !gutils->isConstantValue(op.getDBias());
    if (!dqResActive && !dkResActive && !dvResActive && !dbResActive) {
      gutils->eraseIfUnused(orig);
      return success();
    }

    auto newOp = cast<AttentionGradOp>(gutils->getNewFromOriginal(orig));
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointAfter(newOp);
    Location loc = op.getLoc();

    Value q = newOp.getQuery();
    Value k = newOp.getKey();
    Value v = newOp.getValue();
    Value output = newOp.getOutput();
    Value lse = newOp.getLse();
    Value dOutput = newOp.getDOutput();

    auto qTy = cast<RankedTensorType>(q.getType());
    auto kTy = cast<RankedTensorType>(k.getType());
    auto vTy = cast<RankedTensorType>(v.getType());

    int64_t B = qTy.getDimSize(0);
    int64_t T = qTy.getDimSize(1);
    int64_t N_q = qTy.getDimSize(2);
    int64_t N_kv = kTy.getDimSize(2);
    int64_t D_v = vTy.getDimSize(3);
    int64_t W = gutils->width;

    Type elTy = qTy.getElementType();
    Type computeTy =
        (elTy.isF16() || elTy.isBF16()) ? (Type)builder.getF32Type() : elTy;
    double scaleVal = op.getScale().convertToDouble();

    Value qC = convertElementType(builder, loc, q, computeTy);
    Value kC = expandAttentionKVHeads(
        builder, loc, convertElementType(builder, loc, k, computeTy), N_q,
        /*W=*/1);
    Value vC = expandAttentionKVHeads(
        builder, loc, convertElementType(builder, loc, v, computeTy), N_q,
        /*W=*/1);
    Value outC = convertElementType(builder, loc, output, computeTy);
    Value dOutC = convertElementType(builder, loc, dOutput, computeTy);
    Value lseC = convertElementType(builder, loc, lse, computeTy);

    auto state = recomputeAttentionState(
        builder, loc, qC, kC, lseC, newOp.getBias(), newOp.getMask(),
        newOp.getQLengths(), newOp.getKvLengths(), scaleVal, op.getIsCausal());

    Value qValidBtn = stablehlo::TransposeOp::create(
        builder, loc, state.qValidBnt, builder.getDenseI64ArrayAttr({0, 2, 1}));
    auto btndI1Ty =
        RankedTensorType::get({B, T, N_q, D_v}, builder.getI1Type());
    Value qValidBtnd = stablehlo::BroadcastInDimOp::create(
        builder, loc, btndI1Ty, qValidBtn,
        builder.getDenseI64ArrayAttr({0, 1, 2}));
    auto btndTy = RankedTensorType::get({B, T, N_q, D_v}, computeTy);
    Value zeroBtnd = createSplatConst(builder, loc, btndTy, 0.0);
    Value safeOut =
        stablehlo::SelectOp::create(builder, loc, qValidBtnd, outC, zeroBtnd);
    Value safeDOut =
        stablehlo::SelectOp::create(builder, loc, qValidBtnd, dOutC, zeroBtnd);

    auto bntsTy = cast<RankedTensorType>(state.probs.getType());
    Value dP = attnDotQK(builder, loc, safeDOut, vC, /*W=*/1);
    Value deltaBtn = reduceSumAlongDims(
        builder, loc, stablehlo::MulOp::create(builder, loc, safeDOut, safeOut),
        {3});
    Value deltaBnt = stablehlo::TransposeOp::create(
        builder, loc, deltaBtn, builder.getDenseI64ArrayAttr({0, 2, 1}));
    Value deltaBcast = stablehlo::BroadcastInDimOp::create(
        builder, loc, bntsTy, deltaBnt,
        builder.getDenseI64ArrayAttr({0, 1, 2}));
    Value dPMul = stablehlo::SubtractOp::create(builder, loc, dP, deltaBcast);
    Value dS = stablehlo::MulOp::create(builder, loc, state.probs, dPMul);
    Value dSScaled = dS;
    if (scaleVal != 1.0) {
      Value sConst = createSplatConst(builder, loc, bntsTy, scaleVal);
      dSScaled = stablehlo::MulOp::create(builder, loc, dS, sConst);
    }

    Value qCW = prependAttentionWidth(builder, loc, qC, W);
    Value kCW = prependAttentionWidth(builder, loc, kC, W);
    Value vCW = prependAttentionWidth(builder, loc, vC, W);
    Value safeOutW = prependAttentionWidth(builder, loc, safeOut, W);
    Value safeDOutW = prependAttentionWidth(builder, loc, safeDOut, W);
    Value PW = prependAttentionWidth(builder, loc, state.probs, W);
    Value dPMulW = prependAttentionWidth(builder, loc, dPMul, W);
    Value dSScaledW = prependAttentionWidth(builder, loc, dSScaled, W);
    Value qValidBntW = prependAttentionWidth(builder, loc, state.qValidBnt, W);
    Value qValidBtndW = prependAttentionWidth(builder, loc, qValidBtnd, W);
    Value validAllW = prependAttentionWidth(builder, loc, state.validAll, W);

    auto scoreWTy = cast<RankedTensorType>(PW.getType());
    auto btndWTy = cast<RankedTensorType>(safeOutW.getType());
    Value zeroScoreW = createSplatConst(builder, loc, scoreWTy, 0.0);
    Value zeroBtndW = createSplatConst(builder, loc, btndWTy, 0.0);

    bool qActive = !gutils->isConstantValue(op.getQuery());
    bool kActive = !gutils->isConstantValue(op.getKey());
    bool vActive = !gutils->isConstantValue(op.getValue());
    bool outActive = !gutils->isConstantValue(op.getOutput());
    bool lseActive = !gutils->isConstantValue(op.getLse());
    bool dOutActive = !gutils->isConstantValue(op.getDOutput());
    bool biasActive = op.getBias() && !gutils->isConstantValue(op.getBias());

    Value dotQ;
    if (qActive) {
      dotQ = convertElementType(builder, loc,
                                gutils->invertPointerM(op.getQuery(), builder),
                                computeTy);
    }
    Value dotK;
    if (kActive) {
      Value dkRaw = convertElementType(
          builder, loc, gutils->invertPointerM(op.getKey(), builder),
          computeTy);
      dotK = expandAttentionKVHeads(builder, loc, dkRaw, N_q, W);
    }
    Value dotV;
    if (vActive) {
      Value dvRaw = convertElementType(
          builder, loc, gutils->invertPointerM(op.getValue(), builder),
          computeTy);
      dotV = expandAttentionKVHeads(builder, loc, dvRaw, N_q, W);
    }

    Value dotS = zeroScoreW;
    if (qActive) {
      Value term = attnDotQK(builder, loc, dotQ, kCW, W);
      if (scaleVal != 1.0) {
        Value sConst = createSplatConst(builder, loc, scoreWTy, scaleVal);
        term = stablehlo::MulOp::create(builder, loc, term, sConst);
      }
      dotS = stablehlo::AddOp::create(builder, loc, dotS, term);
    }
    if (kActive) {
      Value term = attnDotQK(builder, loc, qCW, dotK, W);
      if (scaleVal != 1.0) {
        Value sConst = createSplatConst(builder, loc, scoreWTy, scaleVal);
        term = stablehlo::MulOp::create(builder, loc, term, sConst);
      }
      dotS = stablehlo::AddOp::create(builder, loc, dotS, term);
    }
    if (biasActive) {
      Value dotBias = convertElementType(
          builder, loc, gutils->invertPointerM(op.getBias(), builder),
          computeTy);
      dotBias = broadcastAttentionBias(builder, loc, dotBias, scoreWTy, W);
      dotS = stablehlo::AddOp::create(builder, loc, dotS, dotBias);
    }

    Value dotZ = dotS;
    if (lseActive) {
      Value dotLseRaw = convertElementType(
          builder, loc, gutils->invertPointerM(op.getLse(), builder),
          computeTy);
      Value zeroLseW = createSplatConst(
          builder, loc, cast<RankedTensorType>(dotLseRaw.getType()), 0.0);
      Value dotLseSafe = stablehlo::SelectOp::create(builder, loc, qValidBntW,
                                                     dotLseRaw, zeroLseW);
      SmallVector<int64_t> lseBcastDims =
          (W == 1) ? SmallVector<int64_t>{0, 1, 2}
                   : SmallVector<int64_t>{0, 1, 2, 3};
      Value dotLseBcast = stablehlo::BroadcastInDimOp::create(
          builder, loc, scoreWTy, dotLseSafe,
          builder.getDenseI64ArrayAttr(lseBcastDims));
      dotZ = stablehlo::SubtractOp::create(builder, loc, dotZ, dotLseBcast);
    }
    dotZ =
        stablehlo::SelectOp::create(builder, loc, validAllW, dotZ, zeroScoreW);
    Value dotP = stablehlo::MulOp::create(builder, loc, PW, dotZ);

    Value dotSafeOut = zeroBtndW;
    if (outActive) {
      Value dotOutRaw = convertElementType(
          builder, loc, gutils->invertPointerM(op.getOutput(), builder),
          computeTy);
      dotSafeOut = stablehlo::SelectOp::create(builder, loc, qValidBtndW,
                                               dotOutRaw, zeroBtndW);
    }
    Value dotSafeDOut = zeroBtndW;
    if (dOutActive) {
      Value dotDOutRaw = convertElementType(
          builder, loc, gutils->invertPointerM(op.getDOutput(), builder),
          computeTy);
      dotSafeDOut = stablehlo::SelectOp::create(builder, loc, qValidBtndW,
                                                dotDOutRaw, zeroBtndW);
    }

    if (dvResActive) {
      Value dotDVExp = attnDotPTDOut(builder, loc, dotP, safeDOutW, W);
      if (dOutActive) {
        Value term = attnDotPTDOut(builder, loc, PW, dotSafeDOut, W);
        dotDVExp = stablehlo::AddOp::create(builder, loc, dotDVExp, term);
      }
      Value dotDV = reduceAttentionKVHeads(builder, loc, dotDVExp, N_kv, W);
      gutils->setDiffe(op.getDValue(),
                       convertElementType(builder, loc, dotDV, elTy), builder);
    }

    Value dotDP = zeroScoreW;
    if (dOutActive) {
      dotDP = stablehlo::AddOp::create(
          builder, loc, dotDP, attnDotQK(builder, loc, dotSafeDOut, vCW, W));
    }
    if (vActive) {
      dotDP = stablehlo::AddOp::create(
          builder, loc, dotDP, attnDotQK(builder, loc, safeDOutW, dotV, W));
    }

    Value dotDeltaProd = zeroBtndW;
    if (dOutActive) {
      dotDeltaProd = stablehlo::AddOp::create(
          builder, loc, dotDeltaProd,
          stablehlo::MulOp::create(builder, loc, dotSafeDOut, safeOutW));
    }
    if (outActive) {
      dotDeltaProd = stablehlo::AddOp::create(
          builder, loc, dotDeltaProd,
          stablehlo::MulOp::create(builder, loc, safeDOutW, dotSafeOut));
    }
    int64_t dDim = (W == 1) ? 3 : 4;
    Value dotDeltaBtn = reduceSumAlongDims(builder, loc, dotDeltaProd, {dDim});
    SmallVector<int64_t> btnToBntPerm = (W == 1)
                                            ? SmallVector<int64_t>{0, 2, 1}
                                            : SmallVector<int64_t>{0, 1, 3, 2};
    Value dotDeltaBnt = stablehlo::TransposeOp::create(
        builder, loc, dotDeltaBtn, builder.getDenseI64ArrayAttr(btnToBntPerm));
    SmallVector<int64_t> lseBcastDims = (W == 1)
                                            ? SmallVector<int64_t>{0, 1, 2}
                                            : SmallVector<int64_t>{0, 1, 2, 3};
    Value dotDeltaBcast = stablehlo::BroadcastInDimOp::create(
        builder, loc, scoreWTy, dotDeltaBnt,
        builder.getDenseI64ArrayAttr(lseBcastDims));

    Value dotDS = stablehlo::AddOp::create(
        builder, loc, stablehlo::MulOp::create(builder, loc, dotP, dPMulW),
        stablehlo::MulOp::create(
            builder, loc, PW,
            stablehlo::SubtractOp::create(builder, loc, dotDP, dotDeltaBcast)));

    if (dbResActive) {
      auto biasTy = cast<RankedTensorType>(op.getDBias().getType());
      Value dotDB =
          reduceAttentionBias(builder, loc, dotDS, biasTy.getShape(), W);
      gutils->setDiffe(op.getDBias(),
                       convertElementType(builder, loc, dotDB, elTy), builder);
    }

    Value dotDSScaled = dotDS;
    if (scaleVal != 1.0) {
      Value sConst = createSplatConst(builder, loc, scoreWTy, scaleVal);
      dotDSScaled = stablehlo::MulOp::create(builder, loc, dotDS, sConst);
    }

    if (dqResActive) {
      Value dotDQ = attnDotPV(builder, loc, dotDSScaled, kCW, W);
      if (kActive) {
        Value term = attnDotPV(builder, loc, dSScaledW, dotK, W);
        dotDQ = stablehlo::AddOp::create(builder, loc, dotDQ, term);
      }
      gutils->setDiffe(op.getDQuery(),
                       convertElementType(builder, loc, dotDQ, elTy), builder);
    }

    if (dkResActive) {
      Value dotDKExp = attnDotPTDOut(builder, loc, dotDSScaled, qCW, W);
      if (qActive) {
        Value term = attnDotPTDOut(builder, loc, dSScaledW, dotQ, W);
        dotDKExp = stablehlo::AddOp::create(builder, loc, dotDKExp, term);
      }
      Value dotDK = reduceAttentionKVHeads(builder, loc, dotDKExp, N_kv, W);
      gutils->setDiffe(op.getDKey(),
                       convertElementType(builder, loc, dotDK, elTy), builder);
    }

    gutils->eraseIfUnused(orig);
    return success();
  }
};

struct AttentionGradOpInterfaceRev
    : public ReverseAutoDiffOpInterface::ExternalModel<
          AttentionGradOpInterfaceRev, AttentionGradOp> {
  LogicalResult createReverseModeAdjoint(Operation *orig, OpBuilder &builder,
                                         MGradientUtilsReverse *gutils,
                                         SmallVector<Value> caches) const {
    auto op = cast<AttentionGradOp>(orig);
    if (caches.empty())
      return success();

    Location loc = op.getLoc();
    size_t idx = 0;
    Value q = gutils->popCache(caches[idx++], builder);
    Value k = gutils->popCache(caches[idx++], builder);
    Value v = gutils->popCache(caches[idx++], builder);
    Value output = gutils->popCache(caches[idx++], builder);
    Value lse = gutils->popCache(caches[idx++], builder);
    Value dOutput = gutils->popCache(caches[idx++], builder);
    Value bias =
        op.getBias() ? gutils->popCache(caches[idx++], builder) : Value();
    Value mask =
        op.getMask() ? gutils->popCache(caches[idx++], builder) : Value();
    Value qLengths =
        op.getQLengths() ? gutils->popCache(caches[idx++], builder) : Value();
    Value kvLengths =
        op.getKvLengths() ? gutils->popCache(caches[idx++], builder) : Value();

    auto qTy = cast<RankedTensorType>(q.getType());
    auto kTy = cast<RankedTensorType>(k.getType());
    auto vTy = cast<RankedTensorType>(v.getType());
    auto lseTy = cast<RankedTensorType>(lse.getType());

    int64_t B = qTy.getDimSize(0);
    int64_t T = qTy.getDimSize(1);
    int64_t N_q = qTy.getDimSize(2);
    int64_t D_qk = qTy.getDimSize(3);
    int64_t S = kTy.getDimSize(1);
    int64_t N_kv = kTy.getDimSize(2);
    int64_t D_v = vTy.getDimSize(3);
    int64_t W = gutils->width;

    Type elTy = qTy.getElementType();
    Type computeTy =
        (elTy.isF16() || elTy.isBF16()) ? (Type)builder.getF32Type() : elTy;
    double scaleVal = op.getScale().convertToDouble();

    bool dqResActive = !gutils->isConstantValue(op.getDQuery());
    bool dkResActive = !gutils->isConstantValue(op.getDKey());
    bool dvResActive = !gutils->isConstantValue(op.getDValue());
    bool dbResActive = op.getDBias() && !gutils->isConstantValue(op.getDBias());

    Value dQBar;
    if (dqResActive) {
      dQBar = convertElementType(
          builder, loc, gutils->diffe(op.getDQuery(), builder), computeTy);
      gutils->zeroDiffe(op.getDQuery(), builder);
    }
    Value dKBarExp;
    if (dkResActive) {
      Value dKBarRaw = convertElementType(
          builder, loc, gutils->diffe(op.getDKey(), builder), computeTy);
      gutils->zeroDiffe(op.getDKey(), builder);
      dKBarExp = expandAttentionKVHeads(builder, loc, dKBarRaw, N_q, W);
    }
    Value dVBarExp;
    if (dvResActive) {
      Value dVBarRaw = convertElementType(
          builder, loc, gutils->diffe(op.getDValue(), builder), computeTy);
      gutils->zeroDiffe(op.getDValue(), builder);
      dVBarExp = expandAttentionKVHeads(builder, loc, dVBarRaw, N_q, W);
    }
    Value dBBar;
    if (dbResActive) {
      dBBar = convertElementType(
          builder, loc, gutils->diffe(op.getDBias(), builder), computeTy);
      gutils->zeroDiffe(op.getDBias(), builder);
    }

    Value qC = convertElementType(builder, loc, q, computeTy);
    Value kC = expandAttentionKVHeads(
        builder, loc, convertElementType(builder, loc, k, computeTy), N_q,
        /*W=*/1);
    Value vC = expandAttentionKVHeads(
        builder, loc, convertElementType(builder, loc, v, computeTy), N_q,
        /*W=*/1);
    Value outC = convertElementType(builder, loc, output, computeTy);
    Value dOutC = convertElementType(builder, loc, dOutput, computeTy);
    Value lseC = convertElementType(builder, loc, lse, computeTy);

    auto state = recomputeAttentionState(builder, loc, qC, kC, lseC, bias, mask,
                                         qLengths, kvLengths, scaleVal,
                                         op.getIsCausal());

    Value qValidBtn = stablehlo::TransposeOp::create(
        builder, loc, state.qValidBnt, builder.getDenseI64ArrayAttr({0, 2, 1}));
    auto btndI1Ty =
        RankedTensorType::get({B, T, N_q, D_v}, builder.getI1Type());
    Value qValidBtnd = stablehlo::BroadcastInDimOp::create(
        builder, loc, btndI1Ty, qValidBtn,
        builder.getDenseI64ArrayAttr({0, 1, 2}));
    auto btndTy = RankedTensorType::get({B, T, N_q, D_v}, computeTy);
    Value zeroBtnd = createSplatConst(builder, loc, btndTy, 0.0);
    Value safeOut =
        stablehlo::SelectOp::create(builder, loc, qValidBtnd, outC, zeroBtnd);
    Value safeDOut =
        stablehlo::SelectOp::create(builder, loc, qValidBtnd, dOutC, zeroBtnd);

    auto bntsTy = cast<RankedTensorType>(state.probs.getType());
    Value dP = attnDotQK(builder, loc, safeDOut, vC, /*W=*/1);
    Value deltaBtn = reduceSumAlongDims(
        builder, loc, stablehlo::MulOp::create(builder, loc, safeDOut, safeOut),
        {3});
    Value deltaBnt = stablehlo::TransposeOp::create(
        builder, loc, deltaBtn, builder.getDenseI64ArrayAttr({0, 2, 1}));
    Value deltaBcast = stablehlo::BroadcastInDimOp::create(
        builder, loc, bntsTy, deltaBnt,
        builder.getDenseI64ArrayAttr({0, 1, 2}));
    Value dPMul = stablehlo::SubtractOp::create(builder, loc, dP, deltaBcast);
    Value dS = stablehlo::MulOp::create(builder, loc, state.probs, dPMul);
    Value dSScaled = dS;
    if (scaleVal != 1.0) {
      Value sConst = createSplatConst(builder, loc, bntsTy, scaleVal);
      dSScaled = stablehlo::MulOp::create(builder, loc, dS, sConst);
    }

    Value qCW = prependAttentionWidth(builder, loc, qC, W);
    Value kCW = prependAttentionWidth(builder, loc, kC, W);
    Value vCW = prependAttentionWidth(builder, loc, vC, W);
    Value safeOutW = prependAttentionWidth(builder, loc, safeOut, W);
    Value safeDOutW = prependAttentionWidth(builder, loc, safeDOut, W);
    Value PW = prependAttentionWidth(builder, loc, state.probs, W);
    Value dPMulW = prependAttentionWidth(builder, loc, dPMul, W);
    Value dSScaledW = prependAttentionWidth(builder, loc, dSScaled, W);
    Value qValidBntW = prependAttentionWidth(builder, loc, state.qValidBnt, W);
    Value qValidBtndW = prependAttentionWidth(builder, loc, qValidBtnd, W);
    Value validAllW = prependAttentionWidth(builder, loc, state.validAll, W);

    auto scoreWTy = cast<RankedTensorType>(PW.getType());
    auto btndWTy = cast<RankedTensorType>(safeOutW.getType());
    SmallVector<int64_t> qWShape =
        (W == 1) ? SmallVector<int64_t>{B, T, N_q, D_qk}
                 : SmallVector<int64_t>{W, B, T, N_q, D_qk};
    SmallVector<int64_t> kExpWShape =
        (W == 1) ? SmallVector<int64_t>{B, S, N_q, D_qk}
                 : SmallVector<int64_t>{W, B, S, N_q, D_qk};

    Value zeroScoreW = createSplatConst(builder, loc, scoreWTy, 0.0);
    Value zeroBtndW = createSplatConst(builder, loc, btndWTy, 0.0);
    Value qBar = createSplatConst(
        builder, loc, RankedTensorType::get(qWShape, computeTy), 0.0);
    Value kBarExp = createSplatConst(
        builder, loc, RankedTensorType::get(kExpWShape, computeTy), 0.0);

    Value dSScaledBar = zeroScoreW;
    if (dqResActive) {
      dSScaledBar = stablehlo::AddOp::create(
          builder, loc, dSScaledBar, attnDotQK(builder, loc, dQBar, kCW, W));
      kBarExp = stablehlo::AddOp::create(
          builder, loc, kBarExp,
          attnDotPTDOut(builder, loc, dSScaledW, dQBar, W));
    }
    if (dkResActive) {
      dSScaledBar = stablehlo::AddOp::create(
          builder, loc, dSScaledBar, attnDotQK(builder, loc, qCW, dKBarExp, W));
      qBar = stablehlo::AddOp::create(
          builder, loc, qBar, attnDotPV(builder, loc, dSScaledW, dKBarExp, W));
    }

    Value dSBar = dSScaledBar;
    if (scaleVal != 1.0) {
      Value sConst = createSplatConst(builder, loc, scoreWTy, scaleVal);
      dSBar = stablehlo::MulOp::create(builder, loc, dSBar, sConst);
    }
    if (dbResActive) {
      Value dBBcast = broadcastAttentionBias(builder, loc, dBBar, scoreWTy, W);
      dSBar = stablehlo::AddOp::create(builder, loc, dSBar, dBBcast);
    }

    Value PBar = stablehlo::MulOp::create(builder, loc, dSBar, dPMulW);
    if (dvResActive) {
      PBar = stablehlo::AddOp::create(
          builder, loc, PBar, attnDotQK(builder, loc, safeDOutW, dVBarExp, W));
    }

    Value dPBar = stablehlo::MulOp::create(builder, loc, dSBar, PW);
    int64_t sDim = (W == 1) ? 3 : 4;
    Value deltaBntBar = stablehlo::NegOp::create(
        builder, loc, reduceSumAlongDims(builder, loc, dPBar, {sDim}));
    SmallVector<int64_t> bntToBtnPerm = (W == 1)
                                            ? SmallVector<int64_t>{0, 2, 1}
                                            : SmallVector<int64_t>{0, 1, 3, 2};
    Value deltaBtnBar = stablehlo::TransposeOp::create(
        builder, loc, deltaBntBar, builder.getDenseI64ArrayAttr(bntToBtnPerm));
    SmallVector<int64_t> btnBcastDims = (W == 1)
                                            ? SmallVector<int64_t>{0, 1, 2}
                                            : SmallVector<int64_t>{0, 1, 2, 3};
    Value deltaBtndBar = stablehlo::BroadcastInDimOp::create(
        builder, loc, btndWTy, deltaBtnBar,
        builder.getDenseI64ArrayAttr(btnBcastDims));

    if (!gutils->isConstantValue(op.getOutput())) {
      Value safeOutBar =
          stablehlo::MulOp::create(builder, loc, deltaBtndBar, safeDOutW);
      Value outBar = stablehlo::SelectOp::create(builder, loc, qValidBtndW,
                                                 safeOutBar, zeroBtndW);
      gutils->addToDiffe(op.getOutput(),
                         convertElementType(builder, loc, outBar, elTy),
                         builder);
    }

    if (!gutils->isConstantValue(op.getDOutput())) {
      Value safeDOutBar = stablehlo::AddOp::create(
          builder, loc,
          stablehlo::MulOp::create(builder, loc, deltaBtndBar, safeOutW),
          attnDotPV(builder, loc, dPBar, vCW, W));
      if (dvResActive) {
        safeDOutBar =
            stablehlo::AddOp::create(builder, loc, safeDOutBar,
                                     attnDotPV(builder, loc, PW, dVBarExp, W));
      }
      Value dOutBar = stablehlo::SelectOp::create(builder, loc, qValidBtndW,
                                                  safeDOutBar, zeroBtndW);
      gutils->addToDiffe(op.getDOutput(),
                         convertElementType(builder, loc, dOutBar, elTy),
                         builder);
    }

    if (!gutils->isConstantValue(op.getValue())) {
      Value vBarExp = attnDotPTDOut(builder, loc, dPBar, safeDOutW, W);
      Value vBar = reduceAttentionKVHeads(builder, loc, vBarExp, N_kv, W);
      gutils->addToDiffe(op.getValue(),
                         convertElementType(builder, loc, vBar, elTy), builder);
    }

    PBar =
        stablehlo::SelectOp::create(builder, loc, validAllW, PBar, zeroScoreW);
    Value SBar = stablehlo::MulOp::create(builder, loc, PBar, PW);

    if (!gutils->isConstantValue(op.getLse())) {
      Value sumSBar = reduceSumAlongDims(builder, loc, SBar, {sDim});
      Value negSumSBar = stablehlo::NegOp::create(builder, loc, sumSBar);
      Value zeroLseW = createSplatConst(
          builder, loc, cast<RankedTensorType>(negSumSBar.getType()), 0.0);
      Value lseBar = stablehlo::SelectOp::create(builder, loc, qValidBntW,
                                                 negSumSBar, zeroLseW);
      gutils->addToDiffe(
          op.getLse(),
          convertElementType(builder, loc, lseBar, lseTy.getElementType()),
          builder);
    }

    if (op.getBias() && !gutils->isConstantValue(op.getBias())) {
      auto biasTy = cast<RankedTensorType>(bias.getType());
      Value biasBar =
          reduceAttentionBias(builder, loc, SBar, biasTy.getShape(), W);
      gutils->addToDiffe(op.getBias(),
                         convertElementType(builder, loc, biasBar, elTy),
                         builder);
    }

    Value SBarScaled = SBar;
    if (scaleVal != 1.0) {
      Value sConst = createSplatConst(builder, loc, scoreWTy, scaleVal);
      SBarScaled = stablehlo::MulOp::create(builder, loc, SBar, sConst);
    }

    if (!gutils->isConstantValue(op.getQuery())) {
      qBar = stablehlo::AddOp::create(
          builder, loc, qBar, attnDotPV(builder, loc, SBarScaled, kCW, W));
      gutils->addToDiffe(op.getQuery(),
                         convertElementType(builder, loc, qBar, elTy), builder);
    }

    if (!gutils->isConstantValue(op.getKey())) {
      kBarExp = stablehlo::AddOp::create(
          builder, loc, kBarExp,
          attnDotPTDOut(builder, loc, SBarScaled, qCW, W));
      Value kBar = reduceAttentionKVHeads(builder, loc, kBarExp, N_kv, W);
      gutils->addToDiffe(op.getKey(),
                         convertElementType(builder, loc, kBar, elTy), builder);
    }

    return success();
  }

  SmallVector<Value> cacheValues(Operation *orig,
                                 MGradientUtilsReverse *gutils) const {
    auto op = cast<AttentionGradOp>(orig);
    bool anyInputActive =
        !gutils->isConstantValue(op.getQuery()) ||
        !gutils->isConstantValue(op.getKey()) ||
        !gutils->isConstantValue(op.getValue()) ||
        !gutils->isConstantValue(op.getOutput()) ||
        !gutils->isConstantValue(op.getLse()) ||
        !gutils->isConstantValue(op.getDOutput()) ||
        (op.getBias() && !gutils->isConstantValue(op.getBias()));
    bool anyOutputActive =
        !gutils->isConstantValue(op.getDQuery()) ||
        !gutils->isConstantValue(op.getDKey()) ||
        !gutils->isConstantValue(op.getDValue()) ||
        (op.getDBias() && !gutils->isConstantValue(op.getDBias()));
    if (!anyInputActive || !anyOutputActive)
      return {};

    SmallVector<Value> caches;
    Operation *newOp = gutils->getNewFromOriginal(op);
    OpBuilder cacheBuilder(newOp);

    caches.push_back(gutils->initAndPushCache(
        gutils->getNewFromOriginal(op.getQuery()), cacheBuilder));
    caches.push_back(gutils->initAndPushCache(
        gutils->getNewFromOriginal(op.getKey()), cacheBuilder));
    caches.push_back(gutils->initAndPushCache(
        gutils->getNewFromOriginal(op.getValue()), cacheBuilder));
    caches.push_back(gutils->initAndPushCache(
        gutils->getNewFromOriginal(op.getOutput()), cacheBuilder));
    caches.push_back(gutils->initAndPushCache(
        gutils->getNewFromOriginal(op.getLse()), cacheBuilder));
    caches.push_back(gutils->initAndPushCache(
        gutils->getNewFromOriginal(op.getDOutput()), cacheBuilder));
    if (op.getBias())
      caches.push_back(gutils->initAndPushCache(
          gutils->getNewFromOriginal(op.getBias()), cacheBuilder));
    if (op.getMask())
      caches.push_back(gutils->initAndPushCache(
          gutils->getNewFromOriginal(op.getMask()), cacheBuilder));
    if (op.getQLengths())
      caches.push_back(gutils->initAndPushCache(
          gutils->getNewFromOriginal(op.getQLengths()), cacheBuilder));
    if (op.getKvLengths())
      caches.push_back(gutils->initAndPushCache(
          gutils->getNewFromOriginal(op.getKvLengths()), cacheBuilder));

    return caches;
  }

  LogicalResult createShadowValues(Operation *op, OpBuilder &builder,
                                   MGradientUtilsReverse *gutils) const {
    return success();
  }
};

struct AttentionBatchOpInterface
    : public BatchOpInterface::ExternalModel<AttentionBatchOpInterface,
                                             AttentionOp> {
  mlir::LogicalResult createBatch(Operation *src, OpBuilder &builder,
                                  IRMapping &mapper,
                                  ArrayRef<int64_t> batchSizes) const {
    auto op = cast<AttentionOp>(src);
    Location loc = op.getLoc();
    auto qTy = cast<RankedTensorType>(op.getQuery().getType());
    auto outTy = cast<RankedTensorType>(op.getOutput().getType());
    auto lseTy = cast<RankedTensorType>(op.getLse().getType());
    int64_t B = qTy.getDimSize(0);
    int64_t totalB = B;
    for (int64_t bs : batchSizes)
      totalB *= bs;

    Value qFlat = flattenBatchOperand(
        builder, loc, mapper.lookup(op.getQuery()), batchSizes, B);
    Value kFlat = flattenBatchOperand(builder, loc, mapper.lookup(op.getKey()),
                                      batchSizes, B);
    Value vFlat = flattenBatchOperand(
        builder, loc, mapper.lookup(op.getValue()), batchSizes, B);
    Value biasFlat =
        op.getBias()
            ? flattenBatchOperand(builder, loc, mapper.lookup(op.getBias()),
                                  batchSizes, B)
            : Value();
    Value maskFlat =
        op.getMask()
            ? flattenBatchOperand(builder, loc, mapper.lookup(op.getMask()),
                                  batchSizes, B)
            : Value();
    Value qLenFlat =
        op.getQLengths()
            ? flattenBatchOperand(builder, loc, mapper.lookup(op.getQLengths()),
                                  batchSizes, B)
            : Value();
    Value kvLenFlat =
        op.getKvLengths()
            ? flattenBatchOperand(
                  builder, loc, mapper.lookup(op.getKvLengths()), batchSizes, B)
            : Value();

    auto outFlatTy = RankedTensorType::get(
        {totalB, outTy.getDimSize(1), outTy.getDimSize(2), outTy.getDimSize(3)},
        outTy.getElementType());
    auto lseFlatTy = RankedTensorType::get(
        {totalB, lseTy.getDimSize(1), lseTy.getDimSize(2)},
        lseTy.getElementType());

    auto batchedOp = AttentionOp::create(
        builder, loc, outFlatTy, lseFlatTy, qFlat, kFlat, vFlat, biasFlat,
        maskFlat, qLenFlat, kvLenFlat, op.getScaleAttr(), op.getIsCausalAttr());

    mapper.map(op.getOutput(),
               unflattenBatchResult(builder, loc, batchedOp.getOutput(),
                                    batchSizes, B, outTy.getShape()));
    mapper.map(op.getLse(),
               unflattenBatchResult(builder, loc, batchedOp.getLse(),
                                    batchSizes, B, lseTy.getShape()));
    return success();
  }
};

struct AttentionGradBatchOpInterface
    : public BatchOpInterface::ExternalModel<AttentionGradBatchOpInterface,
                                             AttentionGradOp> {
  mlir::LogicalResult createBatch(Operation *src, OpBuilder &builder,
                                  IRMapping &mapper,
                                  ArrayRef<int64_t> batchSizes) const {
    auto op = cast<AttentionGradOp>(src);
    Location loc = op.getLoc();
    auto qTy = cast<RankedTensorType>(op.getQuery().getType());
    auto kTy = cast<RankedTensorType>(op.getKey().getType());
    auto vTy = cast<RankedTensorType>(op.getValue().getType());
    int64_t B = qTy.getDimSize(0);

    Value qFlat = flattenBatchOperand(
        builder, loc, mapper.lookup(op.getQuery()), batchSizes, B);
    Value kFlat = flattenBatchOperand(builder, loc, mapper.lookup(op.getKey()),
                                      batchSizes, B);
    Value vFlat = flattenBatchOperand(
        builder, loc, mapper.lookup(op.getValue()), batchSizes, B);
    Value outFlat = flattenBatchOperand(
        builder, loc, mapper.lookup(op.getOutput()), batchSizes, B);
    Value lseFlat = flattenBatchOperand(
        builder, loc, mapper.lookup(op.getLse()), batchSizes, B);
    Value dOutFlat = flattenBatchOperand(
        builder, loc, mapper.lookup(op.getDOutput()), batchSizes, B);
    Value biasFlat =
        op.getBias()
            ? flattenBatchOperand(builder, loc, mapper.lookup(op.getBias()),
                                  batchSizes, B)
            : Value();
    Value maskFlat =
        op.getMask()
            ? flattenBatchOperand(builder, loc, mapper.lookup(op.getMask()),
                                  batchSizes, B)
            : Value();
    Value qLenFlat =
        op.getQLengths()
            ? flattenBatchOperand(builder, loc, mapper.lookup(op.getQLengths()),
                                  batchSizes, B)
            : Value();
    Value kvLenFlat =
        op.getKvLengths()
            ? flattenBatchOperand(
                  builder, loc, mapper.lookup(op.getKvLengths()), batchSizes, B)
            : Value();

    Type dBiasFlatTy = op.getDBias() ? biasFlat.getType() : Type();
    auto batchedOp = AttentionGradOp::create(
        builder, loc, qFlat.getType(), kFlat.getType(), vFlat.getType(),
        dBiasFlatTy, qFlat, kFlat, vFlat, outFlat, lseFlat, dOutFlat, biasFlat,
        maskFlat, qLenFlat, kvLenFlat, op.getScaleAttr(), op.getIsCausalAttr());

    mapper.map(op.getDQuery(),
               unflattenBatchResult(builder, loc, batchedOp.getDQuery(),
                                    batchSizes, B, qTy.getShape()));
    mapper.map(op.getDKey(),
               unflattenBatchResult(builder, loc, batchedOp.getDKey(),
                                    batchSizes, B, kTy.getShape()));
    mapper.map(op.getDValue(),
               unflattenBatchResult(builder, loc, batchedOp.getDValue(),
                                    batchSizes, B, vTy.getShape()));
    if (op.getDBias()) {
      auto biasTy = cast<RankedTensorType>(op.getBias().getType());
      mapper.map(op.getDBias(),
                 unflattenBatchResult(builder, loc, batchedOp.getDBias(),
                                      batchSizes, B, biasTy.getShape(),
                                      /*sumOverSharedBatch=*/true));
    }
    return success();
  }
};

void mlir::enzyme::registerEnzymeXLADialectAutoDiffInterface(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *context, EnzymeXLADialect *) {
    registerInterfaces(context);
    GPUWrapperOp::attachInterface<GPUWrapperOpInterfaceReverse>(*context);
    GPUWrapperOp::attachInterface<GPUWrapperOpEnzymeOpsRemover>(*context);

    Pointer2MemrefOp::attachInterface<
        ViewCastOpInterfaceReverse<Pointer2MemrefOp>>(*context);
    Memref2PointerOp::attachInterface<
        ViewCastOpInterfaceReverse<Memref2PointerOp>>(*context);

    AttentionOp::attachInterface<AttentionActivity<AttentionOp>>(*context);
    AttentionOp::attachInterface<AttentionOpInterfaceRev>(*context);
    AttentionOp::attachInterface<AttentionOpInterfaceFwd>(*context);
    AttentionOp::attachInterface<AttentionBatchOpInterface>(*context);
    AttentionGradOp::attachInterface<AttentionActivity<AttentionGradOp>>(
        *context);
    AttentionGradOp::attachInterface<AttentionGradOpInterfaceRev>(*context);
    AttentionGradOp::attachInterface<AttentionGradOpInterfaceFwd>(*context);
    AttentionGradOp::attachInterface<AttentionGradBatchOpInterface>(*context);

    // Register batching interfaces
    JITCallOp::attachInterface<SHLOGenericBatchOpInterface<JITCallOp>>(
        *context);
    WrapOp::attachInterface<DimensionedOpBatchInterface<WrapOp>>(*context);
    ExtendOp::attachInterface<DimensionedOpBatchInterface<ExtendOp>>(*context);
    RotateOp::attachInterface<RotateOpBatchInterface>(*context);

    context->loadDialect<stablehlo::StablehloDialect>();
  });
}
