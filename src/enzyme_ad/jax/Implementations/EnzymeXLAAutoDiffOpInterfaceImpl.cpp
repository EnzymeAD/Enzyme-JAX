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
#include "src/enzyme_ad/jax/Implementations/LinalgUtils.h"
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

static LogicalResult checkSquareLU(LUFactorizationOp op) {
  auto ty = cast<RankedTensorType>(op.getInput().getType());
  int64_t rank = ty.getRank();
  if (ty.getDimSize(rank - 1) == ty.getDimSize(rank - 2))
    return success();
  return op->emitError(
      "enzymexla.linalg.lu: derivatives are implemented for square matrices");
}

// the matrix P of the 1-based row permutation perm, with P input = L U
static Value luPermutationMatrix(OpBuilder &builder, Location loc, Value perm,
                                 Value like) {
  auto likeTy = cast<RankedTensorType>(like.getType());
  auto permTy = cast<RankedTensorType>(perm.getType());
  int64_t rank = likeTy.getRank();
  auto indexTy =
      RankedTensorType::get(likeTy.getShape(), permTy.getElementType());
  SmallVector<int64_t> dims(rank - 1);
  std::iota(dims.begin(), dims.end(), 0);
  Value rows =
      stablehlo::BroadcastInDimOp::create(builder, loc, indexTy, perm, dims);
  Value cols = stablehlo::IotaOp::create(builder, loc, indexTy, rank - 1);
  Value one = stablehlo::ConstantOp::create(
      builder, loc, indexTy,
      cast<ElementsAttr>(makeAttr(indexTy, static_cast<int64_t>(1))));
  Value match = stablehlo::CompareOp::create(
      builder, loc, rows, stablehlo::AddOp::create(builder, loc, cols, one),
      stablehlo::ComparisonDirection::EQ);
  return stablehlo::SelectOp::create(builder, loc, match,
                                     splatLike(builder, loc, like, 1.0),
                                     splatLike(builder, loc, like, 0.0));
}

static Value luLowerFactor(OpBuilder &builder, Location loc, Value lu) {
  Value eye = stablehlo::SelectOp::create(
      builder, loc,
      matrixIndexPredicate(builder, loc, lu,
                           stablehlo::ComparisonDirection::EQ),
      splatLike(builder, loc, lu, 1.0), splatLike(builder, loc, lu, 0.0));
  return stablehlo::SelectOp::create(
      builder, loc,
      matrixIndexPredicate(builder, loc, lu,
                           stablehlo::ComparisonDirection::GT),
      lu, eye);
}

static Value luUpperFactor(OpBuilder &builder, Location loc, Value lu) {
  return keepTriangle(builder, loc, lu, stablehlo::ComparisonDirection::LE);
}

class AutoDiffLUFactorizationFwd
    : public AutoDiffOpInterface::ExternalModel<AutoDiffLUFactorizationFwd,
                                                LUFactorizationOp> {
public:
  LogicalResult createForwardModeTangent(Operation *orig, OpBuilder &builder,
                                         MGradientUtils *gutils) const {
    auto op = cast<LUFactorizationOp>(orig);
    if (gutils->isConstantInstruction(op) ||
        gutils->isConstantValue(op.getOutput()))
      return success();
    if (failed(checkSquareLU(op)))
      return failure();

    // the tangent uses the primal results
    builder.setInsertionPointAfter(gutils->getNewFromOriginal(orig));
    Location loc = op.getLoc();
    int64_t width = gutils->width;
    Value lu = broadcastToWidth(
        builder, loc, gutils->getNewFromOriginal(op.getOutput()), width);
    Value perm = broadcastToWidth(
        builder, loc, gutils->getNewFromOriginal(op.getPermutation()), width);
    auto none = stablehlo::Transpose::NO_TRANSPOSE;

    // F = L^-1 P dA U^-1, dL = L tril(F, -1), dU = triu(F) U
    Value x =
        batchedMatmul(builder, loc, luPermutationMatrix(builder, loc, perm, lu),
                      gutils->invertPointerM(op.getInput(), builder));
    Value y = triangularSolve(builder, loc, lu, x, /*leftSide=*/true,
                              /*lower=*/true, /*unitDiagonal=*/true, none);
    Value f = triangularSolve(builder, loc, lu, y, /*leftSide=*/false,
                              /*lower=*/false, /*unitDiagonal=*/false, none);
    Value dL = batchedMatmul(
        builder, loc, luLowerFactor(builder, loc, lu),
        keepTriangle(builder, loc, f, stablehlo::ComparisonDirection::GT));
    Value dU = batchedMatmul(
        builder, loc,
        keepTriangle(builder, loc, f, stablehlo::ComparisonDirection::LE),
        luUpperFactor(builder, loc, lu));
    gutils->setDiffe(op.getOutput(),
                     stablehlo::AddOp::create(builder, loc, dL, dU), builder);
    return success();
  }
};

class AutoDiffLUFactorizationRev
    : public ReverseAutoDiffOpInterface::ExternalModel<
          AutoDiffLUFactorizationRev, LUFactorizationOp> {
public:
  LogicalResult createReverseModeAdjoint(Operation *orig, OpBuilder &builder,
                                         MGradientUtilsReverse *gutils,
                                         SmallVector<Value> caches) const {
    auto op = cast<LUFactorizationOp>(orig);
    if (gutils->isConstantInstruction(op) ||
        gutils->isConstantValue(op.getOutput()))
      return success();
    if (failed(checkSquareLU(op)))
      return failure();

    Value g = gutils->diffe(op.getOutput(), builder);
    gutils->zeroDiffe(op.getOutput(), builder);
    if (gutils->isConstantValue(op.getInput()))
      return success();

    Location loc = op.getLoc();
    int64_t width = gutils->width;
    Value lu = broadcastToWidth(builder, loc,
                                gutils->popCache(caches[0], builder), width);
    Value perm = broadcastToWidth(builder, loc,
                                  gutils->popCache(caches[1], builder), width);
    auto adj = adjointTranspose(lu);

    // F_bar = tril(L^H G, -1) + triu(G U^H), A_bar = P^T L^-H F_bar U^-H
    Value lhg = batchedMatmul(
        builder, loc,
        conjIfComplex(builder, loc, luLowerFactor(builder, loc, lu)), g,
        /*transposeLhs=*/true);
    Value guh = batchedMatmul(
        builder, loc, g,
        conjIfComplex(builder, loc, luUpperFactor(builder, loc, lu)),
        /*transposeLhs=*/false, /*transposeRhs=*/true);
    Value fbar = stablehlo::SelectOp::create(
        builder, loc,
        matrixIndexPredicate(builder, loc, g,
                             stablehlo::ComparisonDirection::GT),
        lhg, guh);
    Value y = triangularSolve(builder, loc, lu, fbar, /*leftSide=*/true,
                              /*lower=*/true, /*unitDiagonal=*/true, adj);
    Value z = triangularSolve(builder, loc, lu, y, /*leftSide=*/false,
                              /*lower=*/false, /*unitDiagonal=*/false, adj);
    Value abar = batchedMatmul(builder, loc,
                               luPermutationMatrix(builder, loc, perm, lu), z,
                               /*transposeLhs=*/true);
    gutils->addToDiffe(op.getInput(), abar, builder);
    return success();
  }

  SmallVector<Value> cacheValues(Operation *orig,
                                 MGradientUtilsReverse *gutils) const {
    auto op = cast<LUFactorizationOp>(orig);
    if (gutils->isConstantInstruction(op) ||
        gutils->isConstantValue(op.getOutput()) ||
        gutils->isConstantValue(op.getInput()))
      return {};
    Operation *newOp = gutils->getNewFromOriginal(orig);
    OpBuilder cacheBuilder(newOp);
    cacheBuilder.setInsertionPointAfter(newOp);
    return {gutils->initAndPushCache(gutils->getNewFromOriginal(op.getOutput()),
                                     cacheBuilder),
            gutils->initAndPushCache(
                gutils->getNewFromOriginal(op.getPermutation()), cacheBuilder)};
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

    LUFactorizationOp::attachInterface<AutoDiffLUFactorizationFwd>(*context);
    LUFactorizationOp::attachInterface<AutoDiffLUFactorizationRev>(*context);

    // Register batching interfaces
    JITCallOp::attachInterface<SHLOGenericBatchOpInterface<JITCallOp>>(
        *context);
    WrapOp::attachInterface<DimensionedOpBatchInterface<WrapOp>>(*context);
    ExtendOp::attachInterface<DimensionedOpBatchInterface<ExtendOp>>(*context);
    RotateOp::attachInterface<RotateOpBatchInterface>(*context);

    context->loadDialect<stablehlo::StablehloDialect>();
  });
}
