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

static LogicalResult checkThinSVD(SVDFactorizationOp op) {
  if (!op.getFull())
    return success();
  return op->emitError("enzymexla.linalg.svd: derivatives are implemented for "
                       "full = false");
}

static Value inElementTypeOf(OpBuilder &builder, Location loc, Value v,
                             Value like) {
  if (!isComplexTensor(like) || isComplexTensor(v))
    return v;
  return stablehlo::ComplexOp::create(builder, loc, v,
                                      splatLike(builder, loc, v, 0.0));
}

// the vector v along the columns (v[j] in column j) or the rows of a matrix
static Value broadcastVector(OpBuilder &builder, Location loc, Value v,
                             Value like, bool columns) {
  auto likeTy = cast<RankedTensorType>(like.getType());
  auto vTy = cast<RankedTensorType>(v.getType());
  int64_t rank = likeTy.getRank();
  SmallVector<int64_t> dims(rank - 2);
  std::iota(dims.begin(), dims.end(), 0);
  dims.push_back(columns ? rank - 1 : rank - 2);
  return stablehlo::BroadcastInDimOp::create(
      builder, loc,
      RankedTensorType::get(likeTy.getShape(), vTy.getElementType()), v, dims);
}

// x diag(v) or diag(v) x
static Value mulDiagonal(OpBuilder &builder, Location loc, Value x, Value v,
                         bool columns) {
  v = inElementTypeOf(builder, loc, v, x);
  return stablehlo::MulOp::create(builder, loc, x,
                                  broadcastVector(builder, loc, v, x, columns));
}

static Value diagonalOf(OpBuilder &builder, Location loc, Value x) {
  auto ty = cast<RankedTensorType>(x.getType());
  int64_t rank = ty.getRank();
  Value masked =
      keepTriangle(builder, loc, x, stablehlo::ComparisonDirection::EQ);
  auto scalarTy = RankedTensorType::get({}, ty.getElementType());
  Value zero = stablehlo::ConstantOp::create(
      builder, loc, scalarTy, cast<ElementsAttr>(makeAttr(scalarTy, 0.0)));
  SmallVector<int64_t> shape(ty.getShape().begin(), ty.getShape().end() - 1);
  SmallVector<int64_t> dims{rank - 2};
  auto sum = stablehlo::ReduceOp::create(
      builder, loc,
      TypeRange{RankedTensorType::get(shape, ty.getElementType())},
      ValueRange{masked}, ValueRange{zero}, dims);
  Block *body = new Block();
  sum.getBody().push_back(body);
  body->addArguments({scalarTy, scalarTy}, {loc, loc});
  OpBuilder bodyBuilder = OpBuilder::atBlockEnd(body);
  Value add = stablehlo::AddOp::create(bodyBuilder, loc, body->getArgument(0),
                                       body->getArgument(1));
  stablehlo::ReturnOp::create(bodyBuilder, loc, ValueRange(add));
  return sum->getResult(0);
}

// 1 / s, and 0 where s is 0
static Value svdInverse(OpBuilder &builder, Location loc, Value s) {
  Value zero = splatLike(builder, loc, s, 0.0);
  Value isZero = stablehlo::CompareOp::create(
      builder, loc, s, zero, stablehlo::ComparisonDirection::EQ);
  Value inv = stablehlo::DivOp::create(builder, loc,
                                       splatLike(builder, loc, s, 1.0), s);
  return stablehlo::SelectOp::create(builder, loc, isZero, zero, inv);
}

// F[i, j] = 1 / (s[j]^2 - s[i]^2), and 0 on the diagonal
static Value svdGapInverse(OpBuilder &builder, Location loc, Value s,
                           Value like) {
  Value s2 = stablehlo::MulOp::create(builder, loc, s, s);
  Value cols = broadcastVector(builder, loc, s2, like, /*columns=*/true);
  Value rows = broadcastVector(builder, loc, s2, like, /*columns=*/false);
  Value diag = matrixIndexPredicate(builder, loc, cols,
                                    stablehlo::ComparisonDirection::EQ);
  Value gap = stablehlo::SelectOp::create(
      builder, loc, diag, splatLike(builder, loc, cols, 1.0),
      stablehlo::SubtractOp::create(builder, loc, cols, rows));
  Value inv = stablehlo::SelectOp::create(
      builder, loc, diag, splatLike(builder, loc, cols, 0.0),
      stablehlo::DivOp::create(builder, loc, splatLike(builder, loc, cols, 1.0),
                               gap));
  return inElementTypeOf(builder, loc, inv, like);
}

// x - x^H
static Value skewHermitian(OpBuilder &builder, Location loc, Value x) {
  return stablehlo::SubtractOp::create(builder, loc, x,
                                       adjointMatrix(builder, loc, x));
}

// (I - q q^H) x
static Value projectOutColumns(OpBuilder &builder, Location loc, Value x,
                               Value q) {
  Value qhx = batchedMatmul(builder, loc, conjIfComplex(builder, loc, q), x,
                            /*transposeLhs=*/true);
  return stablehlo::SubtractOp::create(builder, loc, x,
                                       batchedMatmul(builder, loc, q, qhx));
}

// x (I - q^H q)
static Value projectOutRows(OpBuilder &builder, Location loc, Value x,
                            Value q) {
  Value xqh = batchedMatmul(builder, loc, x, conjIfComplex(builder, loc, q),
                            /*transposeLhs=*/false, /*transposeRhs=*/true);
  return stablehlo::SubtractOp::create(builder, loc, x,
                                       batchedMatmul(builder, loc, xqh, q));
}

class AutoDiffSVDFactorizationFwd
    : public AutoDiffOpInterface::ExternalModel<AutoDiffSVDFactorizationFwd,
                                                SVDFactorizationOp> {
public:
  LogicalResult createForwardModeTangent(Operation *orig, OpBuilder &builder,
                                         MGradientUtils *gutils) const {
    auto op = cast<SVDFactorizationOp>(orig);
    bool uActive = !gutils->isConstantValue(op.getU());
    bool sActive = !gutils->isConstantValue(op.getS());
    bool vActive = !gutils->isConstantValue(op.getVt());
    if (gutils->isConstantInstruction(op) || (!uActive && !sActive && !vActive))
      return success();
    if (failed(checkThinSVD(op)))
      return failure();

    // the tangent uses the primal results
    builder.setInsertionPointAfter(gutils->getNewFromOriginal(orig));
    Location loc = op.getLoc();
    int64_t width = gutils->width;
    Value u = broadcastToWidth(builder, loc,
                               gutils->getNewFromOriginal(op.getU()), width);
    Value s = broadcastToWidth(builder, loc,
                               gutils->getNewFromOriginal(op.getS()), width);
    Value vt = broadcastToWidth(builder, loc,
                                gutils->getNewFromOriginal(op.getVt()), width);
    Value da = gutils->invertPointerM(op.getInput(), builder);
    auto aTy = cast<RankedTensorType>(op.getInput().getType());
    int64_t m = aTy.getDimSize(aTy.getRank() - 2);
    int64_t n = aTy.getDimSize(aTy.getRank() - 1);

    // dS = U^H dA V
    Value uhda = batchedMatmul(builder, loc, conjIfComplex(builder, loc, u), da,
                               /*transposeLhs=*/true);
    Value vc = conjIfComplex(builder, loc, vt);
    Value dS = batchedMatmul(builder, loc, uhda, vc, /*transposeLhs=*/false,
                             /*transposeRhs=*/true);
    Value diag = diagonalOf(builder, loc, dS);
    if (sActive) {
      Value ds = diag;
      if (isComplexTensor(ds))
        ds = stablehlo::RealOp::create(builder, loc, ds);
      gutils->setDiffe(op.getS(), ds, builder);
    }
    if (!uActive && !vActive)
      return success();

    Value f = svdGapInverse(builder, loc, s, dS);
    Value sinv = svdInverse(builder, loc, s);
    if (uActive) {
      // dU = U (F * (dS S + S dS^H) + i Im(diag(dS)) S^-1)
      //      + (I - U U^H) dA V S^-1
      Value dSS = mulDiagonal(builder, loc, dS, s, /*columns=*/true);
      Value sym = stablehlo::AddOp::create(builder, loc, dSS,
                                           adjointMatrix(builder, loc, dSS));
      Value du = batchedMatmul(builder, loc, u,
                               stablehlo::MulOp::create(builder, loc, f, sym));
      if (isComplexTensor(dS)) {
        Value im = stablehlo::MulOp::create(
            builder, loc,
            stablehlo::SubtractOp::create(builder, loc, diag,
                                          conjIfComplex(builder, loc, diag)),
            splatLike(builder, loc, diag, 0.5));
        im = mulDiagonal(builder, loc, u, im, /*columns=*/true);
        du = stablehlo::AddOp::create(
            builder, loc, du,
            mulDiagonal(builder, loc, im, sinv, /*columns=*/true));
      }
      if (m > n) {
        Value daV = batchedMatmul(builder, loc, da, vc, /*transposeLhs=*/false,
                                  /*transposeRhs=*/true);
        du = stablehlo::AddOp::create(
            builder, loc, du,
            mulDiagonal(builder, loc, projectOutColumns(builder, loc, daV, u),
                        sinv,
                        /*columns=*/true));
      }
      gutils->setDiffe(op.getU(), du, builder);
    }
    if (vActive) {
      // dVt = (F * (S dS + dS^H S))^H Vt + S^-1 U^H dA (I - V V^H)
      Value sdS = mulDiagonal(builder, loc, dS, s, /*columns=*/false);
      Value sym = stablehlo::AddOp::create(builder, loc, sdS,
                                           adjointMatrix(builder, loc, sdS));
      Value dvt = batchedMatmul(
          builder, loc,
          adjointMatrix(builder, loc,
                        stablehlo::MulOp::create(builder, loc, f, sym)),
          vt);
      if (n > m) {
        Value x = mulDiagonal(builder, loc, uhda, sinv, /*columns=*/false);
        dvt = stablehlo::AddOp::create(builder, loc, dvt,
                                       projectOutRows(builder, loc, x, vt));
      }
      gutils->setDiffe(op.getVt(), dvt, builder);
    }
    return success();
  }
};

class AutoDiffSVDFactorizationRev
    : public ReverseAutoDiffOpInterface::ExternalModel<
          AutoDiffSVDFactorizationRev, SVDFactorizationOp> {
public:
  LogicalResult createReverseModeAdjoint(Operation *orig, OpBuilder &builder,
                                         MGradientUtilsReverse *gutils,
                                         SmallVector<Value> caches) const {
    auto op = cast<SVDFactorizationOp>(orig);
    bool uActive = !gutils->isConstantValue(op.getU());
    bool sActive = !gutils->isConstantValue(op.getS());
    bool vActive = !gutils->isConstantValue(op.getVt());
    if (gutils->isConstantInstruction(op) || (!uActive && !sActive && !vActive))
      return success();
    if (failed(checkThinSVD(op)))
      return failure();

    Location loc = op.getLoc();
    Value gu = nullptr, gs = nullptr, gvt = nullptr;
    if (uActive) {
      gu = gutils->diffe(op.getU(), builder);
      gutils->zeroDiffe(op.getU(), builder);
    }
    if (sActive) {
      gs = gutils->diffe(op.getS(), builder);
      gutils->zeroDiffe(op.getS(), builder);
    }
    if (vActive) {
      gvt = gutils->diffe(op.getVt(), builder);
      gutils->zeroDiffe(op.getVt(), builder);
    }
    if (gutils->isConstantValue(op.getInput()))
      return success();

    int64_t width = gutils->width;
    Value u = broadcastToWidth(builder, loc,
                               gutils->popCache(caches[0], builder), width);
    Value s = broadcastToWidth(builder, loc,
                               gutils->popCache(caches[1], builder), width);
    Value vt = broadcastToWidth(builder, loc,
                                gutils->popCache(caches[2], builder), width);
    auto aTy = cast<RankedTensorType>(op.getInput().getType());
    int64_t m = aTy.getDimSize(aTy.getRank() - 2);
    int64_t n = aTy.getDimSize(aTy.getRank() - 1);
    auto add = [&](Value &acc, Value v) {
      if (acc)
        acc = stablehlo::AddOp::create(builder, loc, acc, v);
      else
        acc = v;
    };

    // A_bar = (U G + (I - U U^H) U_bar S^-1) Vt
    //         + U S^-1 Vt_bar (I - V V^H)
    // G = F * (J S + S K) + diag(s_bar) + diag(J) / (2 S),
    // J = U^H U_bar - U_bar^H U, K = Vt Vt_bar^H - Vt_bar Vt^H
    Value ug = nullptr, sinv = nullptr;
    if (gs)
      add(ug, mulDiagonal(builder, loc, u, gs, /*columns=*/true));
    if (gu || gvt) {
      sinv = svdInverse(builder, loc, s);
      Value jk = nullptr;
      if (gu) {
        Value j = skewHermitian(
            builder, loc,
            batchedMatmul(builder, loc, conjIfComplex(builder, loc, u), gu,
                          /*transposeLhs=*/true));
        add(jk, mulDiagonal(builder, loc, j, s, /*columns=*/true));
        if (isComplexTensor(u)) {
          Value dj = diagonalOf(builder, loc, j);
          Value d = stablehlo::MulOp::create(builder, loc, dj,
                                             splatLike(builder, loc, dj, 0.5));
          Value ud = mulDiagonal(builder, loc, u, d, /*columns=*/true);
          add(ug, mulDiagonal(builder, loc, ud, sinv, /*columns=*/true));
        }
      }
      if (gvt) {
        Value k = skewHermitian(
            builder, loc,
            batchedMatmul(builder, loc, vt, conjIfComplex(builder, loc, gvt),
                          /*transposeLhs=*/false, /*transposeRhs=*/true));
        add(jk, mulDiagonal(builder, loc, k, s, /*columns=*/false));
      }
      Value g = stablehlo::MulOp::create(
          builder, loc, svdGapInverse(builder, loc, s, jk), jk);
      add(ug, batchedMatmul(builder, loc, u, g));
      if (gu && m > n) {
        Value x = mulDiagonal(builder, loc, gu, sinv, /*columns=*/true);
        add(ug, projectOutColumns(builder, loc, x, u));
      }
    }
    Value abar = batchedMatmul(builder, loc, ug, vt);
    if (gvt && n > m) {
      Value x = mulDiagonal(builder, loc, gvt, sinv, /*columns=*/false);
      add(abar,
          batchedMatmul(builder, loc, u, projectOutRows(builder, loc, x, vt)));
    }
    gutils->addToDiffe(op.getInput(), abar, builder);
    return success();
  }

  SmallVector<Value> cacheValues(Operation *orig,
                                 MGradientUtilsReverse *gutils) const {
    auto op = cast<SVDFactorizationOp>(orig);
    if (gutils->isConstantInstruction(op) ||
        gutils->isConstantValue(op.getInput()) ||
        (gutils->isConstantValue(op.getU()) &&
         gutils->isConstantValue(op.getS()) &&
         gutils->isConstantValue(op.getVt())))
      return {};
    Operation *newOp = gutils->getNewFromOriginal(orig);
    OpBuilder cacheBuilder(newOp);
    cacheBuilder.setInsertionPointAfter(newOp);
    SmallVector<Value> caches;
    for (Value v : {op.getU(), op.getS(), op.getVt()})
      caches.push_back(gutils->initAndPushCache(gutils->getNewFromOriginal(v),
                                                cacheBuilder));
    return caches;
  }

  LogicalResult createShadowValues(Operation *op, OpBuilder &builder,
                                   MGradientUtilsReverse *gutils) const {
    return success();
  }
};

} // namespace

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
    SVDFactorizationOp::attachInterface<AutoDiffSVDFactorizationFwd>(*context);
    SVDFactorizationOp::attachInterface<AutoDiffSVDFactorizationRev>(*context);

    // Register batching interfaces
    JITCallOp::attachInterface<SHLOGenericBatchOpInterface<JITCallOp>>(
        *context);

    context->loadDialect<stablehlo::StablehloDialect>();
  });
}
