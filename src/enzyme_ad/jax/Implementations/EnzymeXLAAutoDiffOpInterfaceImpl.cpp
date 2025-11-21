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
#include "mlir/Dialect/MemRef/IR/MemRef.h"
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
    // Assume GPU allocations need to be in address space 1
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
          if (auto subviewOp = dyn_cast<memref::SubViewOp>(op)) {
            auto newType = cast<MemRefType>(*subviewOp.getType().clonePtrWith(
                globalMemSpace, std::nullopt));
            subviewOp.getResult().setType(newType);
          }
        });
      }
    }

    for (auto &info : caches) {
      rewriter.moveOpBefore(info.pushOp, wrapOp);
      auto revWrapper = info.popOp->getParentOfType<enzymexla::GPUWrapperOp>();
      assert(revWrapper && "failed to find reverse gpu_wrapper");
      rewriter.moveOpBefore(info.popOp, revWrapper);

      SmallVector<Value> frontier{info.popOp.getResult()};
      traverseDownDefUseChains(frontier, [globalMemSpace](Operation *op) {
        if (auto subviewOp = dyn_cast<memref::SubViewOp>(op)) {
          auto newType = cast<MemRefType>(
              *subviewOp.getType().clonePtrWith(globalMemSpace, std::nullopt));
          subviewOp.getResult().setType(newType);
        }
      });

      for (auto user : info.popOp.getResult().getUsers()) {
        if (hasSingleEffect<MemoryEffects::Free>(user)) {
          rewriter.eraseOp(user);
        }
      }
      rewriter.setInsertionPointAfter(revWrapper);
      gpu::DeallocOp::create(rewriter, wrapOp.getLoc(), TypeRange(),
                             info.popOp.getResult());
    }

    return success();
    // TODO need to convert to gpu allocations and conversion/copy

    /*
    for (auto grad : gradients) {
      auto trueValue = trueMapping.lookupOrNull(grad);
      trueTerm->insertOperands(trueTerm->getNumOperands(),
                               ValueRange(trueValue));

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
} for (auto &info : caches) {
  rewriter.moveOpBefore(info.pushOp, wrapOp);
  auto revWrapper = info.popOp->getParentOfType<enzymexla::GPUWrapperOp>();
  assert(revWrapper && "failed to find reverse gpu_wrapper");
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
}
;

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

  void createShadowValues(Operation *op, OpBuilder &builder,
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

  void createShadowValues(Operation *op, OpBuilder &builder,
                          MGradientUtilsReverse *gutils) const {}
};

class Pointer2MemrefRev : public ReverseAutoDiffOpInterface::ExternalModel<
                              Pointer2MemrefRev, enzymexla::Pointer2MemrefOp> {
public:
  LogicalResult createReverseModeAdjoint(Operation *orig, OpBuilder &builder,
                                         MGradientUtilsReverse *gutils,
                                         SmallVector<Value> caches) const {
    return success();
  }

  SmallVector<Value> cacheValues(Operation *orig,
                                 MGradientUtilsReverse *gutils) const {
    return SmallVector<Value>();
  }

  void createShadowValues(Operation *op, OpBuilder &builder,
                          MGradientUtilsReverse *gutils) const {
    auto p2m = cast<enzymexla::Pointer2MemrefOp>(op);
    if (!gutils->isConstantValue(p2m)) {
      Value dres = gutils->invertPointerM(p2m.getSource(), builder);
      Value shadow = builder.create<enzymexla::Pointer2MemrefOp>(
          p2m.getLoc(), p2m.getType(), dres);
      gutils->setInvertedPointer(p2m, shadow);
    }
  }
};
} // namespace

void mlir::enzyme::registerEnzymeXLADialectAutoDiffInterface(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *context, EnzymeXLADialect *) {
    registerInterfaces(context);
    GPUWrapperOp::attachInterface<GPUWrapperOpInterfaceReverse>(*context);
    GPUWrapperOp::attachInterface<GPUWrapperOpEnzymeOpsRemover>(*context);
    enzymexla::Pointer2MemrefOp::attachInterface<Pointer2MemrefRev>(*context);

    Pointer2MemrefOp::attachInterface<
        ViewCastOpInterfaceReverse<Pointer2MemrefOp>>(*context);
    Memref2PointerOp::attachInterface<
        ViewCastOpInterfaceReverse<Memref2PointerOp>>(*context);

    // Register batching interfaces
    JITCallOp::attachInterface<SHLOGenericBatchOpInterface<JITCallOp>>(
        *context);

    context->loadDialect<stablehlo::StablehloDialect>();
  });
}
