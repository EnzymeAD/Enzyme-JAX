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
#include "Enzyme/MLIR/Interfaces/AutoDiffTypeInterface.h"
#include "Enzyme/MLIR/Interfaces/GradientUtils.h"
#include "Enzyme/MLIR/Interfaces/GradientUtilsReverse.h"
#include "Enzyme/MLIR/Passes/RemovalUtils.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
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

        // The gradients of the values of the kernel do not outlive it, which
        // matters when it is nested in a region they cannot escape, like a
        // jit_region.
        bodyBuilder.setInsertionPointToStart(&revBB);
        localizeGradients(bodyBuilder, gutils, &oBB);

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

// A jit_region's body may move data between any of its buffers, so any input
// can flow into any buffer and any result.
struct JITRegionOpADDataFlow
    : public ADDataFlowOpInterface::ExternalModel<JITRegionOpADDataFlow,
                                                  JITRegionOp> {
  SmallVector<Value> getPotentialIncomingValuesRes(Operation *op,
                                                   OpResult v) const {
    auto region = cast<JITRegionOp>(op);
    SmallVector<Value> sources(region.getInputs());
    llvm::append_range(sources, region.getBodyBlock()->getArguments());
    return sources;
  }

  SmallVector<Value> getPotentialIncomingValuesArg(Operation *op,
                                                   BlockArgument v) const {
    return llvm::to_vector(cast<JITRegionOp>(op).getInputs());
  }

  SmallVector<Value> getPotentialTerminatorUsers(Operation *op, Operation *term,
                                                 Value v) const {
    return {};
  }
};

// Lists, on the augmented primal of a jit_region, the inputs whose buffers
// hold the shadows of its active buffers, in the order of the corresponding
// buffers of the reverse jit_region.
constexpr static llvm::StringLiteral kJITRegionShadowsAttrName =
    "enzymexla.jit_region_shadows";

static RankedTensorType getTensorType(Value buffer) {
  auto type = cast<MemRefType>(buffer.getType());
  return RankedTensorType::get(type.getShape(), type.getElementType());
}

// The layout of the input `i` of a jit_region, or null if it has no
// operand_layouts.
static Attribute getInputLayout(JITRegionOp region, unsigned i) {
  auto layouts = dyn_cast_or_null<ArrayAttr>(region.getOperandLayoutsAttr());
  return layouts ? layouts[i] : nullptr;
}

// The layouts of the inputs of a jit_region, null if it has no
// operand_layouts.
static SmallVector<Attribute> getInputLayouts(JITRegionOp region) {
  SmallVector<Attribute> layouts;
  for (unsigned i = 0, e = region.getInputs().size(); i < e; ++i)
    layouts.push_back(getInputLayout(region, i));
  return layouts;
}

// The default, row-major, layout of a buffer.
static Attribute getDefaultLayout(Builder &builder, Type type) {
  int64_t rank = cast<ShapedType>(type).getRank();
  return builder.getIndexTensorAttr(
      llvm::to_vector(llvm::reverse(llvm::seq<int64_t>(0, rank))));
}

// Moves the body of a jit_region to a new one with the given inputs, with one
// result per input. If the region has operand_layouts, the input `i` of the
// new region has the layout `layouts[i]`.
static JITRegionOp rebuildJITRegion(RewriterBase &rewriter, JITRegionOp region,
                                    ValueRange inputs,
                                    ArrayRef<Attribute> layouts) {
  NamedAttrList attrs(region->getAttrDictionary());
  attrs.erase(region.getOutputOperandAliasesAttrName());
  if (region.getOperandLayoutsAttr()) {
    assert(layouts.size() == inputs.size() && "expected a layout per input");
    attrs.set(region.getOperandLayoutsAttrName(),
              rewriter.getArrayAttr(layouts));
  }
  auto newRegion = JITRegionOp::create(rewriter, region.getLoc(),
                                       inputs.getTypes(), inputs, attrs);
  rewriter.inlineRegionBefore(region.getBody(), newRegion.getBody(),
                              newRegion.getBody().end());
  return newRegion;
}

// The result of a jit_region rebuilt by rebuildJITRegion standing for the
// result `result` of the original region.
static OpResult getRebuiltResult(JITRegionOp rebuilt, OpResult result) {
  auto region = cast<JITRegionOp>(result.getOwner());
  return rebuilt->getResult(
      region.getAliasedInputIndex(result.getResultNumber()));
}

// Replaces a jit_region by its rebuilt version.
static void replaceJITRegion(RewriterBase &rewriter, JITRegionOp region,
                             JITRegionOp rebuilt) {
  for (OpResult result : region->getResults())
    rewriter.replaceAllUsesWith(result, getRebuiltResult(rebuilt, result));
  rewriter.eraseOp(region);
}

// Reverse of
//
//   %r = jit_region(%x) { ^bb0(%m): body }
//
// is a jit_region whose buffers are the shadows of the primal buffers:
//
//   %dx = jit_region(%dr) { ^bb0(%dm): reverse(body) }
//
// Each shadow buffer starts out holding the adjoint of the result read from
// it, and ends holding the adjoint of the input it was initialized from.
//
// In the augmented primal, the shadow of a primal buffer is an extra buffer of
// the region, initialized from a zero gradient tensor passed as a new input.
// It stands for the buffer of the reverse region holding the shadow: the
// caches the augmented primal pushes on it are rematerialized on the latter by
// JITRegionOpEnzymeOpsRemover, which then drops the extra buffers.
struct JITRegionOpInterfaceReverse
    : public ReverseAutoDiffOpInterface::ExternalModel<
          JITRegionOpInterfaceReverse, JITRegionOp> {
  // The primal buffers that have a shadow.
  static SmallVector<BlockArgument> getActiveBuffers(JITRegionOp region,
                                                     MGradientUtils *gutils) {
    SmallVector<BlockArgument> active;
    for (BlockArgument arg : region.getBodyBlock()->getArguments())
      if (!gutils->isConstantValue(arg))
        active.push_back(arg);
    return active;
  }

  LogicalResult createReverseModeAdjoint(Operation *op, OpBuilder &builder,
                                         MGradientUtilsReverse *gutils,
                                         SmallVector<Value> caches) const {
    auto region = cast<JITRegionOp>(op);
    Block *oBB = region.getBodyBlock();

    for (BlockArgument arg : oBB->getArguments()) {
      OpResult result = region.getAliasingResult(arg.getArgNumber());
      if (result && !gutils->isConstantValue(result) &&
          gutils->isConstantValue(arg))
        return region.emitError()
               << "active result #" << result.getResultNumber()
               << " of a jit_region is read from an inactive buffer";
    }

    // The adjoint of a result seeds the shadow of the buffer it is read from.
    SmallVector<BlockArgument> activeBuffers = getActiveBuffers(region, gutils);
    SmallVector<Value> seeds;
    SmallVector<Type> shadowTypes;
    SmallVector<Location> shadowLocs;
    for (BlockArgument arg : activeBuffers) {
      OpResult result = region.getAliasingResult(arg.getArgNumber());
      Value seed;
      if (result && !gutils->isConstantValue(result)) {
        seed = gutils->diffe(result, builder);
        gutils->zeroDiffe(result, builder);
      } else {
        seed = cast<AutoDiffTypeInterface>(
                   region.getInputs()[arg.getArgNumber()].getType())
                   .createNullValue(builder, region.getLoc());
      }
      seeds.push_back(seed);
      shadowTypes.push_back(gutils->getShadowType(arg.getType()));
      shadowLocs.push_back(arg.getLoc());
    }

    auto revRegion = JITRegionOp::create(builder, region.getLoc(),
                                         ValueRange(seeds).getTypes(), seeds);
    // The kernels index a shadow like its primal buffer.
    if (region.getOperandLayoutsAttr()) {
      SmallVector<Attribute> layouts;
      for (BlockArgument arg : activeBuffers)
        layouts.push_back(getInputLayout(region, arg.getArgNumber()));
      revRegion.setOperandLayoutsAttr(builder.getArrayAttr(layouts));
    }

    Block *revBB;
    {
      OpBuilder::InsertionGuard guard(builder);
      revBB = builder.createBlock(&revRegion.getBody(), {}, shadowTypes,
                                  shadowLocs);
      JITRegionYieldOp::create(builder, region.getLoc());
    }

    // No gradient of a value of the region can outlive it. The reverse rules
    // of nested ops only localize the gradients of their own block, if at all,
    // so localize those of every block of the region.
    OpBuilder bodyBuilder(revBB, revBB->begin());
    region.getBody().walk(
        [&](Block *block) { localizeGradients(bodyBuilder, gutils, block); });
    bodyBuilder.setInsertionPoint(revBB->getTerminator());

    bool valid = true;
    for (Operation &inner :
         llvm::drop_begin(llvm::reverse(oBB->getOperations()))) {
      valid &=
          gutils->Logic.visitChild(&inner, bodyBuilder, gutils).succeeded();
    }

    // The final contents of a shadow are the adjoint of the input the primal
    // buffer was initialized from.
    for (auto [arg, gradient] :
         llvm::zip_equal(activeBuffers, revRegion.getResults())) {
      Value input = region.getInputs()[arg.getArgNumber()];
      if (!gutils->isConstantValue(input))
        gutils->addToDiffe(input, gradient, builder);
    }

    return success(valid);
  }

  SmallVector<Value> cacheValues(Operation *op,
                                 MGradientUtilsReverse *gutils) const {
    return {};
  }

  // Erases the shadow block arguments the augmented primal inserted after the
  // block arguments of active buffers. Nothing uses them yet. A shadow has the
  // type of its buffer, so an unused block argument of the type of the
  // previous one is taken as a shadow while there are more block arguments
  // than inputs: if it was the buffer of the next input instead, both are
  // unused block arguments of the same type and either can go.
  static void stripShadowArguments(JITRegionOp region) {
    Block *body = region.getBodyBlock();
    if (body->getNumArguments() <= region.getInputs().size())
      return;
    unsigned extra = body->getNumArguments() - region.getInputs().size();

    BitVector shadows(body->getNumArguments());
    for (unsigned k = 0; k + 1 < body->getNumArguments() && extra; ++k) {
      BlockArgument next = body->getArgument(k + 1);
      if (next.getType() == body->getArgument(k).getType() &&
          next.use_empty()) {
        shadows.set(k + 1);
        --extra;
        ++k;
      }
    }
    body->eraseArguments(shadows);
  }

  LogicalResult createShadowValues(Operation *op, OpBuilder &builder,
                                   MGradientUtilsReverse *gutils) const {
    auto region = cast<JITRegionOp>(op);
    auto newRegion = cast<JITRegionOp>(gutils->getNewFromOriginal(op));

    // The augmented primal gives each active buffer a shadow block argument
    // without a matching input, which a jit_region cannot have. Loop
    // checkpointing clones the augmented primal without visiting the clones,
    // so strip them from every jit_region of the function that was not
    // rebuilt below.
    gutils->newFunc->walk([](JITRegionOp jitRegion) {
      if (!jitRegion->hasAttr(kJITRegionShadowsAttrName))
        stripShadowArguments(jitRegion);
    });

    SmallVector<BlockArgument> activeBuffers = getActiveBuffers(region, gutils);
    if (activeBuffers.empty())
      return success();

    // Give each active buffer a shadow buffer, initialized from a zero
    // gradient tensor.
    Block *newBB = newRegion.getBodyBlock();
    SmallVector<Value> inputs(newRegion.getInputs());
    SmallVector<Attribute> layouts = getInputLayouts(newRegion);
    SmallVector<int64_t> shadowInputs;
    for (BlockArgument arg : activeBuffers) {
      BlockArgument shadow = newBB->addArgument(
          gutils->getShadowType(arg.getType()), arg.getLoc());
      shadowInputs.push_back(inputs.size());
      inputs.push_back(cast<AutoDiffTypeInterface>(getTensorType(shadow))
                           .createNullValue(builder, arg.getLoc()));
      layouts.push_back(getInputLayout(newRegion, arg.getArgNumber()));
      gutils->invertedPointers.map(arg, shadow);
    }

    IRRewriter rewriter(builder);
    auto augmented = rebuildJITRegion(rewriter, newRegion, inputs, layouts);
    augmented->setAttr(kJITRegionShadowsAttrName,
                       builder.getDenseI64ArrayAttr(shadowInputs));
    for (OpResult result : region->getResults()) {
      OpResult newResult = getRebuiltResult(augmented, result);
      gutils->getNewFromOriginal(result).replaceAllUsesWith(newResult);
      gutils->originalToNewFn.map(result, newResult);
    }
    gutils->originalToNewFnOps[op] = augmented;
    gutils->erase(newRegion);
    return success();
  }
};

// Whether a value is defined above a jit_region, and so still available
// around its reverse region.
static bool isDefinedAbove(JITRegionOp region, Value v) {
  return v.getParentRegion()->isAncestor(region->getParentRegion());
}

// Moves the caches pushed in the augmented primal of a jit_region to its
// reverse region. As the region's buffers do not outlive it, a cached value
// cannot simply be pushed after it; instead, each cached value is
// reconstructed in the reverse region from:
//
//  * values defined above the region, pushed before it and popped before the
//    reverse region;
//  * shadow buffers, which stand for the buffers of the reverse region holding
//    the shadows;
//  * buffers of the region: their final contents, which are results of the
//    region, are cached and passed to the reverse region as extra inputs,
//    whose buffers replace them. An allocation is first turned into an extra
//    buffer of the region;
//  * side-effect free operations on the above, cloned into the reverse region.
//
// Anything else, like a value loaded from a buffer, is not supported.
class JITRegionCacheRematerializer {
public:
  JITRegionCacheRematerializer(PatternRewriter &rewriter, JITRegionOp fwd,
                               JITRegionOp rev, Block *initBlock)
      : rewriter(rewriter), fwd(fwd), rev(rev), initBlock(initBlock) {}

  FailureOr<Value> materialize(Value v) {
    if (Value known = mapping.lookup(v))
      return known;

    FailureOr<Value> result = materializeUncached(v);
    if (succeeded(result))
      mapping[v] = *result;
    return result;
  }

  // Turns the exported allocations into buffers of a rebuilt forward region
  // and pushes the contents of the exported buffers for the reverse region.
  // Returns the new forward region.
  JITRegionOp finalize() {
    if (!revNewInputs.empty()) {
      rewriter.setInsertionPoint(rev);
      SmallVector<Value> inputs(rev.getInputs());
      llvm::append_range(inputs, revNewInputs);
      SmallVector<Attribute> layouts = getInputLayouts(rev);
      llvm::append_range(layouts, revNewLayouts);
      auto newRev = rebuild(rev, inputs, layouts);
      replaceJITRegion(rewriter, rev, newRev);
      rev = newRev;
    }

    // The exported buffers are read from the results of a region with one
    // result per input.
    if (!newBuffers.empty() || !fwd.getOutputOperandAliases().empty()) {
      // The allocations are uninitialized: any contents will do.
      rewriter.setInsertionPoint(fwd);
      SmallVector<Value> inputs(fwd.getInputs());
      SmallVector<Attribute> layouts = getInputLayouts(fwd);
      for (Value alloc : newBuffers) {
        inputs.push_back(cast<AutoDiffTypeInterface>(getTensorType(alloc))
                             .createNullValue(rewriter, alloc.getLoc()));
        layouts.push_back(getDefaultLayout(rewriter, alloc.getType()));
      }

      auto newFwd = rebuild(fwd, inputs, layouts);
      Block *body = newFwd.getBodyBlock();
      for (Value alloc : newBuffers) {
        BlockArgument arg = body->addArgument(alloc.getType(), alloc.getLoc());
        Operation *allocOp = alloc.getDefiningOp();
        for (Operation *user : llvm::make_early_inc_range(alloc.getUsers()))
          if (hasSingleEffect<MemoryEffects::Free>(user))
            rewriter.eraseOp(user);
        rewriter.replaceAllUsesWith(alloc, arg);
        rewriter.eraseOp(allocOp);
      }
      replaceJITRegion(rewriter, fwd, newFwd);
      fwd = newFwd;
    }

    rewriter.setInsertionPointAfter(fwd);
    for (auto [cache, resultNumber] : exports)
      enzyme::PushOp::create(rewriter, fwd.getLoc(), cache,
                             fwd->getResult(resultNumber));
    return fwd;
  }

private:
  JITRegionOp rebuild(JITRegionOp region, ValueRange inputs,
                      ArrayRef<Attribute> layouts) {
    return rebuildJITRegion(rewriter, region, inputs, layouts);
  }

  Block *revBody() { return rev.getBodyBlock(); }

  // Insert after the previously materialized values, which the new one may
  // depend on.
  void setInsertionPointInReverse() {
    if (lastMaterialized)
      rewriter.setInsertionPointAfter(lastMaterialized);
    else
      rewriter.setInsertionPointToStart(revBody());
  }

  Value createCache(Type type, Location loc) {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(initBlock);
    return enzyme::InitOp::create(
        rewriter, loc, enzyme::CacheType::get(type.getContext(), type));
  }

  // Carries the final contents of a buffer of the forward region, a block
  // argument or an allocation, into a new buffer of the reverse region.
  FailureOr<Value> exportBuffer(Value buffer) {
    auto type = dyn_cast<MemRefType>(buffer.getType());
    if (!type || !type.hasStaticShape())
      return fwd.emitError()
             << "cannot carry a buffer of type " << buffer.getType()
             << " out of a jit_region for the reverse pass";

    // The reverse buffer has the layout of the forward one, so that the
    // kernels find the contents where they left them.
    unsigned resultNumber;
    if (auto arg = dyn_cast<BlockArgument>(buffer)) {
      resultNumber = arg.getArgNumber();
      revNewLayouts.push_back(getInputLayout(fwd, resultNumber));
    } else {
      resultNumber = fwd.getInputs().size() + newBuffers.size();
      newBuffers.push_back(buffer);
      revNewLayouts.push_back(getDefaultLayout(rewriter, type));
    }

    RankedTensorType tensorType = getTensorType(buffer);
    Value cache = createCache(tensorType, buffer.getLoc());
    exports.push_back({cache, resultNumber});

    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(rev);
    Value contents =
        enzyme::PopOp::create(rewriter, buffer.getLoc(), tensorType, cache);
    // The reverse region is rebuilt with the new input in finalize.
    revNewInputs.push_back(contents);
    return Value(revBody()->addArgument(type, buffer.getLoc()));
  }

  FailureOr<Value> materializeUncached(Value v) {
    OpBuilder::InsertionGuard guard(rewriter);
    Operation *def = v.getDefiningOp();

    if (isDefinedAbove(fwd, v)) {
      if (def && def->hasTrait<OpTrait::ConstantLike>()) {
        setInsertionPointInReverse();
        lastMaterialized = rewriter.clone(*def);
        return lastMaterialized->getResult(cast<OpResult>(v).getResultNumber());
      }
      Value cache = createCache(v.getType(), v.getLoc());
      rewriter.setInsertionPoint(fwd);
      enzyme::PushOp::create(rewriter, v.getLoc(), cache, v);
      rewriter.setInsertionPoint(rev);
      return enzyme::PopOp::create(rewriter, v.getLoc(), v.getType(), cache)
          .getResult();
    }

    if (auto arg = dyn_cast<BlockArgument>(v)) {
      if (arg.getOwner() != fwd.getBodyBlock())
        return fwd.emitError()
               << "cannot carry a block argument of a nested region out of a "
                  "jit_region for the reverse pass";
      if (auto shadows = fwd->getAttrOfType<DenseI64ArrayAttr>(
              kJITRegionShadowsAttrName)) {
        auto it = llvm::find(shadows.asArrayRef(), arg.getArgNumber());
        if (it != shadows.asArrayRef().end())
          return Value(
              revBody()->getArgument(it - shadows.asArrayRef().begin()));
      }
      return exportBuffer(arg);
    }

    if (def->getBlock() != fwd.getBodyBlock())
      return def->emitError() << "cannot carry a value defined in a nested "
                                 "region out of a jit_region for the reverse "
                                 "pass";

    if (isa<MemRefType>(v.getType()) &&
        hasEffect<MemoryEffects::Allocate>(def, v))
      return exportBuffer(v);

    if (isMemoryEffectFree(def) && def->getNumRegions() == 0) {
      IRMapping operands;
      for (Value operand : def->getOperands()) {
        FailureOr<Value> revOperand = materialize(operand);
        if (failed(revOperand))
          return failure();
        operands.map(operand, *revOperand);
      }
      setInsertionPointInReverse();
      Operation *clone = rewriter.clone(*def, operands);
      for (auto [result, cloned] :
           llvm::zip_equal(def->getResults(), clone->getResults()))
        mapping[result] = cloned;
      lastMaterialized = clone;
      return clone->getResult(cast<OpResult>(v).getResultNumber());
    }

    return def->emitError() << "cannot carry a value computed by an operation "
                               "with side effects out of a jit_region for the "
                               "reverse pass";
  }

  PatternRewriter &rewriter;
  JITRegionOp fwd;
  JITRegionOp rev;
  // The new caches are initialized at the entry of the function, like the
  // existing ones.
  Block *initBlock;

  DenseMap<Value, Value> mapping;
  Operation *lastMaterialized = nullptr;

  // Allocations of the forward region to turn into extra buffers.
  SmallVector<Value> newBuffers;
  // Contents of the forward buffers to pass to the reverse region.
  SmallVector<Value> revNewInputs;
  // Their layouts, null if the forward region has no operand_layouts.
  SmallVector<Attribute> revNewLayouts;
  // The caches of the forward results passed to the reverse region.
  SmallVector<std::pair<Value, unsigned>> exports;
};

struct JITRegionOpEnzymeOpsRemover
    : public EnzymeOpsRemoverOpInterface::ExternalModel<
          JITRegionOpEnzymeOpsRemover, JITRegionOp> {

  LogicalResult removeEnzymeOps(Operation *op,
                                PatternRewriter &rewriter) const {
    auto fwd = cast<JITRegionOp>(op);
    Block *body = fwd.getBodyBlock();

    // Gradients of values of the region are localized to the reverse region;
    // one outliving it cannot be updated from within.
    WalkResult walk = fwd.walk([&](Operation *inner) {
      Value gradient;
      if (auto set = dyn_cast<enzyme::SetOp>(inner))
        gradient = set.getGradient();
      else if (auto get = dyn_cast<enzyme::GetOp>(inner))
        gradient = get.getGradient();
      if (gradient && !fwd.getBody().isAncestor(gradient.getParentRegion())) {
        inner->emitError()
            << "cannot update a gradient defined outside of a jit_region";
        return WalkResult::interrupt();
      }
      // Pushes in nested regions should have been moved out by their own
      // remover.
      if (isa<enzyme::PushOp>(inner) && inner->getBlock() != body) {
        inner->emitError() << "cannot move a nested cache out of a jit_region";
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (walk.wasInterrupted())
      return failure();

    SmallVector<enzyme::PushOp> pushes;
    for (Operation &inner : *body)
      if (auto push = dyn_cast<enzyme::PushOp>(&inner))
        pushes.push_back(push);

    JITRegionOp rev = nullptr;
    for (enzyme::PushOp push : pushes) {
      CacheInfo info(push.getCache());
      auto popRegion = info.popOp->getParentOfType<JITRegionOp>();
      if (!popRegion || (rev && rev != popRegion))
        return push->emitError() << "cache pushed in a jit_region must be "
                                    "popped in its reverse jit_region";
      rev = popRegion;
    }

    if (rev) {
      JITRegionCacheRematerializer rematerializer(
          rewriter, fwd, rev,
          &fwd->getParentOfType<FunctionOpInterface>()
               .getFunctionBody()
               .front());

      for (enzyme::PushOp push : pushes) {
        CacheInfo info(push.getCache());
        Value pushed = push.getValue();

        // A value defined above is still available after the region.
        if (isDefinedAbove(fwd, pushed)) {
          rewriter.moveOpBefore(push, fwd);
          rewriter.moveOpBefore(info.popOp, rev);
          continue;
        }

        FailureOr<Value> revValue = rematerializer.materialize(pushed);
        if (failed(revValue))
          return failure();

        // A buffer of the reverse region is freed with it.
        if (auto arg = dyn_cast<BlockArgument>(*revValue);
            arg && arg.getOwner() == rev.getBodyBlock()) {
          for (Operation *user :
               llvm::make_early_inc_range(info.popOp->getUsers()))
            if (hasSingleEffect<MemoryEffects::Free>(user))
              rewriter.eraseOp(user);
        }

        rewriter.replaceAllUsesWith(info.popOp.getResult(), *revValue);
        rewriter.eraseOp(info.popOp);
        rewriter.eraseOp(push);
        if (info.initOp && info.initOp->use_empty())
          rewriter.eraseOp(info.initOp);
      }

      fwd = rematerializer.finalize();
      body = fwd.getBodyBlock();
    }

    auto shadows =
        fwd->getAttrOfType<DenseI64ArrayAttr>(kJITRegionShadowsAttrName);
    if (!shadows)
      return success();

    // What is left of the uses of the shadow buffers is dead code.
    auto findShadowUse = [&]() -> Operation * {
      for (int64_t input : shadows.asArrayRef())
        if (!body->getArgument(input).use_empty())
          return *body->getArgument(input).getUsers().begin();
      return nullptr;
    };
    bool changed = findShadowUse();
    while (changed) {
      SmallVector<Operation *> dead;
      fwd.getBody().walk([&](Operation *inner) {
        if (isOpTriviallyDead(inner))
          dead.push_back(inner);
      });
      for (Operation *inner : dead)
        rewriter.eraseOp(inner);
      changed = !dead.empty();
    }
    if (Operation *user = findShadowUse())
      return user->emitError() << "shadow of a jit_region buffer is used in "
                                  "the augmented primal";

    // Drop the shadow buffers.
    BitVector isShadow(fwd.getInputs().size());
    for (int64_t input : shadows.asArrayRef())
      isShadow.set(input);
    SmallVector<Value> inputs;
    SmallVector<Attribute> layouts;
    for (auto [i, input] : llvm::enumerate(fwd.getInputs()))
      if (!isShadow.test(i)) {
        inputs.push_back(input);
        layouts.push_back(getInputLayout(fwd, i));
      }
    body->eraseArguments(isShadow);

    // The augmented region has one result per input.
    rewriter.setInsertionPoint(fwd);
    auto stripped = rebuildJITRegion(rewriter, fwd, inputs, layouts);
    stripped->removeAttr(kJITRegionShadowsAttrName);
    unsigned next = 0;
    for (auto [i, result] : llvm::enumerate(fwd->getResults()))
      if (!isShadow.test(i))
        rewriter.replaceAllUsesWith(result, stripped->getResult(next++));
    rewriter.eraseOp(fwd);
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

    JITRegionOp::attachInterface<JITRegionOpADDataFlow>(*context);
    JITRegionOp::attachInterface<JITRegionOpInterfaceReverse>(*context);
    JITRegionOp::attachInterface<JITRegionOpEnzymeOpsRemover>(*context);

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
