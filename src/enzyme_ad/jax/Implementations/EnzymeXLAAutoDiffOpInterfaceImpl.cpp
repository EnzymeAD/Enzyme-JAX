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

// The result of a jit_region rebuilt by rebuildJITRegion standing for the
// result `result` of the original region.
static OpResult getRebuiltResult(JITRegionOp rebuilt, OpResult result) {
  auto region = cast<JITRegionOp>(result.getOwner());
  return rebuilt->getResult(
      region.getAliasedInputIndex(result.getResultNumber()));
}

// Rebuilds `region` with the inputs `inputs`, the inputs of `region` followed
// by new ones, and one result per input.
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
  // The new inputs, and the results read from buffers no result of `region`
  // was read from, have no attributes.
  auto empty = rewriter.getDictionaryAttr({});
  if (ArrayAttr operandAttrs = region.getOperandAttrsAttr()) {
    SmallVector<Attribute> newAttrs(operandAttrs.getValue());
    newAttrs.resize(inputs.size(), empty);
    attrs.set(region.getOperandAttrsAttrName(),
              rewriter.getArrayAttr(newAttrs));
  }
  if (ArrayAttr resultAttrs = region.getResultAttrsAttr()) {
    SmallVector<Attribute> newAttrs(inputs.size(), empty);
    for (OpResult result : region.getResults())
      newAttrs[region.getAliasedInputIndex(result.getResultNumber())] =
          resultAttrs[result.getResultNumber()];
    attrs.set(region.getResultAttrsAttrName(), rewriter.getArrayAttr(newAttrs));
  }
  auto newRegion = JITRegionOp::create(rewriter, region.getLoc(),
                                       inputs.getTypes(), inputs, attrs);
  rewriter.inlineRegionBefore(region.getBody(), newRegion.getBody(),
                              newRegion.getBody().end());
  return newRegion;
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

  LogicalResult createShadowValues(Operation *op, OpBuilder &builder,
                                   MGradientUtilsReverse *gutils) const {
    auto region = cast<JITRegionOp>(op);
    auto newRegion = cast<JITRegionOp>(gutils->getNewFromOriginal(op));

    SmallVector<BlockArgument> activeBuffers = getActiveBuffers(region, gutils);
    if (activeBuffers.empty())
      return success();

    SmallVector<Value> inputs =
        llvm::map_to_vector(region.getInputs(), [&](Value input) {
          return gutils->getNewFromOriginal(input);
        });
    SmallVector<Attribute> layouts = getInputLayouts(newRegion);

    Block *newBB = newRegion.getBodyBlock();
    for (auto [i, arg] : llvm::enumerate(activeBuffers)) {
      BlockArgument shadow = newBB->getArgument(arg.getArgNumber() + i + 1);
      layouts.push_back(getInputLayout(newRegion, arg.getArgNumber()));
      inputs.push_back(cast<AutoDiffTypeInterface>(getTensorType(shadow))
                           .createNullValue(builder, arg.getLoc()));
    }

    IRRewriter rewriter(builder);
    auto augmented = rebuildJITRegion(rewriter, newRegion, inputs, layouts);

    // Each shadow input points to the input of the reverse region holding the
    // same shadow: the reverse region has one input per active buffer, in
    // order.
    SmallVector<Attribute> operandAttrs(augmented.getNumOperands(),
                                        rewriter.getDictionaryAttr({}));
    if (ArrayAttr attrs = augmented.getOperandAttrsAttr())
      llvm::copy(attrs.getValue(), operandAttrs.begin());
    unsigned numPrimalInputs = region.getInputs().size();
    for (unsigned i = 0, e = activeBuffers.size(); i < e; ++i) {
      NamedAttrList attrs(
          cast<DictionaryAttr>(operandAttrs[numPrimalInputs + i]));
      attrs.set("enzymexla.reverse_operand_index",
                rewriter.getI64IntegerAttr(i));
      operandAttrs[numPrimalInputs + i] = attrs.getDictionary(op->getContext());
    }
    augmented.setOperandAttrsAttr(rewriter.getArrayAttr(operandAttrs));

    gutils->originalToNewFnOps[op] = augmented;
    gutils->erase(newRegion);
    return success();
  }
};

struct JITRegionOpEnzymeOpsRemover
    : public EnzymeOpsRemoverOpInterface::ExternalModel<
          JITRegionOpEnzymeOpsRemover, JITRegionOp> {

  LogicalResult removeEnzymeOps(Operation *op,
                                PatternRewriter &rewriter) const {
    auto fwd = cast<JITRegionOp>(op);
    JITRegionOp rev = nullptr;
    Block *body = fwd.getBodyBlock();

    SmallVector<Value> gradients;

    SmallVector<CacheInfo> caches;

    for (Operation &inner : *body) {
      if (auto push = dyn_cast<enzyme::PushOp>(&inner)) {
        CacheInfo info(push.getCache());

        auto popContainer = dyn_cast<JITRegionOp>(info.popOp->getParentOp());

        if (!popContainer || (rev && popContainer != rev)) {
          push->emitError("push and pop are not in jit regions both.");
          return failure();
        }

        rev = popContainer;
        caches.push_back(info);
        continue;
      }

      if (isa<GetOp, SetOp>(&inner)) {
        gradients.push_back(inner.getOperand(0));
      }
    }

    if (!gradients.empty())
      return op->emitError("cannot handle set / get in jit region yet.");

    if (caches.empty())
      return success();

    // A shadow buffer of the augmented region stands for the buffer of the
    // reverse region holding the same shadow.
    IRMapping fwdrevmap;
    Block *revBody = rev.getBodyBlock();
    if (auto operandAttrs = fwd.getOperandAttrs()) {
      for (auto [attr, arg] :
           llvm::zip_equal(*operandAttrs, body->getArguments())) {
        auto index = cast<DictionaryAttr>(attr).getAs<IntegerAttr>(
            "enzymexla.reverse_operand_index");
        if (!index)
          continue;
        fwdrevmap.map(arg, revBody->getArgument(index.getInt()));
      }
    }

    rewriter.setInsertionPointToStart(revBody);
    minCutCache(body, revBody, caches, rewriter, fwdrevmap);

    // TODO: remove the unused shadows in the primal
    // if (!caches.empty())
    //   return op->emitError("could not remove all caches from jit region op");

    return cacheBuffersAsTensors(fwd, rev, caches, rewriter);
  }

  // Whether `op` allocates a statically shaped buffer, synchronously.
  static bool isStaticAlloc(Operation *op) {
    if (isa<memref::AllocOp, memref::AllocaOp>(op))
      return op->getNumOperands() == 0;
    if (auto alloc = dyn_cast<gpu::AllocOp>(op))
      return op->getNumOperands() == 0 && !alloc.getAsyncToken() &&
             !alloc.getHostShared();
    return false;
  }

  // Erases the deallocations of `buffer`.
  static void eraseDeallocs(Value buffer, PatternRewriter &rewriter) {
    for (Operation *user : llvm::make_early_inc_range(buffer.getUsers())) {
      if (isa<memref::DeallocOp>(user))
        rewriter.eraseOp(user);
      else if (auto dealloc = dyn_cast<gpu::DeallocOp>(user);
               dealloc && !dealloc.getAsyncToken())
        rewriter.eraseOp(user);
    }
  }

  // A buffer does not outlive the jit_region it lives in, so a cache of a
  // buffer pushed in the augmented region and popped in the reverse one is
  // carried between the two as a tensor, the buffer becoming a new buffer of
  // both regions:
  //
  //   jit_region(%x) { ^bb0(%m): %b = memref.alloc(); push(%c, %b) }
  //   jit_region(%dr) { ^bb0(%dm): %b = pop(%c); use(%b) }
  //
  // becomes
  //
  //   %r:2 = jit_region(%x, %zero) { ^bb0(%m, %b): }
  //   push(%c', %r#1)
  //   %t = pop(%c')
  //   jit_region(%dr, %t) { ^bb0(%dm, %b): use(%b) }
  //
  // A cache of a buffer holds the buffer, not its contents at the push: the
  // pop sees the contents the buffer has when the region exits, which is what
  // the result read from the new buffer holds.
  static LogicalResult cacheBuffersAsTensors(JITRegionOp fwd, JITRegionOp rev,
                                             ArrayRef<CacheInfo> caches,
                                             PatternRewriter &rewriter) {
    Block *body = fwd.getBodyBlock();
    Block *revBody = rev.getBodyBlock();

    SmallVector<CacheInfo> bufferCaches;
    for (CacheInfo info : caches) {
      Value buffer = info.pushedValue();
      auto type = dyn_cast<MemRefType>(buffer.getType());
      if (!type)
        continue;
      if (info.pushOp->getBlock() != body || buffer.getParentBlock() != body ||
          !rev.getBody().isAncestor(info.popOp->getParentRegion()))
        return info.pushOp->emitError(
            "cannot carry a cache of a buffer pushed in a nested block of a "
            "jit_region");
      if (!type.hasStaticShape())
        return info.pushOp->emitError(
            "cannot carry a cache of a dynamically shaped buffer out of a "
            "jit_region");
      bufferCaches.push_back(info);
    }
    if (bufferCaches.empty())
      return success();

    // The buffer of the augmented region holding each cached buffer, and
    // where it was pushed.
    SmallVector<unsigned> fwdBuffers;
    SmallVector<Location> pushLocs;
    SmallVector<Value> fwdInputs(fwd.getInputs());
    SmallVector<Attribute> fwdLayouts = getInputLayouts(fwd);
    for (CacheInfo info : bufferCaches) {
      Value buffer = info.pushedValue();
      Location loc = buffer.getLoc();
      pushLocs.push_back(info.pushOp.getLoc());
      rewriter.eraseOp(info.pushOp);

      // A buffer of the region is already read by its result.
      if (auto arg = dyn_cast<BlockArgument>(buffer)) {
        fwdBuffers.push_back(arg.getArgNumber());
        continue;
      }

      fwdBuffers.push_back(fwdInputs.size());
      RankedTensorType tensorType = getTensorType(buffer);
      {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPoint(fwd);
        fwdInputs.push_back(cast<AutoDiffTypeInterface>(tensorType)
                                .createNullValue(rewriter, loc));
      }
      fwdLayouts.push_back(getDefaultLayout(rewriter, tensorType));
      BlockArgument arg = body->addArgument(buffer.getType(), loc);

      // An allocation of the region becomes the new buffer; any other buffer
      // is copied to it once the region is done with it.
      Operation *alloc = buffer.getDefiningOp();
      if (alloc && isStaticAlloc(alloc)) {
        eraseDeallocs(buffer, rewriter);
        rewriter.replaceAllUsesWith(buffer, arg);
        rewriter.eraseOp(alloc);
      } else {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPoint(body->getTerminator());
        memref::CopyOp::create(rewriter, loc, buffer, arg);
      }
    }

    rewriter.setInsertionPoint(fwd);
    JITRegionOp newFwd = rebuildJITRegion(rewriter, fwd, fwdInputs, fwdLayouts);
    for (OpResult result : fwd.getResults())
      rewriter.replaceAllUsesWith(result, getRebuiltResult(newFwd, result));
    rewriter.eraseOp(fwd);

    // Each cached buffer is pushed once the augmented region exits, and popped
    // before the reverse one is entered, in the opposite order.
    SmallVector<Value> inits;
    rewriter.setInsertionPointAfter(newFwd);
    for (auto [info, index, loc] :
         llvm::zip_equal(bufferCaches, fwdBuffers, pushLocs)) {
      Value tensor = newFwd->getResult(index);
      enzyme::InitOp init;
      {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPoint(info.initOp);
        init = enzyme::InitOp::create(
            rewriter, loc,
            enzyme::CacheType::get(rewriter.getContext(), tensor.getType()));
      }
      enzyme::PushOp::create(rewriter, loc, init, tensor);
      inits.push_back(init);
    }

    rewriter.setInsertionPoint(rev);
    unsigned numRevInputs = rev.getInputs().size();
    SmallVector<Value> revInputs(rev.getInputs());
    SmallVector<Attribute> revLayouts = getInputLayouts(rev);
    revInputs.resize(numRevInputs + bufferCaches.size());
    for (unsigned i = bufferCaches.size(); i-- > 0;) {
      Type type = cast<enzyme::CacheType>(inits[i].getType()).getType();
      revInputs[numRevInputs + i] = enzyme::PopOp::create(
          rewriter, bufferCaches[i].popOp.getLoc(), type, inits[i]);
    }

    // The pop of a cached buffer in the reverse region becomes its new buffer.
    for (CacheInfo info : bufferCaches) {
      BlockArgument arg = revBody->addArgument(info.popOp.getType(),
                                               info.popOp.getLoc());
      revLayouts.push_back(getDefaultLayout(rewriter, getTensorType(arg)));
      // The region owns its buffers: the reverse region no longer frees the
      // buffer the augmented one allocated.
      eraseDeallocs(info.popOp.getResult(), rewriter);
      rewriter.replaceAllUsesWith(info.popOp.getResult(), arg);
      rewriter.eraseOp(info.popOp);
      rewriter.eraseOp(info.initOp);
    }

    JITRegionOp newRev = rebuildJITRegion(rewriter, rev, revInputs, revLayouts);
    for (OpResult result : rev.getResults())
      rewriter.replaceAllUsesWith(result, getRebuiltResult(newRev, result));
    rewriter.eraseOp(rev);
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

    // Register batching interfaces
    JITCallOp::attachInterface<SHLOGenericBatchOpInterface<JITCallOp>>(
        *context);

    JITRegionOp::attachInterface<JITRegionOpADDataFlow>(*context);
    JITRegionOp::attachInterface<JITRegionOpInterfaceReverse>(*context);
    JITRegionOp::attachInterface<JITRegionOpEnzymeOpsRemover>(*context);

    context->loadDialect<stablehlo::StablehloDialect>();
  });
}
