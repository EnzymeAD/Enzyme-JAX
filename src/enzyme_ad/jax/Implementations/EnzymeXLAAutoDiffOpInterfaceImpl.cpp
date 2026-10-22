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

#include "Enzyme/MLIR/Dialect/Ops.h"
#include "Enzyme/MLIR/Implementations/CoreDialectsAutoDiffImplementations.h"
#include "Enzyme/MLIR/Interfaces/AutoDiffTypeInterface.h"
#include "Enzyme/MLIR/Interfaces/AutoDiffOpInterface.h"
#include "Enzyme/MLIR/Interfaces/GradientUtils.h"
#include "Enzyme/MLIR/Interfaces/GradientUtilsReverse.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/IRMapping.h"
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

#include <numeric>

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
          enzyme::YieldOp::create(bodyBuilder, repFor->getLoc(), ValueRange());
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


//===----------------------------------------------------------------------===//
// enzymexla.fixed_point
//===----------------------------------------------------------------------===//
//
// Reverse mode is implicit differentiation at the converged state
// (Christianson 1994; Tapenade's `$AD FP-LOOP`; LLVM Enzyme's
// `__enzyme_fixed_point`). The iterations are not taped: only the state the
// loop ended in, z*, is kept, and the adjoint of one step is iterated there,
//
//   d_0 = zbar,   d_{k+1} = phi_z(z*, x)^T d_k,   xbar += phi_x(z*, x)^T d_k,
//
// so xbar accumulates phi_x^T (I - phi_z^T)^{-1} zbar, the derivative the
// implicit function theorem gives for z*(x). The initial guess gets no
// adjoint, and neither the trip count nor the iterates are needed.
//
// The carried values are classified by how the body yields them:
//   - a value yielded unchanged is a parameter threaded through the loop: its
//     result's adjoint flows straight to its operand, and phi_x^T d is added
//     to the operand on every pass;
//   - an inactive value (an iteration counter) takes no part;
//   - every other active value is state: part of d and of the stop rule.
//
// The stop rule is LLVM Enzyme's. After each pass cumul = ||d||^2 over the
// state, in f64. Iterate again while cumul > reduction * ref, with ref the
// value of cumul after the first pass, unless cumul is zero, cumul grew after
// more than five passes, or max_iters passes have run (0 for no limit).

static Value fixedPointConstI64(OpBuilder &b, Location loc, int64_t v) {
  auto ty = RankedTensorType::get({}, b.getI64Type());
  return stablehlo::ConstantOp::create(b, loc, ty,
                                       cast<ElementsAttr>(makeAttr(ty, v)));
}

static Value fixedPointConstF64(OpBuilder &b, Location loc, double v) {
  auto ty = RankedTensorType::get({}, b.getF64Type());
  return stablehlo::ConstantOp::create(b, loc, ty,
                                       cast<ElementsAttr>(makeAttr(ty, v)));
}

static Value fixedPointCompare(OpBuilder &b, Location loc, Value l, Value r,
                               stablehlo::ComparisonDirection dir) {
  return stablehlo::CompareOp::create(b, loc, l, r, dir);
}

// sum(|g|^2) over a float or complex tensor, as a tensor<f64>.
static Value fixedPointSquaredNorm(OpBuilder &b, Location loc, Value g) {
  if (isa<ComplexType>(cast<RankedTensorType>(g.getType()).getElementType()))
    g = stablehlo::AbsOp::create(b, loc, g);
  auto ty = cast<RankedTensorType>(g.getType());
  Type f64 = b.getF64Type();
  if (ty.getElementType() != f64) {
    ty = ty.clone(f64);
    g = stablehlo::ConvertOp::create(b, loc, ty, g);
  }
  Value sq = stablehlo::MulOp::create(b, loc, g, g);
  if (ty.getRank() == 0)
    return sq;
  SmallVector<int64_t> dims(ty.getRank());
  std::iota(dims.begin(), dims.end(), 0);
  auto resTy = RankedTensorType::get({}, f64);
  Value zero = fixedPointConstF64(b, loc, 0.0);
  auto red = stablehlo::ReduceOp::create(b, loc, TypeRange(resTy),
                                         ValueRange(sq), ValueRange(zero), dims);
  Block &body = red.getBody().emplaceBlock();
  body.addArgument(resTy, loc);
  body.addArgument(resTy, loc);
  OpBuilder bb(b.getContext());
  bb.setInsertionPointToEnd(&body);
  Value add = stablehlo::AddOp::create(bb, loc, body.getArgument(0),
                                       body.getArgument(1));
  stablehlo::ReturnOp::create(bb, loc, ValueRange(add));
  return red.getResult(0);
}

static bool isFixedPointStateType(Type t) {
  auto ty = dyn_cast<RankedTensorType>(t);
  if (!ty)
    return false;
  Type elem = ty.getElementType();
  if (auto c = dyn_cast<ComplexType>(elem))
    elem = c.getElementType();
  return isa<FloatType>(elem);
}

// The values the body reads from above the loop. The replay of one step in
// the reverse pass needs them, and so do the reverse rules of the body's ops,
// which fetch their primal operands through originalToNewFn: a value defined
// in an enclosing loop's forward body is not available there, so they are
// cached along with the state. Deterministic for a fixed IR, which is what
// lets cacheValues and createReverseModeAdjoint agree on the cache layout.
static SmallVector<Value> fixedPointOutsideRefs(FixedPointOp op) {
  SetVector<Value> refs;
  getUsedValuesDefinedAbove(op.getBody(), refs);
  return SmallVector<Value>(refs.begin(), refs.end());
}

// Drops the entries `mapping` holds for the block arguments inside `op`'s
// regions, so a clone gets its own: Region::cloneInto reuses an existing
// entry rather than materializing a fresh argument.
static void forgetNestedBlockArgs(Operation *op, IRMapping &mapping) {
  for (Region &region : op->getRegions())
    for (Block &block : region) {
      for (BlockArgument arg : block.getArguments())
        mapping.erase(Value(arg));
      for (Operation &nested : block)
        forgetNestedBlockArgs(&nested, mapping);
    }
}

static bool isDefinedInBlock(Block *b, Value v) {
  Block *owner = isa<BlockArgument>(v) ? cast<BlockArgument>(v).getOwner()
                                       : v.getDefiningOp()->getBlock();
  for (; owner; owner = owner->getParentOp()
                            ? owner->getParentOp()->getBlock()
                            : nullptr)
    if (owner == b)
      return true;
  return false;
}

// Clones the body (without its terminator) at the builder, under `mapping`,
// and points originalToNewFn at the clone for everything the body defines, so
// the body ops' reverse rules fetch their primal operands from the replay.
static void replayFixedPointBody(OpBuilder &builder, Block *body,
                                 MGradientUtilsReverse *gutils,
                                 IRMapping &mapping) {
  for (Operation &op : body->without_terminator()) {
    forgetNestedBlockArgs(&op, mapping);
    builder.clone(op, mapping);
  }
  for (auto [oldVal, newVal] : mapping.getValueMap())
    if (isDefinedInBlock(body, oldVal))
      gutils->originalToNewFn.map(oldVal, newVal);
  for (auto [oldBlock, newBlock] : mapping.getBlockMap()) {
    gutils->originalToNewFn.map(oldBlock, newBlock);
    for (auto [oldArg, newArg] :
         llvm::zip(oldBlock->getArguments(), newBlock->getArguments()))
      gutils->originalToNewFn.map(oldArg, newArg);
  }
  for (auto [oldOp, newOp] : mapping.getOperationMap())
    gutils->originalToNewFnOps[oldOp] = newOp;
}

// A cache a body op's reverse rule asks for while the step is being replayed
// is initialized before the adjoint loop rather than in the enclosing
// function, for as long as this is alive.
class FixedPointCacheCreatorGuard {
public:
  FixedPointCacheCreatorGuard(MGradientUtilsReverse *gutils, Operation *anchor)
      : gutils(gutils) {
    OpBuilder cacheBuilder(anchor);
    Location loc = anchor->getLoc();
    hook = [cacheBuilder, loc](Type t) mutable {
      Value cache = enzyme::InitOp::create(cacheBuilder, loc, t);
      return std::make_pair(cache, cache);
    };
    gutils->registerCacheCreatorHook(hook);
  }
  ~FixedPointCacheCreatorGuard() { gutils->deregisterCacheCreatorHook(hook); }
  FixedPointCacheCreatorGuard(const FixedPointCacheCreatorGuard &) = delete;
  FixedPointCacheCreatorGuard &
  operator=(const FixedPointCacheCreatorGuard &) = delete;

private:
  MGradientUtilsReverse *gutils;
  std::function<std::pair<Value, Value>(Type)> hook;
};

struct FixedPointOpRev
    : public ReverseAutoDiffOpInterface::ExternalModel<FixedPointOpRev,
                                                       FixedPointOp> {
  // Augmented primal: the loop runs untouched; afterwards one push per result
  // (the converged state) and one per value the body reads from above.
  // Cache layout: [ every result | outside refs ].
  SmallVector<Value> cacheValues(Operation *orig,
                                 MGradientUtilsReverse *gutils) const {
    auto op = cast<FixedPointOp>(orig);
    Operation *newOp = gutils->getNewFromOriginal(orig);
    OpBuilder b(newOp);
    b.setInsertionPointAfter(newOp);
    SmallVector<Value> caches;
    for (Value res : op->getResults())
      caches.push_back(
          gutils->initAndPushCache(gutils->getNewFromOriginal(res), b));
    for (Value ref : fixedPointOutsideRefs(op))
      caches.push_back(
          gutils->initAndPushCache(gutils->getNewFromOriginal(ref), b));
    return caches;
  }

  LogicalResult createReverseModeAdjoint(Operation *orig, OpBuilder &builder,
                                         MGradientUtilsReverse *gutils,
                                         SmallVector<Value> caches) const {
    using stablehlo::ComparisonDirection;
    auto op = cast<FixedPointOp>(orig);
    Location loc = op.getLoc();
    Block *origCond = &op.getCond().front();
    Block *origBody = &op.getBody().front();
    Operation *term = origBody->getTerminator();
    unsigned n = op->getNumOperands();

    if (gutils->width != 1)
      return op.emitError(
          "fixed-point loops are not supported in vector (batched) mode");

    // Carried value i takes part when its operand, region arguments or result
    // is active (a loop that never runs returns its operands).
    SmallVector<unsigned> stateIdx, passIdx;
    for (unsigned i = 0; i < n; ++i) {
      bool active = !gutils->isConstantValue(op->getOperand(i)) ||
                    !gutils->isConstantValue(origCond->getArgument(i)) ||
                    !gutils->isConstantValue(origBody->getArgument(i)) ||
                    !gutils->isConstantValue(op->getResult(i));
      if (!active)
        continue;
      bool passthrough = term->getOperand(i) == origBody->getArgument(i);
      (passthrough ? passIdx : stateIdx).push_back(i);
    }
    for (unsigned i : stateIdx)
      if (!isFixedPointStateType(op->getResult(i).getType()))
        return op.emitError("fixed-point state must be float or complex "
                            "tensors, got ")
               << op->getResult(i).getType();

    SmallVector<Value> outside = fixedPointOutsideRefs(op);
    assert(caches.size() == n + outside.size() &&
           "fixed-point cache layout mismatch");
    SmallVector<Value> converged, outsideVals;
    for (unsigned i = 0; i < n; ++i)
      converged.push_back(gutils->popCache(caches[i], builder));
    for (unsigned j = 0; j < outside.size(); ++j)
      outsideVals.push_back(gutils->popCache(caches[n + j], builder));
    // Every lane has stopped at the fixed point; the step to differentiate is
    // the one each lane takes while still active.
    if (auto mask = op.getMaskAttr()) {
      auto maskTy = cast<RankedTensorType>(op->getOperand(mask.getInt()).getType());
      converged[mask.getInt()] = stablehlo::ConstantOp::create(
          builder, loc, maskTy,
          SplatElementsAttr::get(maskTy, builder.getBoolAttr(true)));
    }

    // A parameter threaded through the loop is its own result.
    for (unsigned i : passIdx) {
      Value res = op->getResult(i);
      if (gutils->isConstantValue(res))
        continue;
      Value g = gutils->diffe(res, builder);
      gutils->zeroDiffe(res, builder);
      if (!gutils->isConstantValue(op->getOperand(i)))
        gutils->addToDiffe(op->getOperand(i), g, builder);
    }

    if (stateIdx.empty())
      return success();

    // The adjoint loop carries [ pass count | d per state | cumul, ref, prev ]:
    // cumul is ||d||^2 after the last pass, ref after the first, prev one pass
    // before the last.
    SmallVector<Value> inits{fixedPointConstI64(builder, loc, 0)};
    for (unsigned i : stateIdx) {
      Value res = op->getResult(i);
      inits.push_back(gutils->diffe(res, builder));
      if (!gutils->isConstantValue(res))
        gutils->zeroDiffe(res, builder);
    }
    inits.append(3, fixedPointConstF64(builder, loc, -1.0));
    unsigned cumulPos = 1 + stateIdx.size(), refPos = cumulPos + 1,
             prevPos = cumulPos + 2;

    auto revLoop = stablehlo::WhileOp::create(
        builder, loc, ValueRange(inits).getTypes(), inits);
    Block *cond = new Block(), *body = new Block();
    for (Value v : inits) {
      cond->addArgument(v.getType(), loc);
      body->addArgument(v.getType(), loc);
    }
    revLoop.getCond().push_back(cond);
    revLoop.getBody().push_back(body);

    {
      OpBuilder cb(cond, cond->end());
      Value iter = cond->getArgument(0);
      Value cumul = cond->getArgument(cumulPos);
      Value ref = cond->getArgument(refPos);
      Value prev = cond->getArgument(prevPos);
      Value isFirst =
          fixedPointCompare(cb, loc, iter, fixedPointConstI64(cb, loc, 0),
                            ComparisonDirection::EQ);
      Value positive =
          fixedPointCompare(cb, loc, cumul, fixedPointConstF64(cb, loc, 0.0),
                            ComparisonDirection::GT);
      Value threshold = stablehlo::MulOp::create(
          cb, loc, fixedPointConstF64(cb, loc, op.getReductionOrDefault()),
          ref);
      Value notReduced = fixedPointCompare(cb, loc, cumul, threshold,
                                           ComparisonDirection::GT);
      Value pastFive =
          fixedPointCompare(cb, loc, iter, fixedPointConstI64(cb, loc, 5),
                            ComparisonDirection::GT);
      Value grew =
          fixedPointCompare(cb, loc, cumul, prev, ComparisonDirection::GT);
      Value grown = stablehlo::AndOp::create(cb, loc, pastFive, grew);
      Value notGrown = stablehlo::NotOp::create(cb, loc, grown);
      Value progressing =
          stablehlo::AndOp::create(cb, loc, positive, notReduced);
      Value laterGo = stablehlo::AndOp::create(cb, loc, progressing, notGrown);
      Value go = stablehlo::OrOp::create(cb, loc, isFirst, laterGo);
      if (int64_t maxIters = op.getMaxItersOrDefault(); maxIters > 0) {
        Value capped = fixedPointCompare(
            cb, loc, iter, fixedPointConstI64(cb, loc, maxIters),
            ComparisonDirection::LT);
        go = stablehlo::AndOp::create(cb, loc, go, capped);
      }
      stablehlo::ReturnOp::create(cb, loc, ValueRange(go));
    }

    bool valid = true;
    {
      OpBuilder bb(body, body->end());
      Value iter = body->getArgument(0);

      // Replay one step at the converged state: the body's arguments stand
      // for z*, and the values it reads from above for their cached copies.
      // originalToNewFn points at those copies while the body is being
      // differentiated, and is restored afterwards.
      IRMapping mapping;
      for (unsigned i = 0; i < n; ++i)
        mapping.map(origBody->getArgument(i), converged[i]);
      SmallVector<std::pair<Value, Value>> saved;
      for (auto [ref, val] : llvm::zip_equal(outside, outsideVals)) {
        mapping.map(ref, val);
        saved.emplace_back(ref, gutils->originalToNewFn.lookupOrNull(ref));
        gutils->originalToNewFn.map(ref, val);
      }
      replayFixedPointBody(bb, origBody, gutils, mapping);

      // Every gradient slot a body op touches is local to one pass.
      for (Value arg : origBody->getArguments())
        if (!gutils->isConstantValue(arg))
          gutils->zeroDiffe(arg, bb);
      for (Operation &bodyOp : origBody->getOperations())
        for (Value res : bodyOp.getResults())
          if (!gutils->isConstantValue(res))
            gutils->zeroDiffe(res, bb);

      for (auto [j, i] : llvm::enumerate(stateIdx)) {
        Value yielded = term->getOperand(i);
        if (!gutils->isConstantValue(yielded))
          gutils->addToDiffe(yielded, body->getArgument(1 + j), bb);
      }

      {
        FixedPointCacheCreatorGuard cacheGuard(gutils, revLoop);
        for (auto it = std::next(origBody->rbegin()); it != origBody->rend();
             ++it)
          valid &= gutils->Logic.visitChild(&*it, bb, gutils).succeeded();
      }

      Value nextIter = stablehlo::AddOp::create(
          bb, loc, iter, fixedPointConstI64(bb, loc, 1));
      SmallVector<Value> yields{nextIter};
      Value cumul = fixedPointConstF64(bb, loc, 0.0);
      for (unsigned i : stateIdx) {
        Value arg = origBody->getArgument(i);
        Value g = gutils->diffe(arg, bb);
        if (!gutils->isConstantValue(arg))
          gutils->zeroDiffe(arg, bb);
        yields.push_back(g);
        Value sq = fixedPointSquaredNorm(bb, loc, g);
        cumul = stablehlo::AddOp::create(bb, loc, cumul, sq);
      }
      // phi_x^T d for the parameters threaded through the loop.
      for (unsigned i : passIdx) {
        Value arg = origBody->getArgument(i);
        if (gutils->isConstantValue(arg))
          continue;
        Value g = gutils->diffe(arg, bb);
        gutils->zeroDiffe(arg, bb);
        if (!gutils->isConstantValue(op->getOperand(i)))
          gutils->addToDiffe(op->getOperand(i), g, bb);
      }

      Value isFirst =
          fixedPointCompare(bb, loc, iter, fixedPointConstI64(bb, loc, 0),
                            ComparisonDirection::EQ);
      Value ref = stablehlo::SelectOp::create(bb, loc, isFirst, cumul,
                                              body->getArgument(refPos));
      yields.push_back(cumul);
      yields.push_back(ref);
      yields.push_back(body->getArgument(cumulPos));
      stablehlo::ReturnOp::create(bb, loc, yields);

      for (auto [ref, old] : saved) {
        if (old)
          gutils->originalToNewFn.map(ref, old);
        else
          gutils->originalToNewFn.erase(ref);
      }
    }

    // Nothing flows to the state's operands: the converged state does not
    // depend on the initial guess.
    return success(valid);
  }

  LogicalResult createShadowValues(Operation *, OpBuilder &,
                                   MGradientUtilsReverse *) const {
    return success();
  }
};

// Forward mode: the tangent still goes through every iteration, exactly as
// for stablehlo.while. Iterating the tangent at the converged state instead is
// a follow-up.
struct FixedPointOpFwd
    : public AutoDiffOpInterface::ExternalModel<FixedPointOpFwd, FixedPointOp> {
  LogicalResult createForwardModeTangent(Operation *orig, OpBuilder &builder,
                                         MGradientUtils *gutils) const {
    auto op = cast<FixedPointOp>(orig);

    llvm::SmallDenseSet<unsigned> operandPositionsToShadow;
    llvm::SmallDenseSet<unsigned> resultPositionsToShadow;
    for (auto &&[res, arg] :
         llvm::zip(op->getResults(), op.getBody().front().getArguments()))
      if (!gutils->isConstantValue(res) || !gutils->isConstantValue(arg)) {
        operandPositionsToShadow.insert(res.getResultNumber());
        resultPositionsToShadow.insert(res.getResultNumber());
      }

    auto res = mlir::enzyme::detail::controlFlowForwardHandler(
        op, builder, gutils, operandPositionsToShadow, resultPositionsToShadow);

    // The shadows follow their primals among the region arguments.
    for (auto reg : {&op.getCond(), &op.getBody()}) {
      auto nb = gutils->getNewFromOriginal(&reg->front());
      size_t curidx = 0;
      for (auto arg : reg->front().getArguments()) {
        curidx++;
        auto idx = arg.getArgNumber();
        if (resultPositionsToShadow.count(idx)) {
          if (gutils->isConstantValue(arg))
            nb->insertArgument(curidx,
                               cast<AutoDiffTypeInterface>(arg.getType())
                                   .getShadowType(gutils->width),
                               op.getLoc());
          curidx++;
        }
      }
    }
    return res;
  }
};

struct FixedPointOpCF
    : public ControlFlowAutoDiffOpInterface::ExternalModel<FixedPointOpCF,
                                                           FixedPointOp> {
  Operation *createWithShadows(Operation *op, OpBuilder &builder,
                               MGradientUtils *gutils, Operation *original,
                               ValueRange remappedOperands,
                               TypeRange rettys) const {
    // The shadows interleave with the carried values, so `mask` no longer
    // names the right position; only the reverse rule reads it.
    auto fp = FixedPointOp::create(builder, original->getLoc(), rettys,
                                   remappedOperands, original->getAttrs());
    fp.removeMaskAttr();
    return fp;
  }
};

// The yield of the body carries the shadows of the active carried values after
// their primals; the yield of the condition carries only its predicate.
struct FixedPointYieldFwd
    : public AutoDiffOpInterface::ExternalModel<FixedPointYieldFwd,
                                                enzymexla::YieldOp> {
  LogicalResult createForwardModeTangent(Operation *origTerminator,
                                         OpBuilder &builder,
                                         MGradientUtils *gutils) const {
    auto fp = cast<FixedPointOp>(origTerminator->getParentOp());
    bool inCond = origTerminator->getParentRegion() == &fp.getCond();
    Block &body = fp.getBody().front();

    SmallVector<Value> newOperands;
    for (OpOperand &operand : origTerminator->getOpOperands()) {
      newOperands.push_back(gutils->getNewFromOriginal(operand.get()));
      if (inCond)
        continue;
      unsigned i = operand.getOperandNumber();
      if (gutils->isConstantValue(fp->getResult(i)) &&
          gutils->isConstantValue(body.getArgument(i)))
        continue;
      if (!gutils->isConstantValue(operand.get())) {
        newOperands.push_back(gutils->invertPointerM(operand.get(), builder));
      } else {
        Type shadowTy = cast<AutoDiffTypeInterface>(operand.get().getType())
                            .getShadowType(gutils->width);
        newOperands.push_back(
            cast<AutoDiffTypeInterface>(shadowTy).createNullValue(
                builder, origTerminator->getLoc()));
      }
    }
    Operation *repl = gutils->getNewFromOriginal(origTerminator);
    repl->setOperands(newOperands);
    return success();
  }
};

// Activity flows along the carried values as for stablehlo.while.
struct FixedPointOpADDataFlow
    : public ADDataFlowOpInterface::ExternalModel<FixedPointOpADDataFlow,
                                                  FixedPointOp> {
  SmallVector<Value> getPotentialIncomingValuesRes(Operation *op,
                                                   mlir::OpResult v) const {
    auto fp = cast<FixedPointOp>(op);
    Operation *term = fp.getBody().front().getTerminator();
    return {fp.getInits()[v.getResultNumber()],
            term->getOperand(v.getResultNumber())};
  }

  SmallVector<Value>
  getPotentialIncomingValuesArg(Operation *op, mlir::BlockArgument v) const {
    auto fp = cast<FixedPointOp>(op);
    Operation *term = fp.getBody().front().getTerminator();
    return {fp.getInits()[v.getArgNumber()], term->getOperand(v.getArgNumber())};
  }

  SmallVector<Value> getPotentialTerminatorUsers(Operation *op, Operation *term,
                                                 Value v) const {
    auto fp = cast<FixedPointOp>(op);
    if (&fp.getCond() == term->getParentRegion())
      return {};
    SmallVector<Value> sv;
    for (auto &&[res, arg, barg] :
         llvm::zip(fp.getResults(), term->getOperands(),
                   fp.getBody().front().getArguments())) {
      if (arg == v) {
        sv.push_back(res);
        sv.push_back(barg);
      }
    }
    return sv;
  }
};

void mlir::enzyme::registerEnzymeXLADialectAutoDiffInterface(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *context, EnzymeXLADialect *) {
    registerInterfaces(context);
    GPUWrapperOp::attachInterface<GPUWrapperOpInterfaceReverse>(*context);
    GPUWrapperOp::attachInterface<GPUWrapperOpEnzymeOpsRemover>(*context);

    FixedPointOp::attachInterface<FixedPointOpRev>(*context);
    FixedPointOp::attachInterface<FixedPointOpFwd>(*context);
    FixedPointOp::attachInterface<FixedPointOpCF>(*context);
    FixedPointOp::attachInterface<FixedPointOpADDataFlow>(*context);
    enzymexla::YieldOp::attachInterface<FixedPointYieldFwd>(*context);
    enzymexla::YieldOp::attachInterface<
        mlir::enzyme::detail::NoopRevAutoDiffInterface<enzymexla::YieldOp>>(
        *context);

    Pointer2MemrefOp::attachInterface<
        ViewCastOpInterfaceReverse<Pointer2MemrefOp>>(*context);
    Memref2PointerOp::attachInterface<
        ViewCastOpInterfaceReverse<Memref2PointerOp>>(*context);

    // Register batching interfaces
    JITCallOp::attachInterface<SHLOGenericBatchOpInterface<JITCallOp>>(
        *context);
    WrapOp::attachInterface<DimensionedOpBatchInterface<WrapOp>>(*context);
    ExtendOp::attachInterface<DimensionedOpBatchInterface<ExtendOp>>(*context);
    RotateOp::attachInterface<RotateOpBatchInterface>(*context);

    context->loadDialect<stablehlo::StablehloDialect>();
  });
}
