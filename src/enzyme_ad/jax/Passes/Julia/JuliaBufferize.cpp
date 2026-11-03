#include "src/enzyme_ad/jax/Passes/Julia/Passes.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "src/enzyme_ad/jax/Dialect/Julia/Dialect.h"
#include "src/enzyme_ad/jax/Dialect/Julia/Ops.h"

namespace mlir::enzymexla::julia {

#define GEN_PASS_DEF_JULIABUFFERIZEPASS
#include "src/enzyme_ad/jax/Passes/Julia/Passes.h.inc"

namespace {

static bool isBufferType(Type t) { return isa<ArrayType, ViewType>(t); }

/// Trace through pure zero-copy view operations (`ViewSliceOp`,
/// `ViewDynamicSliceOp`, `ViewReshapeOp`) to find the underlying root buffer.
static Value getRootBuffer(Value v) {
  while (Operation *def = v.getDefiningOp()) {
    if (auto vs = dyn_cast<ViewSliceOp>(def)) {
      v = vs.getSource();
    } else if (auto vds = dyn_cast<ViewDynamicSliceOp>(def)) {
      v = vds.getSource();
    } else if (auto vr = dyn_cast<ViewReshapeOp>(def)) {
      v = vr.getSource();
    } else {
      break;
    }
  }
  return v;
}

static Value createScalarOne(OpBuilder &b, Location loc, Type elemTy) {
  if (auto ft = dyn_cast<FloatType>(elemTy))
    return ConstantOp::create(b, loc, ft, b.getFloatAttr(ft, 1.0));
  if (auto it = dyn_cast<IntegerType>(elemTy))
    return ConstantOp::create(b, loc, it, b.getIntegerAttr(it, 1));
  if (auto ct = dyn_cast<ComplexType>(elemTy)) {
    Type sub = ct.getElementType();
    Value one = createScalarOne(b, loc, sub);
    Value zero = ConstantOp::create(
        b, loc, sub,
        isa<FloatType>(sub) ? TypedAttr(b.getFloatAttr(sub, 0.0))
                            : TypedAttr(b.getIntegerAttr(sub, 0)));
    return BinaryOp::create(b, loc, ct, BinaryOpKind::complex_make, one, zero);
  }
  return Value();
}

/// Emit an owned `ArrayType` copy of `src`. For static shapes, emits
/// `AllocOp` + `CopyToOp` so the buffer participates in linear-scan buffer pool
/// coloring and `WhileOp` scratch hoisting!
static Value emitOwnedCopy(OpBuilder &b, Location loc, Value src,
                           ArrayType dstTy) {
  if (dstTy.hasStaticShape()) {
    Value buf = AllocOp::create(b, loc, dstTy, ValueRange{});
    CopyToOp::create(b, loc, buf, src);
    return buf;
  }
  return CopyOp::create(b, loc, dstTy, src);
}

/// Emit an uninitialized `AllocOp` for `arrTy`, using `explicitDynDims` when
/// provided or querying dynamic dimensions from `donor` when available.
static Value emitAllocForType(OpBuilder &b, Location loc, ArrayType arrTy,
                              ValueRange explicitDynDims = {},
                              Value donor = Value()) {
  if (arrTy.hasStaticShape())
    return AllocOp::create(b, loc, arrTy, ValueRange{});

  size_t numDyn = llvm::count_if(arrTy.getShape(), ShapedType::isDynamic);
  if (explicitDynDims.size() == numDyn)
    return AllocOp::create(b, loc, arrTy, explicitDynDims);

  SmallVector<Value> dynDims;
  if (donor) {
    for (int64_t d = 0; d < arrTy.getRank(); ++d) {
      if (arrTy.isDynamicDim(d))
        dynDims.push_back(SizeOp::create(b, loc, b.getIndexType(), donor, d));
    }
  }
  return AllocOp::create(b, loc, arrTy, dynDims);
}

/// Move `srcReg`'s single block into `dstReg`.
static void moveRegionBody(Region &srcReg, Region &dstReg) {
  dstReg.takeBody(srcReg);
}

/// Check whether `op` is a single-result value-domain array producer that can
/// write directly into a forwarded `@view` slice of a `ConcatOp` destination.
static bool canForwardIntoConcatSlice(Operation *op) {
  if (!op || op->getNumResults() != 1)
    return false;
  if (!isa<ArrayType>(op->getResult(0).getType()))
    return false;
  if (isa<FillOp, KernelOp, DotGeneralOp, ReduceOp, ReduceWindowOp, PadOp,
          GatherOp, ConvOp, DynamicSliceOp>(op))
    return true;
  if (auto mm = dyn_cast<MatmulOp>(op))
    return !mm.getAcc();
  return false;
}

class BlockBufferizer {
public:
  BlockBufferizer(bool reuseBuffers, bool hoistLoopAllocs,
                  bool forwardConcatSlices)
      : reuseBuffers(reuseBuffers), hoistLoopAllocs(hoistLoopAllocs),
        forwardConcatSlices(forwardConcatSlices) {}

  void bufferizeFunc(FuncOp func) {
    if (func.isExternal())
      return;
    Block &entry = func.getBody().front();
    llvm::DenseSet<Value> donatedArgs;
    for (unsigned i = 0; i < entry.getNumArguments(); ++i) {
      Value arg = entry.getArgument(i);
      if (!isa<ArrayType>(arg.getType()))
        continue;
      if (func.getArgAttr(i, "tf.aliasing_output") ||
          func.getArgAttr(i, "jax.buffer_donor")) {
        donatedArgs.insert(arg);
      }
    }
    bufferizeBlock(entry, donatedArgs);
  }

private:
  bool reuseBuffers;
  bool hoistLoopAllocs;
  bool forwardConcatSlices;

  void bufferizeBlock(Block &block,
                      const llvm::DenseSet<Value> &ownedEntryArgs) {
    //----------------------------------------------------------------------
    // Step 1: Concat destination slice forwarding (`forward_concat_slices`)
    //----------------------------------------------------------------------
    llvm::DenseMap<Operation *, Value> forwardedDest;
    if (forwardConcatSlices)
      lowerConcatsWithSliceForwarding(block, forwardedDest);

    // Lower any remaining ConcatOp (e.g. dynamic shapes or when flag disabled).
    lowerRemainingConcats(block);

    //----------------------------------------------------------------------
    // Step 2: Zero-copy view lowering (`SliceOp`, `ReshapeOp`, non-escaping
    //         `DynamicSliceOp`)
    //----------------------------------------------------------------------
    lowerPureViews(block, forwardedDest);

    //----------------------------------------------------------------------
    // Step 3: Root-buffer ownership & liveness analysis
    //----------------------------------------------------------------------
    llvm::DenseMap<Operation *, int64_t> opIdx;
    int64_t curIdx = 0;
    for (Operation &op : block)
      opIdx[&op] = curIdx++;

    llvm::DenseSet<Value> ownedRoots = ownedEntryArgs;
    for (Operation &op : block.without_terminator()) {
      if (isa<AllocOp, CopyOp, MaterializeArrayOp, FillOp, KernelOp, MatmulOp,
              DotGeneralOp, ReduceOp, ReduceWindowOp, PadOp, GatherOp,
              ScatterOp, SortOp, ConvOp, CholeskyOp, TriangularSolveOp,
              DynamicSliceOp, DynamicUpdateSliceOp>(&op)) {
        for (Value r : op.getResults())
          if (isa<ArrayType>(r.getType()))
            ownedRoots.insert(r);
      }
    }

    auto computeRootLastUses = [&]() {
      llvm::DenseMap<Value, int64_t> lastUse;
      for (Operation &topOp : block) {
        int64_t idx = opIdx.lookup(&topOp);
        topOp.walk([&](Operation *inner) {
          for (Value opnd : inner->getOperands()) {
            if (!isBufferType(opnd.getType()))
              continue;
            Value root = getRootBuffer(opnd);
            auto it = lastUse.find(root);
            if (it == lastUse.end() || idx > it->second)
              lastUse[root] = idx;
          }
        });
      }
      return lastUse;
    };

    llvm::DenseMap<Value, int64_t> lastUseOfRoot = computeRootLastUses();

    auto isDeadOwnedArrayAt = [&](Value candidate, Type requiredType,
                                  int64_t idx) -> bool {
      if (!reuseBuffers || !candidate)
        return false;
      if (candidate.getType() != requiredType)
        return false;
      auto arrTy = dyn_cast<ArrayType>(requiredType);
      if (!arrTy || !arrTy.hasStaticShape())
        return false;
      // Candidate itself must be the root buffer (never a view!).
      if (getRootBuffer(candidate) != candidate)
        return false;
      if (!ownedRoots.contains(candidate))
        return false;
      auto it = lastUseOfRoot.find(candidate);
      return it != lastUseOfRoot.end() && it->second == idx;
    };

    //----------------------------------------------------------------------
    // Step 4: Value-Domain -> Destination-Passing / In-Place Conversion
    //----------------------------------------------------------------------
    SmallVector<Operation *> opsToVisit;
    for (Operation &op : block.without_terminator())
      opsToVisit.push_back(&op);

    for (Operation *op : opsToVisit) {
      if (!op->getBlock())
        continue;
      int64_t idx = opIdx.lookup(op);
      OpBuilder b(op);
      Location loc = op->getLoc();

      // 4.1 FillOp -> FillInPlaceOp
      if (auto fillOp = dyn_cast<FillOp>(op)) {
        Value dst = forwardedDest.lookup(op);
        if (!dst)
          dst = emitAllocForType(b, loc, fillOp.getType(),
                                 fillOp.getDynamicDims());
        FillInPlaceOp::create(b, loc, dst, fillOp.getValue());
        fillOp.getResult().replaceAllUsesWith(dst);
        fillOp.erase();
        continue;
      }

      // 4.2 KernelOp -> KernelInPlaceOp (with identity-indexed dead reuse!)
      if (auto kern = dyn_cast<KernelOp>(op)) {
        unsigned numRes = kern.getNumResults();
        Block &kBody = kern.getBody().front();

        // Collect all external buffer values accessed inside `kern` and check
        // whether each root buffer is ONLY accessed directly via identity
        // indices `[iv_0, ..., iv_{rank-1}]`.
        SmallVector<Value> candidateBuffers;
        llvm::DenseSet<Value> seenRoots;
        llvm::DenseSet<Value> unsafeRoots;

        kern.walk([&](Operation *inner) {
          if (inner == kern.getOperation())
            return;
          for (Value opnd : inner->getOperands()) {
            if (!isBufferType(opnd.getType()))
              continue;
            Value root = getRootBuffer(opnd);
            auto load = dyn_cast<LoadOp>(inner);
            bool isIdentityDirectLoad =
                load && load.getSource() == root &&
                load.getIndices().size() == kBody.getNumArguments() &&
                llvm::equal(load.getIndices(), kBody.getArguments());
            if (!isIdentityDirectLoad) {
              unsafeRoots.insert(root);
            } else if (seenRoots.insert(root).second) {
              candidateBuffers.push_back(root);
            }
          }
        });

        llvm::DenseSet<Value> claimedRoots;
        SmallVector<Value> outs(numRes);
        for (unsigned r = 0; r < numRes; ++r) {
          auto resArrTy = cast<ArrayType>(kern.getResult(r).getType());
          if (numRes == 1) {
            if (Value fwd = forwardedDest.lookup(op)) {
              outs[r] = fwd;
              continue;
            }
          }
          Value reused;
          if (reuseBuffers) {
            for (Value cand : candidateBuffers) {
              if (unsafeRoots.contains(cand) || claimedRoots.contains(cand))
                continue;
              if (isDeadOwnedArrayAt(cand, resArrTy, idx)) {
                reused = cand;
                claimedRoots.insert(cand);
                break;
              }
            }
          }
          if (reused) {
            outs[r] = reused;
            // Extend `reused` liveness now that it backs `kern.getResult(r)`!
            extendRootLivenessForReplacement(kern.getResult(r), reused, block,
                                             opIdx, lastUseOfRoot);
          } else {
            Value alloc =
                emitAllocForType(b, loc, resArrTy, kern.getDynamicDims());
            outs[r] = alloc;
            ownedRoots.insert(alloc);
            extendRootLivenessForReplacement(kern.getResult(r), alloc, block,
                                             opIdx, lastUseOfRoot);
          }
        }

        auto kip = KernelInPlaceOp::create(b, loc, outs);
        moveRegionBody(kern.getBody(), kip.getBody());
        for (unsigned r = 0; r < numRes; ++r)
          kern.getResult(r).replaceAllUsesWith(outs[r]);
        kern.erase();
        continue;
      }

      // 4.3 MatmulOp (array result) -> MatmulInPlaceOp
      if (auto mm = dyn_cast<MatmulOp>(op)) {
        auto resArrTy = dyn_cast<ArrayType>(mm.getType());
        if (!resArrTy)
          continue; // 1D scalar dot product stays in scalar form (0 allocs).

        Value dst;
        Value beta = mm.getBeta();
        if (Value acc = mm.getAcc()) {
          Value lhsRoot = getRootBuffer(mm.getLhs());
          Value rhsRoot = getRootBuffer(mm.getRhs());
          if (isDeadOwnedArrayAt(acc, resArrTy, idx) && acc != lhsRoot &&
              acc != rhsRoot) {
            dst = acc;
            extendRootLivenessForReplacement(mm.getResult(), dst, block, opIdx,
                                             lastUseOfRoot);
          } else {
            dst = emitOwnedCopy(b, loc, acc, resArrTy);
            ownedRoots.insert(dst);
            extendRootLivenessForReplacement(mm.getResult(), dst, block, opIdx,
                                             lastUseOfRoot);
          }
          if (!beta)
            beta = createScalarOne(b, loc, resArrTy.getElementType());
        } else {
          dst = forwardedDest.lookup(op);
          if (!dst) {
            dst = emitAllocForType(b, loc, resArrTy);
            ownedRoots.insert(dst);
            extendRootLivenessForReplacement(mm.getResult(), dst, block, opIdx,
                                             lastUseOfRoot);
          }
        }

        MatmulInPlaceOp::create(b, loc, dst, mm.getLhs(), mm.getRhs(),
                                mm.getAlpha(), beta, mm.getLhsTranspose(),
                                mm.getRhsTranspose(), mm.getBatchDimsLhs(),
                                mm.getBatchDimsRhs(), mm.getBatchDimsOut());
        mm.getResult().replaceAllUsesWith(dst);
        mm.erase();
        continue;
      }

      // 4.4 DotGeneralOp (array result) -> DotGeneralInPlaceOp
      if (auto dg = dyn_cast<DotGeneralOp>(op)) {
        auto resArrTy = dyn_cast<ArrayType>(dg.getType());
        if (!resArrTy)
          continue;
        Value dst = forwardedDest.lookup(op);
        if (!dst) {
          dst = emitAllocForType(b, loc, resArrTy);
          ownedRoots.insert(dst);
          extendRootLivenessForReplacement(dg.getResult(), dst, block, opIdx,
                                           lastUseOfRoot);
        }
        DotGeneralInPlaceOp::create(
            b, loc, dst, dg.getLhs(), dg.getRhs(), dg.getLhsBatchingDims(),
            dg.getRhsBatchingDims(), dg.getLhsContractingDims(),
            dg.getRhsContractingDims(), dg.getResultDimOrder());
        dg.getResult().replaceAllUsesWith(dst);
        dg.erase();
        continue;
      }

      // 4.5 ReduceOp (array result) -> ReduceInPlaceOp
      if (auto red = dyn_cast<ReduceOp>(op)) {
        if (red.getNumResults() == 0 ||
            !isa<ArrayType>(red.getResult(0).getType()))
          continue;
        SmallVector<Value> dsts(red.getNumResults());
        for (unsigned r = 0; r < red.getNumResults(); ++r) {
          auto rTy = cast<ArrayType>(red.getResult(r).getType());
          if (red.getNumResults() == 1) {
            if (Value fwd = forwardedDest.lookup(op)) {
              dsts[r] = fwd;
              continue;
            }
          }
          dsts[r] = emitAllocForType(b, loc, rTy);
          ownedRoots.insert(dsts[r]);
          extendRootLivenessForReplacement(red.getResult(r), dsts[r], block,
                                           opIdx, lastUseOfRoot);
        }
        auto rip = ReduceInPlaceOp::create(
            b, loc, dsts, red.getInputs(), red.getInitValues(),
            red.getDimensions());
        moveRegionBody(red.getCombiner(), rip.getCombiner());
        for (unsigned r = 0; r < red.getNumResults(); ++r)
          red.getResult(r).replaceAllUsesWith(dsts[r]);
        red.erase();
        continue;
      }

      // 4.6 ReduceWindowOp -> ReduceWindowInPlaceOp
      if (auto rw = dyn_cast<ReduceWindowOp>(op)) {
        SmallVector<Value> dsts(rw.getNumResults());
        for (unsigned r = 0; r < rw.getNumResults(); ++r) {
          auto rTy = cast<ArrayType>(rw.getResult(r).getType());
          if (rw.getNumResults() == 1) {
            if (Value fwd = forwardedDest.lookup(op)) {
              dsts[r] = fwd;
              continue;
            }
          }
          dsts[r] = emitAllocForType(b, loc, rTy);
          ownedRoots.insert(dsts[r]);
          extendRootLivenessForReplacement(rw.getResult(r), dsts[r], block,
                                           opIdx, lastUseOfRoot);
        }
        auto rwip = ReduceWindowInPlaceOp::create(
            b, loc, dsts, rw.getInputs(), rw.getInitValues(),
            rw.getWindowDimensions(), rw.getWindowStrides(),
            rw.getBaseDilations(), rw.getWindowDilations(), rw.getPadding());
        moveRegionBody(rw.getCombiner(), rwip.getCombiner());
        for (unsigned r = 0; r < rw.getNumResults(); ++r)
          rw.getResult(r).replaceAllUsesWith(dsts[r]);
        rw.erase();
        continue;
      }

      // 4.7 PadOp -> PadInPlaceOp
      if (auto pad = dyn_cast<PadOp>(op)) {
        Value dst = forwardedDest.lookup(op);
        if (!dst) {
          dst = emitAllocForType(b, loc, pad.getType());
          ownedRoots.insert(dst);
          extendRootLivenessForReplacement(pad.getResult(), dst, block, opIdx,
                                           lastUseOfRoot);
        }
        PadInPlaceOp::create(b, loc, dst, pad.getOperand(),
                             pad.getPaddingValue(), pad.getEdgePaddingLow(),
                             pad.getEdgePaddingHigh(),
                             pad.getInteriorPadding());
        pad.getResult().replaceAllUsesWith(dst);
        pad.erase();
        continue;
      }

      // 4.8 DynamicSliceOp (escaping / non-view) -> DynamicSliceInPlaceOp
      if (auto dsl = dyn_cast<DynamicSliceOp>(op)) {
        Value dst = forwardedDest.lookup(op);
        if (!dst) {
          dst = emitAllocForType(b, loc, dsl.getType());
          ownedRoots.insert(dst);
          extendRootLivenessForReplacement(dsl.getResult(), dst, block, opIdx,
                                           lastUseOfRoot);
        }
        DynamicSliceInPlaceOp::create(b, loc, dst, dsl.getOperand(),
                                      dsl.getStartIndices(),
                                      dsl.getSliceSizes());
        dsl.getResult().replaceAllUsesWith(dst);
        dsl.erase();
        continue;
      }

      // 4.9 DynamicUpdateSliceOp -> DynamicUpdateSliceInPlaceOp
      if (auto dus = dyn_cast<DynamicUpdateSliceOp>(op)) {
        auto resArrTy = dus.getType();
        Value operand = dus.getOperand();
        Value updRoot = isBufferType(dus.getUpdate().getType())
                            ? getRootBuffer(dus.getUpdate())
                            : Value();
        Value dst;
        if (isDeadOwnedArrayAt(operand, resArrTy, idx) && operand != updRoot) {
          dst = operand;
        } else {
          dst = emitOwnedCopy(b, loc, operand, resArrTy);
          ownedRoots.insert(dst);
        }
        extendRootLivenessForReplacement(dus.getResult(), dst, block, opIdx,
                                         lastUseOfRoot);
        DynamicUpdateSliceInPlaceOp::create(b, loc, dst, dus.getUpdate(),
                                            dus.getStartIndices());
        dus.getResult().replaceAllUsesWith(dst);
        dus.erase();
        continue;
      }

      // 4.10 GatherOp (array result) -> GatherInPlaceOp
      if (auto gop = dyn_cast<GatherOp>(op)) {
        auto resArrTy = dyn_cast<ArrayType>(gop.getType());
        if (!resArrTy)
          continue;
        Value dst = forwardedDest.lookup(op);
        if (!dst) {
          dst = emitAllocForType(b, loc, resArrTy);
          ownedRoots.insert(dst);
          extendRootLivenessForReplacement(gop.getResult(), dst, block, opIdx,
                                           lastUseOfRoot);
        }
        GatherInPlaceOp::create(
            b, loc, dst, gop.getOperand(), gop.getStartIndices(),
            gop.getOffsetDims(), gop.getCollapsedSliceDims(),
            gop.getOperandBatchingDims(), gop.getStartIndicesBatchingDims(),
            gop.getStartIndexMap(), gop.getIndexVectorDim(),
            gop.getSliceSizes());
        gop.getResult().replaceAllUsesWith(dst);
        gop.erase();
        continue;
      }

      // 4.11 ScatterOp -> ScatterInPlaceOp
      if (auto sc = dyn_cast<ScatterOp>(op)) {
        llvm::DenseSet<Value> forbiddenRoots;
        if (isBufferType(sc.getScatterIndices().getType()))
          forbiddenRoots.insert(getRootBuffer(sc.getScatterIndices()));
        for (Value u : sc.getUpdates())
          if (isBufferType(u.getType()))
            forbiddenRoots.insert(getRootBuffer(u));

        llvm::DenseSet<Value> claimed;
        SmallVector<Value> dsts(sc.getNumResults());
        for (unsigned r = 0; r < sc.getNumResults(); ++r) {
          Value inVal = sc.getInputs()[r];
          auto resArrTy = cast<ArrayType>(sc.getResult(r).getType());
          bool aliasesOtherInput = false;
          for (unsigned j = 0; j < sc.getNumResults(); ++j) {
            if (j != r && getRootBuffer(sc.getInputs()[j]) ==
                              getRootBuffer(inVal)) {
              aliasesOtherInput = true;
              break;
            }
          }
          if (!aliasesOtherInput && !forbiddenRoots.contains(inVal) &&
              !claimed.contains(inVal) &&
              isDeadOwnedArrayAt(inVal, resArrTy, idx)) {
            dsts[r] = inVal;
            claimed.insert(inVal);
          } else {
            dsts[r] = emitOwnedCopy(b, loc, inVal, resArrTy);
            ownedRoots.insert(dsts[r]);
          }
          extendRootLivenessForReplacement(sc.getResult(r), dsts[r], block,
                                           opIdx, lastUseOfRoot);
        }
        auto scip = ScatterInPlaceOp::create(
            b, loc, dsts, sc.getScatterIndices(), sc.getUpdates(),
            sc.getUpdateWindowDims(), sc.getInsertedWindowDims(),
            sc.getInputBatchingDims(), sc.getScatterIndicesBatchingDims(),
            sc.getScatterDimsToOperandDims(), sc.getIndexVectorDim());
        moveRegionBody(sc.getUpdateComputation(), scip.getUpdateComputation());
        for (unsigned r = 0; r < sc.getNumResults(); ++r)
          sc.getResult(r).replaceAllUsesWith(dsts[r]);
        sc.erase();
        continue;
      }

      // 4.12 SortOp -> SortInPlaceOp
      if (auto sop = dyn_cast<SortOp>(op)) {
        llvm::DenseSet<Value> claimed;
        SmallVector<Value> dsts(sop.getNumResults());
        for (unsigned r = 0; r < sop.getNumResults(); ++r) {
          Value inVal = sop.getInputs()[r];
          auto resArrTy = cast<ArrayType>(sop.getResult(r).getType());
          bool aliasesOther = false;
          for (unsigned j = 0; j < sop.getNumResults(); ++j) {
            if (j != r && getRootBuffer(sop.getInputs()[j]) ==
                              getRootBuffer(inVal)) {
              aliasesOther = true;
              break;
            }
          }
          if (!aliasesOther && !claimed.contains(inVal) &&
              isDeadOwnedArrayAt(inVal, resArrTy, idx)) {
            dsts[r] = inVal;
            claimed.insert(inVal);
          } else {
            dsts[r] = emitOwnedCopy(b, loc, inVal, resArrTy);
            ownedRoots.insert(dsts[r]);
          }
          extendRootLivenessForReplacement(sop.getResult(r), dsts[r], block,
                                           opIdx, lastUseOfRoot);
        }
        auto sip = SortInPlaceOp::create(b, loc, dsts, sop.getDimension(),
                                         sop.getIsStable());
        moveRegionBody(sop.getComparator(), sip.getComparator());
        for (unsigned r = 0; r < sop.getNumResults(); ++r)
          sop.getResult(r).replaceAllUsesWith(dsts[r]);
        sop.erase();
        continue;
      }

      // 4.13 ConvOp -> ConvInPlaceOp
      if (auto cv = dyn_cast<ConvOp>(op)) {
        Value dst = forwardedDest.lookup(op);
        if (!dst) {
          dst = emitAllocForType(b, loc, cv.getType());
          ownedRoots.insert(dst);
          extendRootLivenessForReplacement(cv.getResult(), dst, block, opIdx,
                                           lastUseOfRoot);
        }
        ConvInPlaceOp::create(
            b, loc, dst, cv.getLhs(), cv.getRhs(), cv.getWindowStrides(),
            cv.getPadding(), cv.getLhsDilation(), cv.getRhsDilation(),
            cv.getWindowReversal(), cv.getInputBatchDimension(),
            cv.getInputFeatureDimension(), cv.getInputSpatialDimensions(),
            cv.getKernelInputFeatureDimension(),
            cv.getKernelOutputFeatureDimension(),
            cv.getKernelSpatialDimensions(), cv.getOutputBatchDimension(),
            cv.getOutputFeatureDimension(), cv.getOutputSpatialDimensions(),
            cv.getFeatureGroupCount(), cv.getBatchGroupCount());
        cv.getResult().replaceAllUsesWith(dst);
        cv.erase();
        continue;
      }

      // 4.14 CholeskyOp -> CholeskyInPlaceOp
      if (auto ch = dyn_cast<CholeskyOp>(op)) {
        auto resArrTy = ch.getType();
        Value operand = ch.getOperand();
        Value dst;
        if (isDeadOwnedArrayAt(operand, resArrTy, idx)) {
          dst = operand;
        } else {
          dst = emitOwnedCopy(b, loc, operand, resArrTy);
          ownedRoots.insert(dst);
        }
        extendRootLivenessForReplacement(ch.getResult(), dst, block, opIdx,
                                         lastUseOfRoot);
        CholeskyInPlaceOp::create(b, loc, dst, ch.getLower());
        ch.getResult().replaceAllUsesWith(dst);
        ch.erase();
        continue;
      }

      // 4.15 TriangularSolveOp -> TriangularSolveInPlaceOp
      if (auto ts = dyn_cast<TriangularSolveOp>(op)) {
        auto resArrTy = ts.getType();
        Value bVal = ts.getB();
        Value aRoot = getRootBuffer(ts.getA());
        Value dst;
        if (isDeadOwnedArrayAt(bVal, resArrTy, idx) && bVal != aRoot) {
          dst = bVal;
        } else {
          dst = emitOwnedCopy(b, loc, bVal, resArrTy);
          ownedRoots.insert(dst);
        }
        extendRootLivenessForReplacement(ts.getResult(), dst, block, opIdx,
                                         lastUseOfRoot);
        TriangularSolveInPlaceOp::create(
            b, loc, dst, ts.getA(), ts.getLeftSide(), ts.getLower(),
            ts.getUnitDiagonal(), ts.getTransposeA());
        ts.getResult().replaceAllUsesWith(dst);
        ts.erase();
        continue;
      }

      // 4.16 CallOp boundary view materialization
      if (auto callOp = dyn_cast<CallOp>(op)) {
        for (OpOperand &opnd : callOp->getOpOperands()) {
          if (auto vt = dyn_cast<ViewType>(opnd.get().getType())) {
            auto arrTy = ArrayType::get(vt.getShape(), vt.getElementType());
            Value mat = MaterializeArrayOp::create(b, loc, arrTy, opnd.get());
            opnd.set(mat);
          }
        }
        continue;
      }

      // 4.17 IfOp recursive bufferization
      if (auto ifOp = dyn_cast<IfOp>(op)) {
        llvm::DenseSet<Value> emptyOwned;
        bufferizeBlock(ifOp.getThenRegion().front(), emptyOwned);
        materializeTerminatorViews(ifOp.getThenRegion().front(),
                                   ifOp.getResultTypes());
        if (!ifOp.getElseRegion().empty()) {
          bufferizeBlock(ifOp.getElseRegion().front(), emptyOwned);
          materializeTerminatorViews(ifOp.getElseRegion().front(),
                                     ifOp.getResultTypes());
        }
        continue;
      }

      // 4.18 WhileOp zero-allocation bufferization + ping-pong hoisting
      if (auto whileOp = dyn_cast<WhileOp>(op)) {
        bufferizeWhileOp(whileOp, idx, block, opIdx, ownedRoots, lastUseOfRoot);
        continue;
      }
    }

    //----------------------------------------------------------------------
    // Step 5: Materialize any escaping `ViewType` at block terminator
    //----------------------------------------------------------------------
    if (isa<ReturnOp>(block.getTerminator())) {
      if (auto fn = dyn_cast<FuncOp>(block.getParentOp()))
        materializeTerminatorViews(block, fn.getResultTypes());
    }

    // Erase dead pure view ops before coloring.
    eraseDeadPureOps(block);

    //----------------------------------------------------------------------
    // Step 6: Linear-scan buffer pool coloring (`reuse_buffers`)
    //----------------------------------------------------------------------
    if (reuseBuffers)
      colorNonOverlappingAllocations(block);
  }

  static void extendRootLivenessForReplacement(
      Value oldVal, Value newRoot, Block &block,
      const llvm::DenseMap<Operation *, int64_t> &opIdx,
      llvm::DenseMap<Value, int64_t> &lastUseOfRoot) {
    int64_t maxIdx =
        std::max(lastUseOfRoot.lookup(newRoot), lastUseOfRoot.lookup(oldVal));
    for (Operation *user : oldVal.getUsers()) {
      if (Operation *anc = block.findAncestorOpInBlock(*user))
        maxIdx = std::max(maxIdx, opIdx.lookup(anc));
    }
    lastUseOfRoot[newRoot] = maxIdx;
  }

  static void materializeTerminatorViews(Block &block,
                                         TypeRange expectedTypes) {
    Operation *term = block.getTerminator();
    OpBuilder b(term);
    Location loc = term->getLoc();
    for (unsigned i = 0;
         i < term->getNumOperands() && i < expectedTypes.size(); ++i) {
      Value v = term->getOperand(i);
      Type expTy = expectedTypes[i];
      if (isa<ViewType>(v.getType()) && isa<ArrayType>(expTy)) {
        Value mat =
            MaterializeArrayOp::create(b, loc, cast<ArrayType>(expTy), v);
        term->setOperand(i, mat);
      }
    }
  }

  static void eraseDeadPureOps(Block &block) {
    for (Operation *cur = block.getTerminator()->getPrevNode(); cur;) {
      Operation *prev = cur->getPrevNode();
      if (cur->use_empty() &&
          isa<ViewSliceOp, ViewDynamicSliceOp, ViewReshapeOp, ConstantOp,
              SizeOp>(cur)) {
        cur->erase();
      }
      cur = prev;
    }
  }

  /// Pre-Pass 1: Allocate static `ConcatOp` buffers upfront and forward
  /// `@view` slices directly into single-use value-domain inputs!
  static void lowerConcatsWithSliceForwarding(
      Block &block, llvm::DenseMap<Operation *, Value> &forwardedDest) {
    SmallVector<ConcatOp> concats;
    for (Operation &op : block.without_terminator())
      if (auto cat = dyn_cast<ConcatOp>(&op))
        concats.push_back(cat);

    for (ConcatOp cat : concats) {
      auto resTy = cat.getType();
      if (!resTy.hasStaticShape())
        continue;
      bool allStaticInputs = llvm::all_of(cat.getInputs(), [](Value v) {
        auto st = dyn_cast<ShapedType>(v.getType());
        return st && st.hasStaticShape();
      });
      if (!allStaticInputs)
        continue;

      int64_t rank = resTy.getRank();
      int64_t concatDim = cat.getDimension();

      // Place `%dst = AllocOp(resTy)` right before the earliest forwarded
      // producer in `block` (or right before `cat` if none are forwarded).
      Operation *insertPt = cat.getOperation();
      for (Value inVal : cat.getInputs()) {
        Operation *prod = inVal.getDefiningOp();
        if (prod && prod->getBlock() == &block && inVal.hasOneUse() &&
            canForwardIntoConcatSlice(prod) && !forwardedDest.contains(prod)) {
          if (prod->isBeforeInBlock(insertPt))
            insertPt = prod;
        }
      }

      OpBuilder allocBuilder(insertPt);
      Location loc = cat.getLoc();
      Value dst = AllocOp::create(allocBuilder, loc, resTy, ValueRange{});

      int64_t offset = 0;
      SmallVector<int64_t> ones(rank, 1);
      for (Value inVal : cat.getInputs()) {
        auto inTy = cast<ShapedType>(inVal.getType());
        int64_t len = inTy.getDimSize(concatDim);

        SmallVector<int64_t> starts(rank, 0);
        SmallVector<int64_t> limits(inTy.getShape().begin(),
                                    inTy.getShape().end());
        starts[concatDim] = offset;
        limits[concatDim] = offset + len;
        offset += len;

        auto viewTy = ViewType::get(inTy.getShape(), inTy.getElementType());
        Operation *prod = inVal.getDefiningOp();
        if (prod && prod->getBlock() == &block && inVal.hasOneUse() &&
            canForwardIntoConcatSlice(prod) && !forwardedDest.contains(prod)) {
          OpBuilder viewBuilder(prod);
          Value sliceView = ViewSliceOp::create(viewBuilder, loc, viewTy, dst,
                                                starts, limits, ones);
          forwardedDest[prod] = sliceView;
        } else {
          OpBuilder copyBuilder(cat);
          Value sliceView = ViewSliceOp::create(copyBuilder, loc, viewTy, dst,
                                                starts, limits, ones);
          CopyToOp::create(copyBuilder, loc, sliceView, inVal);
        }
      }

      cat.getResult().replaceAllUsesWith(dst);
      cat.erase();
    }
  }

  static void lowerRemainingConcats(Block &block) {
    for (Operation &op :
         llvm::make_early_inc_range(block.without_terminator())) {
      auto cat = dyn_cast<ConcatOp>(&op);
      if (!cat)
        continue;
      auto resTy = cat.getType();
      if (!resTy.hasStaticShape())
        continue;
      bool allStatic = llvm::all_of(cat.getInputs(), [](Value v) {
        auto st = dyn_cast<ShapedType>(v.getType());
        return st && st.hasStaticShape();
      });
      if (!allStatic)
        continue;

      OpBuilder b(cat);
      Location loc = cat.getLoc();
      Value dst = AllocOp::create(b, loc, resTy, ValueRange{});
      int64_t rank = resTy.getRank();
      int64_t concatDim = cat.getDimension();
      int64_t offset = 0;
      SmallVector<int64_t> ones(rank, 1);
      for (Value inVal : cat.getInputs()) {
        auto inTy = cast<ShapedType>(inVal.getType());
        int64_t len = inTy.getDimSize(concatDim);
        SmallVector<int64_t> starts(rank, 0);
        SmallVector<int64_t> limits(inTy.getShape().begin(),
                                    inTy.getShape().end());
        starts[concatDim] = offset;
        limits[concatDim] = offset + len;
        offset += len;
        auto viewTy = ViewType::get(inTy.getShape(), inTy.getElementType());
        Value sliceView =
            ViewSliceOp::create(b, loc, viewTy, dst, starts, limits, ones);
        CopyToOp::create(b, loc, sliceView, inVal);
      }
      cat.getResult().replaceAllUsesWith(dst);
      cat.erase();
    }
  }

  static void lowerPureViews(
      Block &block, const llvm::DenseMap<Operation *, Value> &forwardedDest) {
    for (Operation &op :
         llvm::make_early_inc_range(block.without_terminator())) {
      OpBuilder b(&op);
      Location loc = op.getLoc();

      if (auto sl = dyn_cast<SliceOp>(&op)) {
        auto rTy = sl.getType();
        auto vTy = ViewType::get(rTy.getShape(), rTy.getElementType());
        Value v = ViewSliceOp::create(b, loc, vTy, sl.getOperand(),
                                      sl.getStartIndices(),
                                      sl.getLimitIndices(), sl.getStrides());
        sl.getResult().replaceAllUsesWith(v);
        sl.erase();
        continue;
      }

      if (auto resh = dyn_cast<ReshapeOp>(&op)) {
        auto rTy = resh.getType();
        auto vTy = ViewType::get(rTy.getShape(), rTy.getElementType());
        Value v = ViewReshapeOp::create(b, loc, vTy, resh.getOperand(),
                                        resh.getDynamicDims());
        resh.getResult().replaceAllUsesWith(v);
        resh.erase();
        continue;
      }

      if (auto dsl = dyn_cast<DynamicSliceOp>(&op)) {
        if (forwardedDest.contains(&op))
          continue;
        bool escapes = llvm::any_of(dsl->getUsers(), [](Operation *user) {
          return isa<ReturnOp, YieldOp, CallOp, WhileOp>(user);
        });
        if (!escapes) {
          auto rTy = dsl.getType();
          auto vTy = ViewType::get(rTy.getShape(), rTy.getElementType());
          Value v = ViewDynamicSliceOp::create(
              b, loc, vTy, dsl.getOperand(), dsl.getStartIndices(),
              dsl.getSliceSizes());
          dsl.getResult().replaceAllUsesWith(v);
          dsl.erase();
        }
        continue;
      }
    }
  }

  void bufferizeWhileOp(WhileOp whileOp, int64_t whileIdx, Block &parentBlock,
                        llvm::DenseMap<Operation *, int64_t> &opIdx,
                        llvm::DenseSet<Value> &ownedRoots,
                        llvm::DenseMap<Value, int64_t> &lastUseOfRoot) {
    OpBuilder b(whileOp);
    Location loc = whileOp.getLoc();
    Block &condBlk = whileOp.getCond().front();
    Block &bodyBlk = whileOp.getBody().front();

    // Find all outer buffer roots captured directly inside `cond` or `body` so
    // an `init` operand never mutates a loop-invariant captured buffer!
    llvm::DenseSet<Value> capturedOuterRoots;
    auto scanCaptures = [&](Region &reg) {
      reg.walk([&](Operation *inner) {
        for (Value opnd : inner->getOperands()) {
          if (!isBufferType(opnd.getType()))
            continue;
          if (auto ba = dyn_cast<BlockArgument>(opnd)) {
            if (ba.getParentRegion() == &reg)
              continue;
          } else if (Operation *def = opnd.getDefiningOp()) {
            if (reg.isAncestor(def->getParentRegion()))
              continue;
          }
          capturedOuterRoots.insert(getRootBuffer(opnd));
        }
      });
    };
    scanCaptures(whileOp.getCond());
    scanCaptures(whileOp.getBody());

    // Step A: Ensure every `ArrayType` `init` operand of `whileOp` is an
    // exclusively-owned writable `ArrayType` buffer!
    llvm::DenseSet<Value> usedInitRoots;
    for (unsigned i = 0; i < whileOp.getInits().size(); ++i) {
      Value initVal = whileOp.getInits()[i];
      auto arrTy = dyn_cast<ArrayType>(whileOp.getResult(i).getType());
      if (!arrTy)
        continue;

      bool canTakeOwnership =
          reuseBuffers && initVal.getType() == arrTy &&
          arrTy.hasStaticShape() && getRootBuffer(initVal) == initVal &&
          ownedRoots.contains(initVal) &&
          lastUseOfRoot.lookup(initVal) == whileIdx &&
          !capturedOuterRoots.contains(initVal) &&
          !usedInitRoots.contains(initVal);

      if (canTakeOwnership) {
        usedInitRoots.insert(initVal);
      } else {
        Value initCopy = emitOwnedCopy(b, loc, initVal, arrTy);
        ownedRoots.insert(initCopy);
        usedInitRoots.insert(initCopy);
        whileOp.getInitsMutable()[i].assign(initCopy);
      }
    }

    // Step B: Bufferize `cond` (read-only args) and `body` (owned array args!).
    llvm::DenseSet<Value> condOwnedArgs;
    bufferizeBlock(condBlk, condOwnedArgs);

    llvm::DenseSet<Value> bodyOwnedArgs;
    for (BlockArgument arg : bodyBlk.getArguments())
      if (isa<ArrayType>(arg.getType()))
        bodyOwnedArgs.insert(arg);
    bufferizeBlock(bodyBlk, bodyOwnedArgs);
    materializeTerminatorViews(bodyBlk, whileOp.getResultTypes());

    if (!hoistLoopAllocs) {
      for (Value res : whileOp.getResults())
        if (isa<ArrayType>(res.getType())) {
          ownedRoots.insert(res);
          extendRootLivenessForReplacement(res, res, parentBlock, opIdx,
                                           lastUseOfRoot);
        }
      return;
    }

    // Step C: Ping-pong double-buffer hoisting!
    // When `bodyBlk` could not overwrite `bodyArg[i]` in-place (e.g. `w = A * w`
    // in BLAS `mul!` where `w` aliases `rhs`), `bodyBlk` allocated a fresh
    // static `AllocOp` `%next` and yielded `%next` in slot `i`.
    // Hoist `%pong = AllocOp` before `whileOp`, pass `%pong` as an extra
    // loop-carried state arg, replace `%next` with `%pongBodyArg`, and yield
    // `(%pongBodyArg, %bodyArg[i])` so Julia swaps pointers `(w, pong) = (pong, w)`
    // with ZERO heap allocations inside the loop!
    auto bodyYield = cast<YieldOp>(bodyBlk.getTerminator());
    unsigned origCount = whileOp.getNumResults();

    llvm::DenseSet<Value> allYieldedRoots;
    llvm::DenseMap<Value, unsigned> yieldRootUseCount;
    for (Value yv : bodyYield.getOperands()) {
      if (isBufferType(yv.getType())) {
        Value r = getRootBuffer(yv);
        allYieldedRoots.insert(r);
        yieldRootUseCount[r]++;
      }
    }

    SmallVector<Value> extraInits;
    SmallVector<Type> extraTypes;
    SmallVector<Value> extraYields;

    for (unsigned i = 0; i < origCount; ++i) {
      BlockArgument bArg = bodyBlk.getArgument(i);
      auto arrTy = dyn_cast<ArrayType>(bArg.getType());
      if (!arrTy || !arrTy.hasStaticShape())
        continue;

      Value yv = bodyYield.getOperand(i);
      auto localAlloc = yv.getDefiningOp<AllocOp>();
      if (!localAlloc || localAlloc->getBlock() != &bodyBlk ||
          localAlloc.getType() != arrTy)
        continue;

      // Ensure `bArg` is not yielded in another slot and `localAlloc` is
      // yielded exclusively in slot `i` (and doesn't escape via `CallOp`).
      if (allYieldedRoots.contains(bArg) ||
          yieldRootUseCount.lookup(localAlloc) != 1)
        continue;
      bool escapesCall = llvm::any_of(localAlloc->getUsers(), [](Operation *u) {
        return isa<CallOp>(u);
      });
      if (escapesCall)
        continue;

      Value pongInit = AllocOp::create(b, loc, arrTy, ValueRange{});
      ownedRoots.insert(pongInit);
      lastUseOfRoot[pongInit] = whileIdx;

      extraInits.push_back(pongInit);
      extraTypes.push_back(arrTy);

      condBlk.addArgument(arrTy, loc);
      BlockArgument pongBodyArg = bodyBlk.addArgument(arrTy, loc);
      localAlloc.getResult().replaceAllUsesWith(pongBodyArg);
      localAlloc.erase();

      // Note: `bodyYield.getOperand(i)` was already updated to `pongBodyArg`
      // by `replaceAllUsesWith`, and the new pong slot yields `bArg`!
      extraYields.push_back(bArg);
    }

    WhileOp activeWhile = whileOp;
    if (!extraInits.empty()) {
      SmallVector<Value> newYieldOperands(bodyYield.getOperands().begin(),
                                          bodyYield.getOperands().end());
      newYieldOperands.append(extraYields.begin(), extraYields.end());
      OpBuilder yb(bodyYield);
      YieldOp::create(yb, bodyYield.getLoc(), newYieldOperands);
      bodyYield.erase();

      SmallVector<Value> newInits(whileOp.getInits().begin(),
                                  whileOp.getInits().end());
      newInits.append(extraInits.begin(), extraInits.end());

      SmallVector<Type> newResTypes(whileOp.getResultTypes().begin(),
                                    whileOp.getResultTypes().end());
      newResTypes.append(extraTypes.begin(), extraTypes.end());

      auto newWhile = WhileOp::create(b, loc, newResTypes, newInits);
      moveRegionBody(whileOp.getCond(), newWhile.getCond());
      moveRegionBody(whileOp.getBody(), newWhile.getBody());

      for (unsigned i = 0; i < origCount; ++i)
        whileOp.getResult(i).replaceAllUsesWith(newWhile.getResult(i));
      opIdx[newWhile.getOperation()] = whileIdx;
      whileOp.erase();
      activeWhile = newWhile;
    }

    // Step D: Hoist remaining non-escaping static scratch `AllocOp`s out of
    // `cond` and `body` right before `activeWhile`!
    auto hoistScratchFromBlock = [&](Block &loopBlk) {
      llvm::DenseSet<Value> escapingLoopRoots;
      Operation *term = loopBlk.getTerminator();
      for (Value v : term->getOperands())
        if (isBufferType(v.getType()))
          escapingLoopRoots.insert(getRootBuffer(v));
      loopBlk.walk([&](CallOp callOp) {
        for (Value v : callOp.getOperands())
          if (isBufferType(v.getType()))
            escapingLoopRoots.insert(getRootBuffer(v));
      });
      loopBlk.walk([&](WhileOp innerWhile) {
        for (Value v : innerWhile.getInits())
          if (isBufferType(v.getType()))
            escapingLoopRoots.insert(getRootBuffer(v));
      });

      for (Operation &inner :
           llvm::make_early_inc_range(loopBlk.without_terminator())) {
        auto alloc = dyn_cast<AllocOp>(&inner);
        if (!alloc || !alloc.getType().hasStaticShape() ||
            !alloc.getDynamicDims().empty())
          continue;
        if (escapingLoopRoots.contains(alloc.getResult()))
          continue;
        alloc->moveBefore(activeWhile);
        ownedRoots.insert(alloc.getResult());
        lastUseOfRoot[alloc.getResult()] = whileIdx;
      }
    };
    hoistScratchFromBlock(activeWhile.getCond().front());
    hoistScratchFromBlock(activeWhile.getBody().front());

    for (unsigned i = 0; i < origCount; ++i) {
      Value res = activeWhile.getResult(i);
      if (isa<ArrayType>(res.getType())) {
        ownedRoots.insert(res);
        extendRootLivenessForReplacement(res, res, parentBlock, opIdx,
                                         lastUseOfRoot);
      }
    }
  }

  /// Step 6: Linear-scan buffer pool coloring across non-overlapping static
  /// non-escaping `AllocOp` intervals within `block`.
  static void colorNonOverlappingAllocations(Block &block) {
    llvm::DenseMap<Operation *, int64_t> opIdx;
    int64_t curIdx = 0;
    for (Operation &op : block)
      opIdx[&op] = curIdx++;

    // Identify all root buffers that escape `block` (returned, yielded, passed
    // as `WhileOp` init state, or passed to `CallOp`).
    llvm::DenseSet<Value> escapingRoots;
    Operation *term = block.getTerminator();
    for (Value v : term->getOperands())
      if (isBufferType(v.getType()))
        escapingRoots.insert(getRootBuffer(v));

    for (Operation &op : block.without_terminator()) {
      if (auto whileOp = dyn_cast<WhileOp>(&op)) {
        for (Value v : whileOp.getInits())
          if (isBufferType(v.getType()))
            escapingRoots.insert(getRootBuffer(v));
      }
      op.walk([&](CallOp callOp) {
        for (Value v : callOp.getOperands())
          if (isBufferType(v.getType()))
            escapingRoots.insert(getRootBuffer(v));
      });
    }

    // Compute exact `[defIdx, lastUseIdx]` for every pool-eligible `AllocOp`.
    llvm::DenseMap<Value, int64_t> lastUse;
    for (Operation &topOp : block) {
      int64_t idx = opIdx.lookup(&topOp);
      topOp.walk([&](Operation *inner) {
        for (Value opnd : inner->getOperands()) {
          if (!isBufferType(opnd.getType()))
            continue;
          Value root = getRootBuffer(opnd);
          auto it = lastUse.find(root);
          if (it == lastUse.end() || idx > it->second)
            lastUse[root] = idx;
        }
      });
    }

    struct PoolSlot {
      AllocOp repr;
      int64_t busyUntil;
    };
    SmallVector<PoolSlot> pool;

    for (Operation &op :
         llvm::make_early_inc_range(block.without_terminator())) {
      auto alloc = dyn_cast<AllocOp>(&op);
      if (!alloc || !alloc.getType().hasStaticShape() ||
          !alloc.getDynamicDims().empty())
        continue;
      if (alloc.use_empty()) {
        alloc.erase();
        continue;
      }
      if (escapingRoots.contains(alloc.getResult()))
        continue;

      int64_t defIdx = opIdx.lookup(&op);
      int64_t endIdx = std::max(defIdx, lastUse.lookup(alloc.getResult()));

      PoolSlot *bestSlot = nullptr;
      for (PoolSlot &slot : pool) {
        if (slot.repr.getType() == alloc.getType() && slot.busyUntil < defIdx) {
          bestSlot = &slot;
          break;
        }
      }

      if (bestSlot) {
        alloc.getResult().replaceAllUsesWith(bestSlot->repr.getResult());
        bestSlot->busyUntil = std::max(bestSlot->busyUntil, endIdx);
        alloc.erase();
      } else {
        pool.push_back({alloc, endIdx});
      }
    }
  }
};

struct JuliaBufferizePass
    : public impl::JuliaBufferizePassBase<JuliaBufferizePass> {
  using Base::Base;

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    BlockBufferizer bufferizer(reuse_buffers, hoist_loop_allocs,
                               forward_concat_slices);
    for (FuncOp fn : mod.getOps<FuncOp>())
      bufferizer.bufferizeFunc(fn);
  }
};

} // namespace

} // namespace mlir::enzymexla::julia
