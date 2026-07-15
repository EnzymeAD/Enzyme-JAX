#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h"

#include <limits>

#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "src/enzyme_ad/jax/Dialect/Distributed/Dialect.h"
#include "src/enzyme_ad/jax/Passes/Distributed/TimingAnalysis.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir {
namespace enzyme {
namespace distributed {

#define GEN_PASS_DEF_DISTRIBUTEDOVERLAPCOMMUNICATIONPASS
#define GEN_PASS_DEF_DISTRIBUTEDOVERLAPCOMMUNICATIONMODULEPASS
#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h.inc"

// Forward declaration from SinkRecvs.cpp.
void sinkRecvs(MeshComputationOp meshOp);

namespace {

// ===----------------------------------------------------------------------===
// Timing helpers
// ===----------------------------------------------------------------------===

struct TimingResult {
  double criticalPathTime;
};

TimingResult analyzeTiming(MeshComputationOp meshOp) {
  HappensBeforeAnalysis hb(meshOp);
  AffineTimingCostModel costModel;
  TimingAnalysis timing(hb, costModel);

  double maxTime = 0.0;
  for (Operation *root : hb.classesInTopologicalOrder()) {
    auto timeRange = timing.getTimeRange(root);
    maxTime = std::max(maxTime, timeRange.second);
  }
  return {maxTime};
}

// ===----------------------------------------------------------------------===
// CollectiveInfo: all ops associated with a single collective token.
// ===----------------------------------------------------------------------===

struct CollectiveInfo {
  CollectiveOp collective;
  SmallVector<SendOp> sends;
  SmallVector<RecvOp> recvs;
  TransferOp transfer;
};

CollectiveInfo gatherCollectiveInfo(Value token, MeshComputationOp meshOp) {
  CollectiveInfo info;
  info.collective = cast<CollectiveOp>(token.getDefiningOp());
  info.transfer = nullptr;
  for (uint32_t i = 0; i < meshOp.getNumDeviceBodies(); ++i) {
    meshOp.getDeviceBody(i).walk([&](Operation *op) {
      if (auto send = dyn_cast<SendOp>(op)) {
        if (send.getToken() == token)
          info.sends.push_back(send);
      } else if (auto recv = dyn_cast<RecvOp>(op)) {
        if (recv.getToken() == token)
          info.recvs.push_back(recv);
      }
    });
  }
  for (uint32_t i = 0; i < meshOp.getNumCommunicationBodies(); ++i) {
    meshOp.getCommunicationBody(i).walk([&](TransferOp xfer) {
      if (xfer.getToken() == token)
        info.transfer = xfer;
    });
  }
  return info;
}

// ===----------------------------------------------------------------------===
// Ordinal-based collective correspondence for clone matching.
// ===----------------------------------------------------------------------===

unsigned findCollectiveOrdinal(CollectiveOp target, MeshComputationOp meshOp) {
  unsigned idx = 0;
  for (Operation &op : meshOp->getBlock()->getOperations()) {
    if (auto coll = dyn_cast<CollectiveOp>(&op)) {
      if (coll == target)
        return idx;
      ++idx;
    }
  }
  llvm_unreachable("CollectiveOp not found");
}

CollectiveOp findCollectiveByOrdinal(unsigned targetIdx,
                                     MeshComputationOp meshOp) {
  unsigned idx = 0;
  for (Operation &op : meshOp->getBlock()->getOperations()) {
    if (auto coll = dyn_cast<CollectiveOp>(&op)) {
      if (idx == targetIdx)
        return coll;
      ++idx;
    }
  }
  return nullptr;
}

// ===----------------------------------------------------------------------===
// Determine which dimensions of the transported tensor are "free" (not
// contracted) in ALL producers feeding Sends and ALL consumers of Recvs.
// ===----------------------------------------------------------------------===

/// Collect the set of contracting dimensions for an op that consumes `val`
/// at operand position `operandIdx`.  Returns true if the op is a known
/// compute op; false if unknown (we conservatively mark all dims as used).
static bool getContractingDimsForOperand(Operation *op, unsigned operandIdx,
                                         llvm::SmallDenseSet<int64_t> &out) {
  if (auto dotGen = dyn_cast<stablehlo::DotGeneralOp>(op)) {
    auto dn = dotGen.getDotDimensionNumbers();
    ArrayRef<int64_t> contracting;
    ArrayRef<int64_t> batching;
    if (operandIdx == 0) {
      contracting = dn.getLhsContractingDimensions();
      batching = dn.getLhsBatchingDimensions();
    } else {
      contracting = dn.getRhsContractingDimensions();
      batching = dn.getRhsBatchingDimensions();
    }
    for (int64_t d : contracting)
      out.insert(d);
    // Batching dims are also "pinned" — tiling them changes semantics.
    for (int64_t d : batching)
      out.insert(d);
    return true;
  }
  if (auto dot = dyn_cast<stablehlo::DotOp>(op)) {
    // stablehlo.dot: [M,K] x [K,N] -> [M,N]
    // LHS contracting dim = 1 (last), RHS contracting dim = 0 (first).
    if (operandIdx == 0)
      out.insert(cast<RankedTensorType>(dot.getLhs().getType()).getRank() - 1);
    else
      out.insert(0);
    return true;
  }
  // Element-wise ops (broadcast, etc.) have no contracting dims; all dims free.
  if (op->hasTrait<OpTrait::Elementwise>())
    return true;
  // For DotGeneral with no contracting dims (elementwise broadcast):
  return false;
}

/// Check if `dim` of the transferred tensor type is free on the producer side
/// (i.e., the Send message producer) for every Send in the collective.
static bool isDimFreeOnProducers(int64_t dim, const CollectiveInfo &info) {
  for (SendOp send : info.sends) {
    Value msg = send.getMessage();
    Operation *producer = msg.getDefiningOp();
    if (!producer)
      return false; // block arg — can't analyze
    // Find which result of the producer this is, then check operand dims.
    // For now we check the producer's output result dims.
    auto outTy = dyn_cast<RankedTensorType>(msg.getType());
    if (!outTy || dim >= outTy.getRank())
      return false;
    // Check if the producer contracts this dim.
    llvm::SmallDenseSet<int64_t> contracted;
    if (auto dotGen = dyn_cast<stablehlo::DotGeneralOp>(producer)) {
      // Output dims of dot_general: [batch..., lhs_free..., rhs_free...]
      // None of the output dims are "contracted" — they are all free results.
      // So any output dim is tilable from the producer side.
      (void)dotGen;
    } else if (auto dot = dyn_cast<stablehlo::DotOp>(producer)) {
      (void)dot;
      // dot output [M,N] — both dims free.
    } else {
      // Unknown producer — conservatively reject.
      return false;
    }
  }
  return true;
}

/// Check if `dim` of the transferred tensor type is free on the consumer side
/// (i.e., every user of every Recv in the collective).
static bool isDimFreeOnConsumers(int64_t dim, const CollectiveInfo &info) {
  for (RecvOp recv : info.recvs) {
    for (Operation *user : recv.getMessage().getUsers()) {
      // Find which operand index this recv result feeds.
      unsigned operandIdx = 0;
      bool found = false;
      for (unsigned i = 0; i < user->getNumOperands(); ++i) {
        if (user->getOperand(i) == recv.getMessage()) {
          operandIdx = i;
          found = true;
          break;
        }
      }
      if (!found)
        return false;
      llvm::SmallDenseSet<int64_t> contractedDims;
      if (!getContractingDimsForOperand(user, operandIdx, contractedDims))
        return false; // unknown op
      if (contractedDims.contains(dim))
        return false; // this dim is contracted in a consumer
    }
  }
  return true;
}

/// Find the best dimension to tile for a collective, considering all producers
/// and consumers.  Returns -1 if no tilable dimension exists.
/// Prefers the largest dimension that is free on both sides.
int64_t findTilableDimension(CollectiveInfo &info) {
  auto localTy =
      dyn_cast<RankedTensorType>(info.collective.getLocalInputTensorType());
  if (!localTy)
    return -1;

  int64_t bestDim = -1;
  int64_t bestSize = 0;
  for (int64_t d = 0; d < localTy.getRank(); ++d) {
    int64_t sz = localTy.getShape()[d];
    if (sz <= 1)
      continue;
    if (!isDimFreeOnProducers(d, info))
      continue;
    if (!isDimFreeOnConsumers(d, info))
      continue;
    if (sz > bestSize) {
      bestSize = sz;
      bestDim = d;
    }
  }
  return bestDim;
}

// ===----------------------------------------------------------------------===
// applyTiling — tile a collective along `tileDim` by `tilingFactor`.
//
// For each Send, hoists the slice through the producer op (e.g. dot) so that
// N smaller producer ops are created, each immediately followed by its Send.
// This allows the first transfer to start as soon as the first tile is
// computed.
//
// For each Recv, clones each immediate consumer per tile and concatenates
// results so that consumers can begin as soon as each tile arrives.
// ===----------------------------------------------------------------------===

/// Slice a single tile out of `input` along `dim`.
static Value sliceOneTile(Value input, int64_t dim, int64_t tile,
                          int64_t tilingFactor, Location loc,
                          IRRewriter &rewriter) {
  auto ty = cast<RankedTensorType>(input.getType());
  int64_t fullSize = ty.getShape()[dim];
  int64_t tileSize = fullSize / tilingFactor;

  SmallVector<int64_t> starts(ty.getRank(), 0);
  SmallVector<int64_t> limits(ty.getShape().begin(), ty.getShape().end());
  SmallVector<int64_t> strides(ty.getRank(), 1);
  starts[dim] = tile * tileSize;
  limits[dim] = (tile + 1) * tileSize;
  return stablehlo::SliceOp::create(rewriter, loc, input, starts, limits,
                                    strides)
      .getResult();
}

/// Concatenate values along `dim`.
static Value concatAlongDim(ValueRange pieces, int64_t dim,
                            RankedTensorType resultType, Location loc,
                            IRRewriter &rewriter) {
  return stablehlo::ConcatenateOp::create(rewriter, loc, resultType, pieces,
                                          dim)
      .getResult();
}

// ===----------------------------------------------------------------------===
// Output-dim → input-dim mapping for dot-like ops.
//
// For a dot/dot_general producer, determines which operand and input dimension
// corresponds to a given output dimension, so we can hoist the slice above the
// producer: instead of slice(dot(A,B)) we create dot(slice(A), B).
// ===----------------------------------------------------------------------===

struct InputDimMapping {
  unsigned operandIdx; // 0 = lhs, 1 = rhs
  int64_t inputDim;
};

struct OutputDimMapping {
  int64_t outputDim;
};

static std::optional<InputDimMapping> mapOutputDimToInput(Operation *producer,
                                                          int64_t outputDim) {
  if (auto dot = dyn_cast<stablehlo::DotOp>(producer)) {
    auto lhsTy = cast<RankedTensorType>(dot.getLhs().getType());
    auto rhsTy = cast<RankedTensorType>(dot.getRhs().getType());
    if (lhsTy.getRank() == 2 && rhsTy.getRank() == 2) {
      // [M,K] x [K,N] -> [M,N]
      if (outputDim == 0)
        return InputDimMapping{0, 0}; // M → LHS dim 0
      if (outputDim == 1)
        return InputDimMapping{1, 1}; // N → RHS dim 1
    }
    return std::nullopt;
  }

  if (auto dotGen = dyn_cast<stablehlo::DotGeneralOp>(producer)) {
    auto dn = dotGen.getDotDimensionNumbers();
    auto lhsBatch = dn.getLhsBatchingDimensions();
    auto lhsContract = dn.getLhsContractingDimensions();
    auto rhsBatch = dn.getRhsBatchingDimensions();
    auto rhsContract = dn.getRhsContractingDimensions();
    auto lhsTy = cast<RankedTensorType>(dotGen.getLhs().getType());
    auto rhsTy = cast<RankedTensorType>(dotGen.getRhs().getType());

    // Output layout: [batch..., lhs_free..., rhs_free...]
    SmallVector<int64_t> lhsFree, rhsFree;
    for (int64_t d = 0; d < lhsTy.getRank(); ++d)
      if (!llvm::is_contained(lhsBatch, d) &&
          !llvm::is_contained(lhsContract, d))
        lhsFree.push_back(d);
    for (int64_t d = 0; d < rhsTy.getRank(); ++d)
      if (!llvm::is_contained(rhsBatch, d) &&
          !llvm::is_contained(rhsContract, d))
        rhsFree.push_back(d);

    int64_t numBatch = lhsBatch.size();
    int64_t numLhsFree = lhsFree.size();

    if (outputDim < numBatch)
      return InputDimMapping{0, lhsBatch[outputDim]};
    if (outputDim < numBatch + numLhsFree)
      return InputDimMapping{0, lhsFree[outputDim - numBatch]};
    int64_t rhsFreeIdx = outputDim - numBatch - numLhsFree;
    if (rhsFreeIdx < static_cast<int64_t>(rhsFree.size()))
      return InputDimMapping{1, rhsFree[rhsFreeIdx]};
    return std::nullopt;
  }

  return std::nullopt;
}

static std::optional<OutputDimMapping> mapInputDimToOutput(Operation *consumer,
                                                           unsigned operandIdx,
                                                           int64_t inputDim) {
  if (auto dot = dyn_cast<stablehlo::DotOp>(consumer)) {
    auto lhsTy = cast<RankedTensorType>(dot.getLhs().getType());
    auto rhsTy = cast<RankedTensorType>(dot.getRhs().getType());
    if (lhsTy.getRank() == 2 && rhsTy.getRank() == 2) {
      // [M,K] x [K,N] -> [M,N]
      if (operandIdx == 0 && inputDim == 0)
        return OutputDimMapping{0};
      if (operandIdx == 1 && inputDim == 1)
        return OutputDimMapping{1};
    }
    return std::nullopt;
  }

  if (auto dotGen = dyn_cast<stablehlo::DotGeneralOp>(consumer)) {
    auto dn = dotGen.getDotDimensionNumbers();
    ArrayRef<int64_t> lhsBatch = dn.getLhsBatchingDimensions();
    ArrayRef<int64_t> lhsContract = dn.getLhsContractingDimensions();
    ArrayRef<int64_t> rhsBatch = dn.getRhsBatchingDimensions();
    ArrayRef<int64_t> rhsContract = dn.getRhsContractingDimensions();
    auto lhsTy = cast<RankedTensorType>(dotGen.getLhs().getType());
    auto rhsTy = cast<RankedTensorType>(dotGen.getRhs().getType());

    SmallVector<int64_t> lhsFree, rhsFree;
    for (int64_t d = 0; d < lhsTy.getRank(); ++d)
      if (!llvm::is_contained(lhsBatch, d) &&
          !llvm::is_contained(lhsContract, d))
        lhsFree.push_back(d);
    for (int64_t d = 0; d < rhsTy.getRank(); ++d)
      if (!llvm::is_contained(rhsBatch, d) &&
          !llvm::is_contained(rhsContract, d))
        rhsFree.push_back(d);

    int64_t numBatch = lhsBatch.size();
    if (operandIdx == 0) {
      for (int64_t i = 0; i < static_cast<int64_t>(lhsBatch.size()); ++i)
        if (lhsBatch[i] == inputDim)
          return OutputDimMapping{i};
      for (int64_t i = 0; i < static_cast<int64_t>(lhsFree.size()); ++i)
        if (lhsFree[i] == inputDim)
          return OutputDimMapping{numBatch + i};
      return std::nullopt;
    }

    int64_t numLhsFree = lhsFree.size();
    for (int64_t i = 0; i < static_cast<int64_t>(rhsBatch.size()); ++i)
      if (rhsBatch[i] == inputDim)
        return OutputDimMapping{i};
    for (int64_t i = 0; i < static_cast<int64_t>(rhsFree.size()); ++i)
      if (rhsFree[i] == inputDim)
        return OutputDimMapping{numBatch + numLhsFree + i};
    return std::nullopt;
  }

  if (consumer->hasTrait<OpTrait::Elementwise>())
    return OutputDimMapping{inputDim};

  return std::nullopt;
}

void applyTiling(CollectiveOp collective, MeshComputationOp meshOp,
                 int64_t tilingFactor, int64_t tileDim, IRRewriter &rewriter) {
  CollectiveInfo info = gatherCollectiveInfo(collective.getToken(), meshOp);

  auto localInTy =
      dyn_cast<RankedTensorType>(collective.getLocalInputTensorType());
  auto localOutTy =
      dyn_cast<RankedTensorType>(collective.getLocalOutputTensorType());

  // Compute sliced local tensor types.
  SmallVector<int64_t> slicedInShape(localInTy.getShape());
  slicedInShape[tileDim] /= tilingFactor;
  auto slicedInType =
      RankedTensorType::get(slicedInShape, localInTy.getElementType());

  SmallVector<int64_t> slicedOutShape(localOutTy.getShape());
  slicedOutShape[tileDim] /= tilingFactor;
  auto slicedOutType =
      RankedTensorType::get(slicedOutShape, localOutTy.getElementType());

  Location loc = collective.getLoc();
  MLIRContext *ctx = rewriter.getContext();

  // 1. Create N new CollectiveOps.
  SmallVector<CollectiveOp> newCollectives;
  for (int64_t tile = 0; tile < tilingFactor; ++tile) {
    rewriter.setInsertionPoint(collective);
    auto newColl = CollectiveOp::create(
        rewriter, loc, MessageTokenType::get(ctx), collective.getAxes(),
        collective.getGlobalInputTensorTypeAttr(),
        collective.getGlobalOutputTensorTypeAttr(), TypeAttr::get(slicedInType),
        TypeAttr::get(slicedOutType), collective.getInputSharding(),
        collective.getOutputSharding());
    newCollectives.push_back(newColl);
  }

  // 2. Create N TransferOps, erase original.
  if (info.transfer) {
    for (int64_t tile = 0; tile < tilingFactor; ++tile) {
      rewriter.setInsertionPoint(info.transfer);
      TransferOp::create(rewriter, loc, newCollectives[tile].getToken());
    }
    rewriter.eraseOp(info.transfer);
  }

  // 3. Replace each Send: hoist slice through producer if possible.
  //    Instead of slice(dot(A,B)) → Send, create dot(slice(A),B) → Send.
  //    This creates N independent smaller ops, each immediately triggering
  //    its Send, so the first transfer starts after only 1/N of the compute.
  for (SendOp send : info.sends) {
    Value message = send.getMessage();
    Operation *producer = message.getDefiningOp();
    auto dimMapping =
        producer ? mapOutputDimToInput(producer, tileDim) : std::nullopt;

    rewriter.setInsertionPoint(send);

    for (int64_t tile = 0; tile < tilingFactor; ++tile) {
      Value tileMessage;

      if (dimMapping) {
        // Hoist slice above producer: slice the relevant input, clone the op.
        unsigned opIdx = dimMapping->operandIdx;
        int64_t inDim = dimMapping->inputDim;
        Value origInput = producer->getOperand(opIdx);

        Value slicedInput =
            sliceOneTile(origInput, inDim, tile, tilingFactor, loc, rewriter);

        IRMapping mapping;
        mapping.map(origInput, slicedInput);
        Operation *clonedProducer = rewriter.clone(*producer, mapping);

        // Update result types: tileDim shrinks by tilingFactor.
        for (unsigned r = 0; r < clonedProducer->getNumResults(); ++r) {
          if (auto resTy = dyn_cast<RankedTensorType>(
                  clonedProducer->getResult(r).getType())) {
            SmallVector<int64_t> newShape(resTy.getShape());
            newShape[tileDim] /= tilingFactor;
            clonedProducer->getResult(r).setType(
                RankedTensorType::get(newShape, resTy.getElementType()));
          }
        }
        tileMessage = clonedProducer->getResult(0);
      } else {
        // Fallback: slice the output directly.
        tileMessage =
            sliceOneTile(message, tileDim, tile, tilingFactor, loc, rewriter);
      }

      SendOp::create(rewriter, loc, newCollectives[tile].getToken(),
                     tileMessage);
    }

    rewriter.eraseOp(send);
    // Clean up dead producer if all uses were through the send.
    if (dimMapping && producer && producer->use_empty())
      rewriter.eraseOp(producer);
  }

  // 4. Replace each Recv: create N sub-Recvs and tile each consumer.
  for (RecvOp recv : info.recvs) {
    // Create N tiled Recvs.
    rewriter.setInsertionPoint(recv);
    SmallVector<Value> tileRecvResults;
    for (int64_t tile = 0; tile < tilingFactor; ++tile) {
      auto newRecv = RecvOp::create(rewriter, loc, slicedOutType,
                                    newCollectives[tile].getToken());
      tileRecvResults.push_back(newRecv.getMessage());
    }

    // For each consumer of the recv, tile it: clone the consumer per tile
    // (replacing the recv operand with the tile recv), concatenate results.
    SmallVector<Operation *> consumers(recv.getMessage().getUsers().begin(),
                                       recv.getMessage().getUsers().end());
    for (Operation *consumer : consumers) {
      unsigned recvOperandIdx = 0;
      bool foundRecvOperand = false;
      for (unsigned i = 0; i < consumer->getNumOperands(); ++i) {
        if (consumer->getOperand(i) == recv.getMessage()) {
          recvOperandIdx = i;
          foundRecvOperand = true;
          break;
        }
      }
      auto resultDimMapping =
          foundRecvOperand
              ? mapInputDimToOutput(consumer, recvOperandIdx, tileDim)
              : std::nullopt;
      if (!resultDimMapping) {
        llvm::outs() << "Skipping tiling for unsupported consumer: "
                     << consumer->getName() << "\n";
        continue;
      }

      unsigned numResults = consumer->getNumResults();
      SmallVector<SmallVector<Value>> tiledResults(numResults);

      rewriter.setInsertionPoint(consumer);
      for (int64_t tile = 0; tile < tilingFactor; ++tile) {
        IRMapping mapping;
        mapping.map(recv.getMessage(), tileRecvResults[tile]);
        Operation *cloned = rewriter.clone(*consumer, mapping);
        for (unsigned r = 0; r < numResults; ++r) {
          if (auto resTy = dyn_cast<RankedTensorType>(
                  consumer->getResult(r).getType())) {
            SmallVector<int64_t> newShape(resTy.getShape());
            if (resultDimMapping->outputDim >=
                static_cast<int64_t>(newShape.size())) {
              llvm::outs()
                  << "Skipping tiling for consumer result rank mismatch: "
                  << consumer->getName() << "\n";
              rewriter.eraseOp(cloned);
              tiledResults[r].clear();
              break;
            }
            newShape[resultDimMapping->outputDim] /= tilingFactor;
            cloned->getResult(r).setType(
                RankedTensorType::get(newShape, resTy.getElementType()));
          }
          tiledResults[r].push_back(cloned->getResult(r));
        }
      }

      for (unsigned r = 0; r < numResults; ++r) {
        if (tiledResults[r].empty())
          break;
        auto origResTy =
            cast<RankedTensorType>(consumer->getResult(r).getType());
        Value concatenated =
            concatAlongDim(tiledResults[r], resultDimMapping->outputDim,
                           origResTy, loc, rewriter);
        consumer->getResult(r).replaceAllUsesWith(concatenated);
      }
      if (!llvm::all_of(tiledResults, [](const SmallVector<Value> &values) {
            return !values.empty();
          })) {
        continue;
      }
      rewriter.eraseOp(consumer);
    }

    rewriter.eraseOp(recv);
  }

  // 5. Erase the original collective.
  rewriter.eraseOp(collective);
}

// ===----------------------------------------------------------------------===
// Simplify slice(concat(...)) patterns in a MeshComputationOp.
// When we tile both producer and consumer, the consumer clones see
//   slice(concat(tile0, tile1, ..., tileN-1))
// which should simplify to just the relevant tile.
// ===----------------------------------------------------------------------===

static void simplifySliceOfConcat(MeshComputationOp meshOp) {
  IRRewriter rewriter(meshOp.getContext());
  bool changed = true;
  while (changed) {
    changed = false;
    meshOp->walk([&](stablehlo::SliceOp sliceOp) {
      if (changed)
        return; // restart walk after mutation
      auto concat =
          sliceOp.getOperand().getDefiningOp<stablehlo::ConcatenateOp>();
      if (!concat)
        return;

      int64_t concatDim = concat.getDimension();
      auto starts = sliceOp.getStartIndices();
      auto limits = sliceOp.getLimitIndices();
      auto strides = sliceOp.getStrides();

      if (strides[concatDim] != 1)
        return;

      // Walk the concat inputs, tracking cumulative offset along concatDim.
      SmallVector<Value> pieces;
      int64_t curdim = 0;
      for (Value v : concat.getInputs()) {
        auto ty = cast<RankedTensorType>(v.getType());
        int64_t nextdim = ty.getShape()[concatDim];
        if (starts[concatDim] >= curdim + nextdim) {
          curdim += nextdim;
          continue;
        }
        if (limits[concatDim] <= curdim) {
          curdim += nextdim;
          continue;
        }
        SmallVector<int64_t> nstart(starts.begin(), starts.end());
        SmallVector<int64_t> nend(limits.begin(), limits.end());
        nstart[concatDim] -= curdim;
        if (nstart[concatDim] < 0)
          nstart[concatDim] = 0;
        nend[concatDim] -= curdim;
        if (nend[concatDim] > nextdim)
          nend[concatDim] = nextdim;

        // Check if this is an identity slice (takes the full input).
        bool isIdentity = true;
        for (int64_t d = 0; d < ty.getRank(); ++d) {
          if (nstart[d] != 0 || nend[d] != ty.getShape()[d] ||
              strides[d] != 1) {
            isIdentity = false;
            break;
          }
        }
        if (isIdentity) {
          pieces.push_back(v);
        } else {
          rewriter.setInsertionPoint(sliceOp);
          auto subSlice = stablehlo::SliceOp::create(rewriter, sliceOp.getLoc(),
                                                     v, nstart, nend, strides);
          pieces.push_back(subSlice.getResult());
        }
        curdim += nextdim;
      }

      if (pieces.size() == 1) {
        rewriter.replaceOp(sliceOp, pieces[0]);
        changed = true;
      } else if (!pieces.empty()) {
        rewriter.setInsertionPoint(sliceOp);
        auto newConcat = stablehlo::ConcatenateOp::create(
            rewriter, sliceOp.getLoc(), sliceOp.getType(), pieces, concatDim);
        rewriter.replaceOp(sliceOp, newConcat.getResult());
        changed = true;
      }
    });
  }
}

/// Remove identity slices (slice that takes the full tensor).
static void simplifyNoopSlices(MeshComputationOp meshOp) {
  IRRewriter rewriter(meshOp.getContext());
  bool changed = true;
  while (changed) {
    changed = false;
    meshOp->walk([&](stablehlo::SliceOp sliceOp) {
      if (changed)
        return;
      auto inTy = cast<RankedTensorType>(sliceOp.getOperand().getType());
      auto starts = sliceOp.getStartIndices();
      auto limits = sliceOp.getLimitIndices();
      auto strides = sliceOp.getStrides();
      bool isIdentity = true;
      for (int64_t d = 0; d < inTy.getRank(); ++d) {
        if (starts[d] != 0 || limits[d] != inTy.getShape()[d] ||
            strides[d] != 1) {
          isIdentity = false;
          break;
        }
      }
      if (isIdentity) {
        rewriter.replaceOp(sliceOp, sliceOp.getOperand());
        changed = true;
      }
    });
  }
}

// ===----------------------------------------------------------------------===
// findBestTilingFactor — clone-and-measure search.
// ===----------------------------------------------------------------------===

int64_t findBestTilingFactor(CollectiveOp collective, int64_t tileDim,
                             int64_t kMax, MeshComputationOp meshOp) {
  auto localTy =
      dyn_cast<RankedTensorType>(collective.getLocalInputTensorType());
  int64_t dimSize = localTy.getShape()[tileDim];

  int64_t bestFactor = 2; // DEBUG: force tiling to see IR transformation
  double bestTime = std::numeric_limits<double>::max();

  // Factor 1 (no tiling): measure current state directly.
  {
    TimingResult timing = analyzeTiming(meshOp);
    llvm::outs() << "  k=0 factor=1 time=" << timing.criticalPathTime << "\n";
    // DEBUG: don't use factor 1 as baseline
    // bestTime = timing.criticalPathTime;
  }

  unsigned collectiveIdx = findCollectiveOrdinal(collective, meshOp);
  auto funcOp = meshOp->getParentOfType<func::FuncOp>();
  if (!funcOp)
    return bestFactor;

  for (int64_t exp = 1; exp <= kMax; ++exp) {
    int64_t factor = 1LL << exp;
    if (dimSize < factor || dimSize % factor != 0)
      break;

    // Deep-clone the entire enclosing function.
    auto clonedFunc = cast<func::FuncOp>(funcOp->clone());

    MeshComputationOp clonedMeshOp = nullptr;
    clonedFunc.walk([&](MeshComputationOp op) { clonedMeshOp = op; });
    if (!clonedMeshOp) {
      clonedFunc->erase();
      continue;
    }

    CollectiveOp clonedCollective =
        findCollectiveByOrdinal(collectiveIdx, clonedMeshOp);
    if (!clonedCollective) {
      clonedFunc->erase();
      continue;
    }

    // Apply tiling to the clone.
    IRRewriter rewriter(clonedMeshOp.getContext());
    applyTiling(clonedCollective, clonedMeshOp, factor, tileDim, rewriter);

    // Simplify slice(concat) patterns so timing sees actual tiled ops.
    simplifySliceOfConcat(clonedMeshOp);
    simplifyNoopSlices(clonedMeshOp);

    // Sink recvs so timing sees the reduced latency from earlier recv
    // placement.
    sinkRecvs(clonedMeshOp);

    // Measure critical path on the tiled clone.
    TimingResult timing = analyzeTiming(clonedMeshOp);
    llvm::outs() << "  k=" << exp << " factor=" << factor
                 << " time=" << timing.criticalPathTime << "\n";
    if (timing.criticalPathTime < bestTime) {
      bestTime = timing.criticalPathTime;
      bestFactor = factor;
    }

    clonedFunc->erase();
  }

  return bestFactor;
}

} // end anonymous namespace

// ===----------------------------------------------------------------------===
// runOverlapCommunication — entry point.
// ===----------------------------------------------------------------------===

void runOverlapCommunication(MeshComputationOp meshOp, int64_t kMaxVal) {
  // Gather all collectives that have a Transfer (i.e., actual communication).
  SmallVector<CollectiveOp> collectives;
  for (uint32_t i = 0; i < meshOp.getNumCommunicationBodies(); ++i) {
    meshOp.getCommunicationBody(i).walk([&](TransferOp xfer) {
      if (auto coll = xfer.getToken().getDefiningOp<CollectiveOp>())
        collectives.push_back(coll);
    });
  }

  if (collectives.empty()) {
    llvm::outs() << "No collectives with transfers found.\n";
    return;
  }

  llvm::outs() << "Found " << collectives.size() << " collectives.\n";

  for (CollectiveOp collective : collectives) {
    CollectiveInfo info = gatherCollectiveInfo(collective.getToken(), meshOp);

    int64_t tileDim = findTilableDimension(info);
    if (tileDim < 0) {
      llvm::outs() << "No tilable dimension found for collective.\n";
      continue;
    }

    llvm::outs() << "Searching tiling factors for dim " << tileDim
                 << " (tensor: ";
    auto localTy =
        dyn_cast<RankedTensorType>(collective.getLocalInputTensorType());
    if (localTy)
      llvm::outs() << localTy;
    llvm::outs() << ")...\n";

    int64_t bestFactor =
        findBestTilingFactor(collective, tileDim, kMaxVal, meshOp);

    if (bestFactor > 1) {
      IRRewriter rewriter(meshOp.getContext());
      applyTiling(collective, meshOp, bestFactor, tileDim, rewriter);
      // Also simplify the real IR.
      simplifySliceOfConcat(meshOp);
      simplifyNoopSlices(meshOp);
      llvm::outs() << "Applied tiling factor " << bestFactor << ".\n";
    } else {
      llvm::outs() << "No improvement; using factor 1 (no tiling).\n";
    }
  }
}

namespace {

struct DistributedOverlapCommunicationPass
    : public enzyme::distributed::impl::DistributedOverlapCommunicationPassBase<
          DistributedOverlapCommunicationPass> {
  using DistributedOverlapCommunicationPassBase::
      DistributedOverlapCommunicationPassBase;

  void runOnOperation() override {
    runOverlapCommunication(getOperation(), kMax);
  }
};

struct DistributedOverlapCommunicationModulePass
    : public enzyme::distributed::impl::
          DistributedOverlapCommunicationModulePassBase<
              DistributedOverlapCommunicationModulePass> {
  using DistributedOverlapCommunicationModulePassBase::
      DistributedOverlapCommunicationModulePassBase;

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    SmallVector<MeshComputationOp> meshOps;
    moduleOp.walk([&](MeshComputationOp meshOp) { meshOps.push_back(meshOp); });
    for (MeshComputationOp meshOp : meshOps)
      runOverlapCommunication(meshOp, kMax);
  }
};

} // namespace

} // namespace distributed
} // namespace enzyme
} // namespace mlir
