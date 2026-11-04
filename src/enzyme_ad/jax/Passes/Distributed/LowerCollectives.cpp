#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h"

#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "src/enzyme_ad/jax/Dialect/Distributed/CollectiveDecomposer.h"
#include "src/enzyme_ad/jax/Dialect/Distributed/Dialect.h"
#include "src/enzyme_ad/jax/Dialect/Distributed/MeshCostMetadata.h"
#include "src/enzyme_ad/jax/Dialect/Distributed/Utilities.h"
#include "src/enzyme_ad/jax/Passes/Distributed/CollectivePrimitiveBuilders.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "llvm/ADT/STLExtras.h"

#include <map>
#include <numeric>
#include <set>

namespace mlir::enzyme::distributed {

#define GEN_PASS_DEF_LOWERCOLLECTIVESPASS
#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h.inc"

namespace {

bool isSameAtom(const StepAtom &lhs, const StepAtom &rhs) {
  return lhs.axis == rhs.axis && lhs.atom == rhs.atom;
}

// How the open peels (D17) shard the local digits. A peel's reduce-scatter
// keeps on each device the piece of the tensor at the device's own coordinate
// of the atom, and its all-gather restores it. That shard may come from
// several digits, or from part of one, so a digit is split into pieces,
// major-first: its index is the row-major index over its pieces. Each piece is
// either in the chain's tensor or held by one open peel. Invariants:
//   - a digit's piece extents multiply to its extent;
//   - no two adjacent pieces are both in the tensor (they would be one piece);
//   - a digit without an entry is one piece, in the tensor.
// A digit's dimension in the tensor therefore holds its pieces that are in the
// tensor, in order. The model's digits themselves never change.
class PeelShards {
public:
  // One peel's reduce-scatter operand, or all-gather result, as dimensions.
  struct View {
    // Every local digit split into its pieces in the tensor plus the peel's.
    SmallVector<int64_t> pieceShape;
    // The transpose of pieceShape that makes the peel's pieces adjacent, at
    // the position of its last piece; the other pieces keep their order.
    SmallVector<int64_t> order;
    // pieceShape in `order`, with the peel's pieces and the piece after them
    // merged into one dimension, `dimension`, that the peel shards. Merging
    // the following piece means a shard taken from the major part of one
    // digit scatters that digit's own dimension.
    SmallVector<int64_t> mergedShape;
    int64_t dimension;
  };

  // `digit`'s extent in the chain tensor: its pieces no open peel holds.
  uint64_t tensorExtent(const ChainDigitModel &model, int digit) const {
    auto it = piecesOf.find(digit);
    if (it == piecesOf.end())
      return model.extent(digit);
    uint64_t extent = 1;
    for (const Piece &piece : it->second)
      if (piece.holder == kInTensor)
        extent *= piece.extent;
    return extent;
  }

  // Takes a shard of `extent` for peel `peel` from the tensor pieces of
  // `freeDigits` (in local order), from the donors chooseDonors picks. Each
  // tensor piece gives the largest factor that still divides what is left of
  // `extent`, as its major part. That is a choice of prime factors of the
  // digits' extents whose product is `extent`. Returns false, changing
  // nothing, when no such choice exists.
  bool take(const ChainDigitModel &model, ArrayRef<int> freeDigits,
            uint64_t extent, int peel) {
    assert(extent > 1 && "a peel's atom has extent above one");
    std::optional<SmallVector<int>> donors =
        chooseDonors(model, freeDigits, extent);
    if (!donors)
      return false;

    uint64_t remaining = extent;
    for (int digit : *donors) {
      SmallVector<Piece> &pieces = piecesFor(model, digit);
      SmallVector<Piece> split;
      for (const Piece &piece : pieces) {
        uint64_t shard =
            piece.holder == kInTensor ? std::gcd(remaining, piece.extent) : 1;
        if (shard == 1) {
          split.push_back(piece);
          continue;
        }
        remaining /= shard;
        split.push_back({shard, peel});
        if (piece.extent != shard)
          split.push_back({piece.extent / shard, kInTensor});
      }
      pieces = std::move(split);
    }
    return true;
  }

  // Returns peel `peel`'s pieces to the tensor.
  void release(int peel) {
    for (auto it = piecesOf.begin(); it != piecesOf.end();) {
      SmallVector<Piece> merged;
      for (Piece piece : it->second) {
        if (piece.holder == peel)
          piece.holder = kInTensor;
        if (!merged.empty() && merged.back().holder == kInTensor &&
            piece.holder == kInTensor)
          merged.back().extent *= piece.extent;
        else
          merged.push_back(piece);
      }
      if (merged.size() == 1 && merged.front().holder == kInTensor)
        it = piecesOf.erase(it);
      else
        (it++)->second = std::move(merged);
    }
  }

  View view(const ChainDigitModel &model, int peel) const {
    View view;
    SmallVector<bool> isShard;
    int64_t last = -1;
    auto addPiece = [&](uint64_t extent, bool shard) {
      if (shard)
        last = isShard.size();
      view.pieceShape.push_back(static_cast<int64_t>(extent));
      isShard.push_back(shard);
    };
    for (int digit : model.local()) {
      auto it = piecesOf.find(digit);
      if (it == piecesOf.end()) {
        addPiece(model.extent(digit), false);
        continue;
      }
      for (const Piece &piece : it->second)
        if (piece.holder == kInTensor || piece.holder == peel)
          addPiece(piece.extent, piece.holder == peel);
    }
    assert(last >= 0 && "an open peel holds a piece of a local digit");

    for (int64_t position = 0; position < last; ++position)
      if (!isShard[position])
        view.order.push_back(position);
    for (int64_t position = 0; position <= last; ++position)
      if (isShard[position])
        view.order.push_back(position);
    for (int64_t position = last + 1;
         position < static_cast<int64_t>(isShard.size()); ++position)
      view.order.push_back(position);

    // The peel's pieces and the one after them are adjacent in `order`.
    view.dimension = -1;
    for (int64_t position : view.order) {
      if (!isShard[position] && position != last + 1) {
        view.mergedShape.push_back(view.pieceShape[position]);
        continue;
      }
      if (view.dimension < 0) {
        view.dimension = view.mergedShape.size();
        view.mergedShape.push_back(1);
      }
      view.mergedShape[view.dimension] *= view.pieceShape[position];
    }
    return view;
  }

private:
  static constexpr int kInTensor = -1;
  struct Piece {
    uint64_t extent;
    // The open peel holding the piece, or kInTensor.
    int holder;
  };

  // The digits a shard of `extent` comes from, preferring choices that need
  // no transpose:
  //   1. the first free digit whose tensor extent `extent` divides;
  //   2. a run of free digits adjacent in local order, all but the last taken
  //      whole and the last giving its major part;
  //   3. every free digit, in local order.
  // nullopt when even all of them together are not divisible by `extent`.
  std::optional<SmallVector<int>> chooseDonors(const ChainDigitModel &model,
                                               ArrayRef<int> freeDigits,
                                               uint64_t extent) const {
    for (int digit : freeDigits)
      if (tensorExtent(model, digit) % extent == 0)
        return SmallVector<int>{digit};
    for (size_t start = 0; start < freeDigits.size(); ++start) {
      uint64_t remaining = extent;
      for (size_t index = start; index < freeDigits.size(); ++index) {
        int digit = freeDigits[index];
        if (index > start && model.dimensionOf(digit) !=
                                 model.dimensionOf(freeDigits[index - 1]) + 1)
          break;
        uint64_t digitExtent = tensorExtent(model, digit);
        if (digitExtent % remaining == 0)
          return SmallVector<int>(freeDigits.begin() + start,
                                  freeDigits.begin() + index + 1);
        if (remaining % digitExtent != 0)
          break;
        remaining /= digitExtent;
      }
    }
    uint64_t remaining = extent;
    for (int digit : freeDigits)
      remaining /= std::gcd(remaining, tensorExtent(model, digit));
    if (remaining != 1)
      return std::nullopt;
    return llvm::to_vector(freeDigits);
  }

  SmallVector<Piece> &piecesFor(const ChainDigitModel &model, int digit) {
    auto [it, inserted] = piecesOf.try_emplace(digit);
    if (inserted)
      it->second.push_back({model.extent(digit), kInTensor});
    return it->second;
  }

  std::map<int, SmallVector<Piece>> piecesOf;
};

// A peel whose reduce-scatter has run and whose all-gather on the same atom
// has not. `sharded` is false when it was lowered as an all-reduce, because
// its free digits could not supply the atom's extent.
struct OpenPeel {
  StepAtom atom;
  int id;
  bool sharded;
};

// The chain tensor's shape between steps: one dimension per local digit, in
// the model's order, holding the digit's pieces no open peel holds.
SmallVector<int64_t> digitShape(const ChainDigitModel &model,
                                const PeelShards &shards) {
  SmallVector<int64_t> shape;
  for (int digit : model.local())
    shape.push_back(static_cast<int64_t>(shards.tensorExtent(model, digit)));
  return shape;
}

SmallVector<int64_t> withUnitDimension(SmallVector<int64_t> shape,
                                       size_t position) {
  shape.insert(shape.begin() + position, 1);
  return shape;
}

// The local digits, in local order, that stay local from the peel opened by
// chain[scatterIndex] until the all-gather on the same atom closes it. The
// steps in between then find every digit they act on whole.
SmallVector<int> peelFreeDigits(const ChainDigitModel &model,
                                ArrayRef<PrimitiveStep> chain,
                                size_t scatterIndex) {
  const StepAtom &atom = chain[scatterIndex].atoms.front();
  std::set<int> leaving;
  size_t close = scatterIndex + 1;
  for (; close < chain.size(); ++close) {
    const PrimitiveStep &step = chain[close];
    if (step.kind == PrimitiveKind::AllGather &&
        isSameAtom(step.atoms.front(), atom))
      break;
    if (std::optional<int> placed = model.digitToPlace(step))
      leaving.insert(*placed);
  }
  assert(close < chain.size() && "an all-gather on its atom closes a peel");
  SmallVector<int> free;
  for (int digit : model.local())
    if (!leaving.count(digit))
      free.push_back(digit);
  return free;
}

Value transposeTo(OpBuilder &builder, Location loc, Value tensor,
                  ArrayRef<int64_t> permutation) {
  return builder.create<stablehlo::TransposeOp>(
      loc, tensor, builder.getDenseI64ArrayAttr(permutation));
}

// A peel's reduce-scatter on `current`, a tensor whose elements are in
// `view.pieceShape`'s row-major layout (any shape): the peel's pieces are moved
// together and merged into one dimension, which the scatter shards. The result
// has the pieces not held by the peel, in their order.
Value buildPeelScatter(OpBuilder &builder, Location loc, Value current,
                       const PrimitiveStep &step, const PeelShards::View &view,
                       ArrayRef<PhysicalCommAxisType> meshAxisTypes,
                       Region &reductionBody) {
  if (!isIdentityPermutation(view.order))
    current = transposeTo(
        builder, loc, reshapeToDigits(builder, loc, current, view.pieceShape),
        view.order);
  current = reshapeToDigits(builder, loc, current, view.mergedShape);
  return buildReduceScatterStep(builder, loc, current, step, view.dimension,
                                meshAxisTypes, reductionBody);
}

// The all-gather closing a peel, the inverse of buildPeelScatter: `current`
// lacks the peel's pieces (in any shape), and the result is in
// `view.pieceShape`'s layout.
Value buildPeelGather(OpBuilder &builder, Location loc, Value current,
                      const PrimitiveStep &step, const PeelShards::View &view,
                      ArrayRef<PhysicalCommAxisType> meshAxisTypes) {
  SmallVector<int64_t> operandShape = view.mergedShape;
  operandShape[view.dimension] /=
      static_cast<int64_t>(step.atoms.front().extent);
  current = buildAllGatherStep(
      builder, loc, reshapeToDigits(builder, loc, current, operandShape), step,
      view.dimension, meshAxisTypes);
  if (isIdentityPermutation(view.order))
    return current;
  return transposeTo(
      builder, loc,
      reshapeToDigits(
          builder, loc, current,
          applyPermutation(ArrayRef<int64_t>(view.pieceShape), view.order)),
      invertPermutationVector(view.order));
}

// Replaces `plan`'s distributed.Collective/Await pair with the real
// StableHLO op chain its decomposition calls for: one cost-tagged op per
// PrimitiveStep (CollectivePrimitiveBuilders.h), each step's result threaded
// into the next step's operand.
//
// The chain runs in digit shape (see CollectivePrimitiveBuilders.h's file
// comment), following the plan's ChainDigitModel. A step that acts on a digit
// first reshapes its operand to one dimension per local digit (digitShape);
// a gather, or an all-to-all's concat half, also gets an extent-1 dimension
// where the model inserts the gathered digit. All-reduces and permutes take
// any shape. If the surviving digits are not in the output tile's order, one
// transpose fixes that before the final reshape to the result type. All of
// these reshapes only regroup row-major elements, so they move no data.
//
// An empty chain moves nothing between devices, but may still relabel the
// tile's digits: it lowers to that transpose and reshape alone, or to the
// input itself when the layouts already agree.
//
// A peel's scatter shards prime factors of the digits that stay local until
// its gather (peelFreeDigits, PeelShards::take), merged into one dimension,
// and its gather restores them. When those digits cannot supply the atom's
// extent, as for a tile already sharded down to scalars, the peel is lowered
// as an all-reduce at the scatter and nothing at the gather. That moves more
// data than the plan's cost assumes, but is correct.
void lowerCollectivePlan(const CollectivePlan &plan,
                         ArrayRef<PhysicalCommAxisType> meshAxisTypes,
                         OpBuilder &builder) {
  DistributedCollectiveOp collective = plan.collective;
  DistributedAwait await = plan.await;
  Location loc = collective.getLoc();
  builder.setInsertionPoint(collective);

  // Every AllReduce/ReduceScatter step in the chain shares one reduction
  // semantics (CollectiveDecomposer.cpp asserts every reduction_bodies
  // region classifies to the same ReductionKind), so the first reduction
  // body stands in for all of them.
  Region *reductionBody = collective.getReductionBodies().empty()
                              ? nullptr
                              : &collective.getReductionBodies()[0];

  std::string why;
  std::optional<ChainDigitModel> model =
      ChainDigitModel::create(plan.normalized, why);
  assert(model && "a planned collective has a digit model");
  PeelShards shards;
  SmallVector<OpenPeel> peels;
  int nextPeelId = 0;
  Value current = plan.inputPort;
  auto toDigitShape = [&](Value tensor) {
    return reshapeToDigits(builder, loc, tensor, digitShape(*model, shards));
  };
  auto toDigitShapeWithUnitDimension = [&](Value tensor, size_t position) {
    return reshapeToDigits(
        builder, loc, tensor,
        withUnitDimension(digitShape(*model, shards), position));
  };
  for (size_t i = 0; i < plan.chain.size(); ++i) {
    const PrimitiveStep &step = plan.chain[i];
    switch (step.kind) {
    case PrimitiveKind::AllReduce:
      assert(reductionBody && "an AllReduce step implies a reduction group");
      current = buildAllReduceStep(builder, loc, current, step, meshAxisTypes,
                                   *reductionBody);
      break;
    case PrimitiveKind::ReduceScatter: {
      assert(reductionBody && "a ReduceScatter step implies a reduction group");
      if (std::optional<int> digit = model->digitToPlace(step)) {
        current = buildReduceScatterStep(builder, loc, toDigitShape(current),
                                         step, model->dimensionOf(*digit),
                                         meshAxisTypes, *reductionBody);
        break;
      }
      const StepAtom &atom = step.atoms.front();
      int id = nextPeelId++;
      bool sharded = shards.take(*model, peelFreeDigits(*model, plan.chain, i),
                                 atom.extent, id);
      peels.push_back({atom, id, sharded});
      if (!sharded) {
        current = buildAllReduceStep(builder, loc, current, step, meshAxisTypes,
                                     *reductionBody);
        break;
      }
      current = toDigitShape(buildPeelScatter(builder, loc, current, step,
                                              shards.view(*model, id),
                                              meshAxisTypes, *reductionBody));
      break;
    }
    case PrimitiveKind::AllGather: {
      std::optional<int> gathered = model->digitToGather(step);
      if (!gathered) {
        auto peel = llvm::find_if(peels, [&](const OpenPeel &open) {
          return isSameAtom(open.atom, step.atoms.front());
        });
        assert(peel != peels.end() &&
               "a gather of an empty atom closes a peel");
        if (peel->sharded) {
          current =
              buildPeelGather(builder, loc, current, step,
                              shards.view(*model, peel->id), meshAxisTypes);
          shards.release(peel->id);
          current = toDigitShape(current);
        }
        peels.erase(peel);
        break;
      }
      size_t position = model->insertPosition(*gathered);
      current = toDigitShapeWithUnitDimension(current, position);
      current = buildAllGatherStep(builder, loc, current, step, position,
                                   meshAxisTypes);
      break;
    }
    case PrimitiveKind::AllToAll: {
      size_t position = model->insertPosition(*model->digitToGather(step));
      size_t split = model->dimensionOf(*model->digitToPlace(step));
      if (split >= position)
        ++split;
      current = toDigitShapeWithUnitDimension(current, position);
      current = buildAllToAllStep(builder, loc, current, step, split, position,
                                  meshAxisTypes);
      break;
    }
    case PrimitiveKind::Permute:
      current = buildPermuteStep(builder, loc, current, step, meshAxisTypes,
                                 plan.normalized);
      break;
    case PrimitiveKind::LocalSlice:
      current = buildLocalSliceStep(
          builder, loc, toDigitShape(current), step,
          model->dimensionOf(*model->digitToPlace(step)), meshAxisTypes);
      break;
    }
    bool applied = model->apply(step, why);
    assert(applied && "a planned chain realizes its collective");
    (void)applied;
  }

  const std::vector<int> &outputOrder = model->outputOrder();
  if (model->local() != outputOrder) {
    SmallVector<int64_t> permutation;
    for (int digit : outputOrder)
      permutation.push_back(model->dimensionOf(digit));
    current = transposeTo(builder, loc, toDigitShape(current), permutation);
  }
  Value result = reshapeToDigits(
      builder, loc, current,
      cast<RankedTensorType>(await.getValue().getType()).getShape());
  await.getValue().replaceAllUsesWith(result);
  await.erase();
  collective.erase();
}

struct LowerCollectivesPass
    : public impl::LowerCollectivesPassBase<LowerCollectivesPass> {
  using LowerCollectivesPassBase::LowerCollectivesPassBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();

    FailureOr<PhysicalMeshOp> physicalMesh = findUniquePhysicalMesh(module);
    if (failed(physicalMesh)) {
      // findUniquePhysicalMesh already emitted a diagnostic.
      signalPassFailure();
      return;
    }
    SmallVector<PhysicalCommAxisType> meshAxisTypes;
    for (Attribute axisAttr : physicalMesh->getAxesAttr())
      meshAxisTypes.push_back(
          cast<PhysicalCommAxisType>(cast<TypeAttr>(axisAttr).getValue()));
    MeshCostParams params = meshCostParamsFromPhysicalMesh(*physicalMesh);

    // Collect before mutating: lowering a plan erases its Collective/Await
    // pair, which would invalidate an in-flight walk.
    SmallVector<DistributedCollectiveOp> collectives;
    module.walk([&](DistributedCollectiveOp collective) {
      collectives.push_back(collective);
    });

    OpBuilder builder(&getContext());
    for (DistributedCollectiveOp collective : collectives) {
      std::string failureReason;
      std::optional<CollectivePlan> plan = planCollective(
          collective, meshAxisTypes, params, failureReason, PlanOptions{});
      if (!plan) {
        // A hard pass failure, not a diagnostic recorded for the caller to
        // inspect: a search candidate this happens on simply fails to lower
        // (StrategyScorer::score already treats a failed lowering pipeline
        // as an infeasible, -infinity-scored candidate), and the eventual
        // winner is never expected to reach this pass with an unplannable
        // collective at all, since it was already lowered successfully once
        // during scoring.
        collective.emitError()
            << "distributed-lower-collectives: could not plan collective ("
            << failureReason << ")";
        signalPassFailure();
        return;
      }
      lowerCollectivePlan(*plan, meshAxisTypes, builder);
    }
  }
};

} // namespace

} // namespace mlir::enzyme::distributed
