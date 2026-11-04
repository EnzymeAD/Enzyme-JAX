#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h"

#include "src/enzyme_ad/jax/Dialect/Distributed/CollectiveDecomposer.h"
#include "src/enzyme_ad/jax/Dialect/Distributed/Dialect.h"
#include "src/enzyme_ad/jax/Dialect/Distributed/MeshCostMetadata.h"
#include "src/enzyme_ad/jax/Dialect/Distributed/Utilities.h"
#include "src/enzyme_ad/jax/Passes/Distributed/CollectivePrimitiveBuilders.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "llvm/ADT/STLExtras.h"

#include <set>

namespace mlir::enzyme::distributed {

#define GEN_PASS_DEF_LOWERCOLLECTIVESPASS
#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h.inc"

namespace {

// A peel (D17) whose reduce-scatter has run and whose all-gather has not. The
// scatter shrank `digit`'s dimension by the atom's extent; the gather on the
// same atom grows it back.
struct OpenPeel {
  StepAtom atom;
  int digit;
};

bool isSameAtom(const StepAtom &lhs, const StepAtom &rhs) {
  return lhs.axis == rhs.axis && lhs.atom == rhs.atom;
}

bool isShardedByPeel(ArrayRef<OpenPeel> peels, int digit) {
  return llvm::any_of(
      peels, [&](const OpenPeel &peel) { return peel.digit == digit; });
}

// The chain tensor's shape between steps: one dimension per local digit, in
// the model's order. A digit an open peel shards has the extent of the shard.
SmallVector<int64_t> digitShape(const ChainDigitModel &model,
                                ArrayRef<OpenPeel> peels) {
  SmallVector<int64_t> shape;
  for (int digit : model.local()) {
    int64_t extent = static_cast<int64_t>(model.extent(digit));
    for (const OpenPeel &peel : peels)
      if (peel.digit == digit)
        extent /= static_cast<int64_t>(peel.atom.extent);
    shape.push_back(extent);
  }
  return shape;
}

SmallVector<int64_t> withUnitDimension(SmallVector<int64_t> shape,
                                       size_t position) {
  shape.insert(shape.begin() + position, 1);
  return shape;
}

// The local digit that the peel opened by chain[scatterIndex] shards: the
// first, in local order, whose extent the peel atom's extent divides, that no
// other open peel shards, and that stays local until the all-gather on the
// same atom closes the peel. The steps in between then find every digit they
// act on whole. Returns nullopt when no local digit qualifies.
std::optional<int> choosePeelDigit(const ChainDigitModel &model,
                                   ArrayRef<PrimitiveStep> chain,
                                   size_t scatterIndex,
                                   ArrayRef<OpenPeel> peels) {
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
  for (int digit : model.local())
    if (model.extent(digit) % atom.extent == 0 && !leaving.count(digit) &&
        !isShardedByPeel(peels, digit))
      return digit;
  return std::nullopt;
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
// any shape. A peel's scatter shrinks one local digit's dimension
// (choosePeelDigit) until its gather restores it. If the surviving digits are
// not in the output tile's order, one transpose fixes that before the final
// reshape to the result type. All of these reshapes only regroup row-major
// elements, so they move no data.
//
// An empty chain moves nothing between devices, but may still relabel the
// tile's digits: it lowers to that transpose and reshape alone, or to the
// input itself when the layouts already agree.
//
// Fails with a diagnostic on the collective when a peel has no local digit to
// shard.
LogicalResult lowerCollectivePlan(const CollectivePlan &plan,
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
  SmallVector<OpenPeel> peels;
  Value current = plan.inputPort;
  auto toDigitShape = [&](Value tensor) {
    return reshapeToDigits(builder, loc, tensor, digitShape(*model, peels));
  };
  auto toDigitShapeWithUnitDimension = [&](Value tensor, size_t position) {
    return reshapeToDigits(
        builder, loc, tensor,
        withUnitDimension(digitShape(*model, peels), position));
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
      current = toDigitShape(current);
      std::optional<int> digit = model->digitToPlace(step);
      if (!digit) {
        digit = choosePeelDigit(*model, plan.chain, i, peels);
        if (!digit) {
          const StepAtom &atom = step.atoms.front();
          return collective.emitError()
                 << "distributed-lower-collectives: cannot lower the peeled "
                    "all-reduce over mesh axis "
                 << atom.axis << " atom " << atom.atom
                 << ": no local digit that stays local until the peel closes "
                    "is divisible by its extent "
                 << atom.extent;
        }
        peels.push_back({step.atoms.front(), *digit});
      }
      current = buildReduceScatterStep(builder, loc, current, step,
                                       model->dimensionOf(*digit),
                                       meshAxisTypes, *reductionBody);
      break;
    }
    case PrimitiveKind::AllGather: {
      std::optional<int> gathered = model->digitToGather(step);
      if (!gathered) {
        current = toDigitShape(current);
        auto peel = llvm::find_if(peels, [&](const OpenPeel &open) {
          return isSameAtom(open.atom, step.atoms.front());
        });
        assert(peel != peels.end() &&
               "a gather of an empty atom closes a peel");
        int64_t dimension = model->dimensionOf(peel->digit);
        peels.erase(peel);
        current = buildAllGatherStep(builder, loc, current, step, dimension,
                                     meshAxisTypes);
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
    current = builder.create<stablehlo::TransposeOp>(
        loc, toDigitShape(current), builder.getDenseI64ArrayAttr(permutation));
  }
  Value result = reshapeToDigits(
      builder, loc, current,
      cast<RankedTensorType>(await.getValue().getType()).getShape());
  await.getValue().replaceAllUsesWith(result);
  await.erase();
  collective.erase();
  return success();
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
      if (failed(lowerCollectivePlan(*plan, meshAxisTypes, builder))) {
        signalPassFailure();
        return;
      }
    }
  }
};

} // namespace

} // namespace mlir::enzyme::distributed
