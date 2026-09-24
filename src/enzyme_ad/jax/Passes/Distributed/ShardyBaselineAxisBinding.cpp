#include "src/enzyme_ad/jax/Passes/Distributed/ShardyLogicalAxisAnalysis.h"

#include "src/enzyme_ad/jax/Dialect/Axis/Utilities.h"
#include "src/enzyme_ad/jax/Dialect/Distributed/Utilities.h"

#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/utils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"

// Shardy baseline mode of ShardyLogicalAxisAnalysis.
//
// The analysis normally leaves every symbol for a later search to place. Here
// the expert Shardy plan is the placement: each tensor dimension is
// factorized as [physical digit ranges Shardy shards it over..., remainder],
// and that factorization is merged into the dimension's symbols before
// propagation runs. Propagation then extends the same factorization across
// the program, and the result is checked against every Shardy sharding.
//
// Invariants relied on by the checks below:
//  - Each dimension's sharded factors form its major prefix. A device-local
//    factor ahead of a sharded one would need a transpose to express, which
//    the lowering pipeline does not do.
//  - A root symbol has at most one physical digit range.
//  - A root symbol that Shardy shards is not one the compiler must keep
//    device-local (permutation and need-replication factors).

namespace mlir::enzyme::distributed {

namespace {

bool sameFactor(const PhysicalFactorRef &a, const PhysicalFactorRef &b) {
  return a.axisIndex == b.axisIndex && a.extent == b.extent &&
         a.strideWithinAxis == b.strideWithinAxis;
}

// Merges major-first neighbors that are adjacent digit ranges of one axis, so
// two factorizations of the same digits compare equal.
llvm::SmallVector<PhysicalFactorRef>
coalesce(llvm::ArrayRef<PhysicalFactorRef> factors) {
  llvm::SmallVector<PhysicalFactorRef> result;
  for (const PhysicalFactorRef &factor : factors) {
    if (!result.empty() && result.back().axisIndex == factor.axisIndex &&
        result.back().strideWithinAxis ==
            factor.strideWithinAxis * factor.extent) {
      result.back().extent *= factor.extent;
      result.back().strideWithinAxis = factor.strideWithinAxis;
      continue;
    }
    result.push_back(factor);
  }
  return result;
}

// Whether `prefix` (coalesced) is the major-most part of the digits `whole`
// (coalesced) covers, with digits left over.
bool isProperMajorPrefix(llvm::ArrayRef<PhysicalFactorRef> prefix,
                         llvm::ArrayRef<PhysicalFactorRef> whole) {
  size_t next = 0;
  PhysicalFactorRef current{};
  bool hasCurrent = false;
  for (const PhysicalFactorRef &factor : prefix) {
    if (!hasCurrent) {
      if (next == whole.size()) {
        return false;
      }
      current = whole[next++];
      hasCurrent = true;
    }
    // The factor must take the most significant digits of what remains of the
    // current whole-side factor.
    if (factor.axisIndex != current.axisIndex ||
        factor.strideWithinAxis * factor.extent !=
            current.strideWithinAxis * current.extent ||
        current.extent % factor.extent != 0) {
      return false;
    }
    current.extent /= factor.extent;
    hasCurrent = current.extent > 1;
  }
  return hasCurrent || next < whole.size();
}

std::string describe(llvm::ArrayRef<PhysicalFactorRef> factors) {
  std::string text;
  llvm::raw_string_ostream os(text);
  os << "[";
  llvm::interleaveComma(factors, os, [&](const PhysicalFactorRef &factor) {
    os << "axis" << factor.axisIndex << "<" << factor.extent << ","
       << factor.strideWithinAxis << ">";
  });
  os << "]";
  return text;
}

bool sameFactors(llvm::ArrayRef<PhysicalFactorRef> a,
                 llvm::ArrayRef<PhysicalFactorRef> b) {
  return llvm::equal(a, b, sameFactor);
}

} // namespace

void ShardyLogicalAxisAnalysis::initShardyBaseline() {
  auto moduleOp = sdy_func->getParentOfType<ModuleOp>();
  if (!moduleOp || !moduleOp->hasAttr(kShardyBaselineAttr)) {
    return;
  }
  FailureOr<PhysicalMeshOp> mesh = findUniquePhysicalMesh(moduleOp);
  if (failed(mesh)) {
    valid = false;
    return;
  }
  shardyBaseline = true;
  baselineModule = moduleOp;
  baselinePhysicalMesh = *mesh;
  for (Attribute axisAttr : mesh->getAxesAttr()) {
    auto axis = cast<PhysicalCommAxisType>(cast<TypeAttr>(axisAttr).getValue());
    physicalAxes.push_back({static_cast<int64_t>(axis.getExtent()),
                            static_cast<int64_t>(axis.getIdStride())});
  }
}

FailureOr<llvm::SmallVector<llvm::SmallVector<PhysicalFactorRef>>>
ShardyLogicalAxisAnalysis::projectShardingDims(
    Operation *diagOp, llvm::StringRef what, sdy::TensorShardingAttr sharding) {
  if (!sharding.getUnreducedAxes().empty()) {
    diagOp->emitError() << "Shardy baseline: " << what
                        << " has unreduced axes, which are not supported";
    valid = false;
    return failure();
  }

  sdy::MeshAttr mesh = sharding.getMesh(diagOp);
  if (!mesh) {
    diagOp->emitError() << "Shardy baseline: " << what
                        << " references a mesh that cannot be resolved";
    valid = false;
    return failure();
  }
  // The sdy meshes use iota device order (checked when the module is stamped),
  // so an axis's device-id stride is the product of the axes minor to it.
  llvm::SmallVector<int64_t> axisDeviceStrides(mesh.getAxes().size());
  int64_t stride = 1;
  for (size_t i = mesh.getAxes().size(); i-- > 0;) {
    axisDeviceStrides[i] = stride;
    stride *= mesh.getAxes()[i].getSize();
  }

  llvm::SmallVector<llvm::SmallVector<PhysicalFactorRef>> dims;
  for (auto [dimIdx, dimSharding] :
       llvm::enumerate(sharding.getDimShardings())) {
    llvm::SmallVector<PhysicalFactorRef> &expected = dims.emplace_back();
    for (sdy::AxisRefAttr ref : dimSharding.getAxes()) {
      int64_t meshIdx = -1;
      for (auto [idx, meshAxis] : llvm::enumerate(mesh.getAxes())) {
        if (meshAxis.getName() == ref.getName()) {
          meshIdx = idx;
        }
      }
      assert(meshIdx >= 0 && "sdy verifier guarantees axis names are in mesh");
      auto projected = projectShardySubAxis(
          mesh.getAxes()[meshIdx].getSize(), axisDeviceStrides[meshIdx],
          ref.getSubAxisPreSize(), ref.getSize(mesh), physicalAxes);
      if (!projected) {
        diagOp->emitError()
            << "Shardy baseline: " << what << " dim " << dimIdx << " axis "
            << ref.toString() << " cannot be expressed on the physical mesh: "
            << llvm::toString(projected.takeError());
        valid = false;
        return failure();
      }
      expected.append(projected->begin(), projected->end());
    }
  }
  return dims;
}

void ShardyLogicalAxisAnalysis::seedTensorDims(
    Operation *diagOp, llvm::StringRef what, Value value,
    sdy::TensorShardingAttr sharding,
    llvm::ArrayRef<llvm::SmallVector<AxisSymbol>> dims) {
  auto type = cast<RankedTensorType>(value.getType());
  assert(sharding.getRank() == type.getRank() &&
         dims.size() == static_cast<size_t>(type.getRank()) &&
         "sharding and analysis dimensions must match the tensor rank");

  auto projected = projectShardingDims(diagOp, what, sharding);
  if (failed(projected)) {
    return;
  }

  for (auto [dimIdx, expected] : llvm::enumerate(*projected)) {
    if (type.isDynamicDim(dimIdx)) {
      diagOp->emitError() << "Shardy baseline: " << what << " dim " << dimIdx
                          << " is dynamic";
      valid = false;
      return;
    }
    int64_t dimSize = type.getDimSize(dimIdx);
    std::string where = (what + " dim " + llvm::Twine(dimIdx)).str();

    int64_t shardProduct = 1;
    for (const PhysicalFactorRef &factor : expected) {
      shardProduct *= factor.extent;
    }
    if (dimSize % shardProduct != 0) {
      diagOp->emitError() << "Shardy baseline: " << where << " has size "
                          << dimSize << " which is not divisible by its "
                          << shardProduct << " shards (padding is unsupported)";
      valid = false;
      return;
    }

    llvm::SmallVector<AxisSymbol> seedSymbols;
    for (const PhysicalFactorRef &factor : expected) {
      AxisSymbol atom = AxisSymbol::create(factor.extent);
      boundAtoms.push_back({atom, factor, diagOp, where});
      seedSymbols.push_back(atom);
    }
    if (dimSize / shardProduct > 1) {
      seedSymbols.push_back(AxisSymbol::create(dimSize / shardProduct));
    }
    if (seedSymbols.empty()) {
      continue;
    }

    uint64_t existingExtent = 1;
    for (AxisSymbol symbol : symbolFactorMerge.resolve(dims[dimIdx])) {
      existingExtent *= symbol.getExtent();
    }
    if (existingExtent != static_cast<uint64_t>(dimSize)) {
      diagOp->emitError() << "Shardy baseline: " << where << " has size "
                          << dimSize << " but its axis symbols span "
                          << existingExtent;
      valid = false;
      return;
    }

    if (seedSymbols.size() > 1) {
      symbolFactorMerge.markOverlapping(seedSymbols);
    }
    symbolFactorMerge.mergeSymbols(dims[dimIdx], seedSymbols);
    dimSeeds.push_back({diagOp, where, dims[dimIdx], seedSymbols, expected});
  }
  rootBindingsDirty = true;
}

const llvm::SmallVector<llvm::SmallVector<PhysicalFactorRef>> *
ShardyLogicalAxisAnalysis::getProjection(Operation *diagOp,
                                         llvm::StringRef what,
                                         sdy::TensorShardingAttr sharding) {
  auto it = projectionCache.find(sharding);
  if (it == projectionCache.end()) {
    auto projected = projectShardingDims(diagOp, what, sharding);
    if (failed(projected)) {
      return nullptr;
    }
    it = projectionCache.try_emplace(sharding, std::move(*projected)).first;
  }
  return &it->second;
}

void ShardyLogicalAxisAnalysis::mergeEdgeRespectingShardy(
    const TensorAxesToPartitionAxes &producerMapping, Value producerValue,
    const TensorAxesToPartitionAxes &consumerMapping, OpOperand &use,
    bool isBoundary) {
  auto mergeAll = [&]() {
    for (auto [prod, cons] : llvm::zip(producerMapping, consumerMapping)) {
      symbolFactorMerge.mergeSymbols(prod, cons);
    }
  };
  if (producerMapping.size() != consumerMapping.size()) {
    return;
  }

  // Across a call, each end is judged by the sharding Shardy gave the value on
  // that side of it: the callee's own argument or returned value, or the
  // caller's value.
  auto calleeBody = [](Operation *call) -> Block & {
    return cast<CallOpInterface>(call).resolveCallable()->getRegion(0).front();
  };
  Value producerEnd = producerValue;
  if (auto result = dyn_cast<OpResult>(producerValue);
      result && isa<CallOpInterface>(result.getOwner())) {
    producerEnd = calleeBody(result.getOwner())
                      .getTerminator()
                      ->getOperand(result.getResultNumber());
  }
  sdy::TensorShardingAttr producerSharding = sdy::getSharding(producerEnd);
  if (!producerSharding) {
    mergeAll();
    return;
  }
  const auto *producerDims =
      getProjection(use.getOwner(), "producer", producerSharding);
  if (!producerDims) {
    return;
  }

  if (isBoundary) {
    Value consumerEnd = use.get();
    if (isa<CallOpInterface>(use.getOwner())) {
      consumerEnd =
          calleeBody(use.getOwner()).getArgument(use.getOperandNumber());
    }
    sdy::TensorShardingAttr consumerSharding = sdy::getSharding(consumerEnd);
    if (!consumerSharding) {
      mergeAll();
      return;
    }
    const auto *consumerDims =
        getProjection(use.getOwner(), "consumer", consumerSharding);
    if (!consumerDims) {
      return;
    }
    for (auto [dimIdx, dims] : llvm::enumerate(llvm::zip(
             producerMapping, consumerMapping, *producerDims, *consumerDims))) {
      auto &[prod, cons, producerFactors, consumerFactors] = dims;
      if (sameFactors(coalesce(producerFactors), coalesce(consumerFactors))) {
        symbolFactorMerge.mergeSymbols(prod, cons);
      }
    }
    return;
  }

  // Bindings reflect every merge so far, which includes the consumer op's own
  // result shardings seeded earlier.
  if (rootBindingsDirty) {
    rebuildRootBindings(/*diagnose=*/false);
  }
  for (auto [dimIdx, dims] : llvm::enumerate(
           llvm::zip(producerMapping, consumerMapping, *producerDims))) {
    auto &[prod, cons, producerFactors] = dims;
    llvm::SmallVector<PhysicalFactorRef> consumerFactors;
    for (AxisSymbol piece : symbolFactorMerge.resolve(cons)) {
      auto it = rootBindings.find(piece);
      if (it == rootBindings.end()) {
        break;
      }
      consumerFactors.push_back(it->second);
    }
    if (!isProperMajorPrefix(coalesce(producerFactors),
                             coalesce(consumerFactors))) {
      symbolFactorMerge.mergeSymbols(prod, cons);
      rootBindingsDirty = true;
    }
  }
}

void ShardyLogicalAxisAnalysis::seedShardyShardings() {
  if (!shardyBaseline) {
    return;
  }
  for (Operation *func : analyzedFuncs) {
    Block &body = func->getRegion(0).front();
    for (BlockArgument arg : body.getArguments()) {
      if (!isa<RankedTensorType>(arg.getType())) {
        continue;
      }
      sdy::TensorShardingAttr sharding = sdy::getSharding(arg);
      auto argIt = argToPartitioningAxes.find(arg);
      if (!sharding || argIt == argToPartitioningAxes.end()) {
        continue;
      }
      seedTensorDims(func, "argument #" + std::to_string(arg.getArgNumber()),
                     arg, sharding, argIt->second);
      if (!valid) {
        return;
      }
    }

    for (Operation &op : body) {
      if (isa<sdy::ShardingConstraintOp>(op)) {
        op.emitError() << "Shardy baseline: a sharding constraint survived "
                          "explicit reshard insertion";
        valid = false;
        return;
      }
      if (auto reshard = dyn_cast<sdy::ReshardOp>(op)) {
        // The input side carries the operand's sharding, the output side the
        // reshard's target.
        auto seedSide = [&](llvm::StringRef side, Value value,
                            const DimToSymbol &symbols) {
          sdy::TensorShardingAttr sharding = sdy::getSharding(value);
          llvm::SmallVector<llvm::SmallVector<AxisSymbol>> dims;
          for (AxisSymbol symbol : symbols) {
            dims.push_back({symbol});
          }
          seedTensorDims(&op, ("reshard " + side).str(), value, sharding, dims);
        };
        seedSide("input", reshard.getInput(), reshardLHSSymbols[&op]);
        if (valid) {
          seedSide("result", reshard.getResult(), reshardRHSSymbols[&op]);
        }
        if (!valid) {
          return;
        }
        continue;
      }
      for (OpResult result : op.getResults()) {
        if (!isa<RankedTensorType>(result.getType())) {
          continue;
        }
        sdy::TensorShardingAttr sharding = sdy::getSharding(result);
        if (!sharding) {
          continue;
        }
        std::optional<TensorAxesToPartitionAxes> dims =
            getTensorPartitionDims(result);
        if (!dims) {
          continue;
        }
        seedTensorDims(&op,
                       "result #" + std::to_string(result.getResultNumber()),
                       result, sharding, *dims);
        if (!valid) {
          return;
        }
      }
    }
  }
}

void ShardyLogicalAxisAnalysis::rebuildRootBindings(bool diagnose) {
  rootBindings.clear();
  llvm::DenseMap<AxisSymbol, const BoundAtom *> owners;
  for (const BoundAtom &atom : boundAtoms) {
    // Pieces are major-first; the minor-most piece keeps the atom's stride.
    llvm::SmallVector<AxisSymbol> pieces =
        symbolFactorMerge.resolve(atom.symbol);
    int64_t stride = atom.factor.strideWithinAxis;
    for (AxisSymbol piece : llvm::reverse(pieces)) {
      PhysicalFactorRef binding{atom.factor.axisIndex,
                                static_cast<int64_t>(piece.getExtent()),
                                stride};
      stride *= piece.getExtent();
      auto [it, inserted] = rootBindings.try_emplace(piece, binding);
      if (inserted) {
        owners[piece] = &atom;
        if (diagnose && symbolFactorMerge.isUnshardable(piece)) {
          atom.diagOp->emitError()
              << "Shardy baseline: " << atom.description << " is sharded over "
              << describe(binding)
              << ", but propagation ties it to a permutation or "
                 "need-replication factor the compiler keeps device-local";
          valid = false;
        }
        continue;
      }
      if (diagnose && !sameFactor(it->second, binding)) {
        atom.diagOp->emitError()
            << "Shardy baseline: " << atom.description << " is sharded over "
            << describe(binding) << ", but propagation ties it to "
            << owners[piece]->description << " which is sharded over "
            << describe(it->second);
        valid = false;
      }
    }
  }
  rootBindingsDirty = false;
}

std::optional<PhysicalFactorRef>
ShardyLogicalAxisAnalysis::getPhysicalBinding(AxisSymbol symbol) {
  if (rootBindingsDirty) {
    rebuildRootBindings(/*diagnose=*/false);
  }
  llvm::SmallVector<AxisSymbol> roots = symbolFactorMerge.resolve(symbol);
  assert(roots.size() == 1 && "materialized symbols must be roots");
  auto it = rootBindings.find(roots.front());
  if (it == rootBindings.end()) {
    return std::nullopt;
  }
  return it->second;
}

void ShardyLogicalAxisAnalysis::validateShardyBaseline() {
  if (!shardyBaseline) {
    return;
  }
  rebuildRootBindings(/*diagnose=*/true);
  if (!valid) {
    return;
  }

  for (const DimSeed &seed : dimSeeds) {
    llvm::SmallVector<AxisSymbol> pieces =
        symbolFactorMerge.resolve(seed.symbols);
    if (pieces != symbolFactorMerge.resolve(seed.original)) {
      seed.diagOp->emitError()
          << "Shardy baseline: " << seed.description
          << " could not be reconciled with the axes propagation assigned to "
             "it (the factorizations conflict)";
      valid = false;
      return;
    }

    llvm::SmallVector<PhysicalFactorRef> actual;
    bool sawUnsharded = false;
    for (AxisSymbol piece : pieces) {
      auto it = rootBindings.find(piece);
      if (it == rootBindings.end()) {
        sawUnsharded = true;
        continue;
      }
      if (sawUnsharded) {
        seed.diagOp->emitError()
            << "Shardy baseline: " << seed.description
            << " has a sharded factor (" << describe(it->second)
            << ") minor to an unsharded one; sharded factors must form the "
               "major prefix of a dimension";
        valid = false;
        return;
      }
      actual.push_back(it->second);
    }
    if (!sameFactors(coalesce(actual), coalesce(seed.expected))) {
      seed.diagOp->emitError()
          << "Shardy baseline: " << seed.description << " is sharded over "
          << describe(coalesce(seed.expected))
          << " by Shardy, but the axes propagated to it from neighboring "
             "values shard it over "
          << describe(coalesce(actual))
          << " (Shardy's shardings conflict across ops)";
      valid = false;
      return;
    }
  }
}

TypedValue<axis::AxisFactorType>
ShardyLogicalAxisAnalysis::createPhysicalFactor(PhysicalFactorRef factor,
                                                OpBuilder &builder,
                                                Location loc) {
  if (physicalAxisValues.empty()) {
    for (auto getAxes : baselineModule.getOps<GetPhysicalMeshAxesOp>()) {
      if (getAxes.getPhysicalMeshAttr().getAttr() ==
          baselinePhysicalMesh.getSymNameAttr()) {
        physicalAxisValues.assign(getAxes.getAxes().begin(),
                                  getAxes.getAxes().end());
        break;
      }
    }
  }
  if (physicalAxisValues.empty()) {
    SmallVector<Type> axisTypes;
    for (Attribute axisAttr : baselinePhysicalMesh.getAxesAttr()) {
      axisTypes.push_back(cast<TypeAttr>(axisAttr).getValue());
    }
    OpBuilder moduleBuilder(baselineModule.getContext());
    moduleBuilder.setInsertionPointToStart(baselineModule.getBody());
    auto getAxes = moduleBuilder.create<GetPhysicalMeshAxesOp>(
        baselinePhysicalMesh.getLoc(), axisTypes,
        FlatSymbolRefAttr::get(baselinePhysicalMesh.getSymNameAttr()));
    physicalAxisValues.assign(getAxes.getAxes().begin(),
                              getAxes.getAxes().end());
  }
  auto factorOp = builder.create<axis::AxisFactorOp>(
      loc, physicalAxisValues[factor.axisIndex], factor.extent,
      factor.strideWithinAxis);
  return cast<TypedValue<axis::AxisFactorType>>(factorOp.getResult());
}

} // namespace mlir::enzyme::distributed
