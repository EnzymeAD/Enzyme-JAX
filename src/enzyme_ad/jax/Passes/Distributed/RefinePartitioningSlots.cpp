#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h"

#include "mlir/IR/BuiltinOps.h"

#include "src/enzyme_ad/jax/Dialect/Axis/Utilities.h"
#include "src/enzyme_ad/jax/Dialect/Distributed/Utilities.h"

namespace mlir::enzyme::distributed {

#define GEN_PASS_DEF_REFINEPARTITIONINGSLOTSPASS
#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h.inc"

namespace {

bool isShardedFactor(TypedValue<axis::AxisFactorType> factor) {
  auto provenance = axis::getFactorProvenanceAxis(factor);
  assert(succeeded(provenance) && "factor without a provenance axis");
  return isa<LogicalMeshAxisType, PhysicalCommAxisType>(
      (*provenance).getType());
}

// Rewrites every slot index in `attr` to the list of slots it was split into.
IndexedTensorShardingAttr
remapSharding(IndexedTensorShardingAttr attr,
              ArrayRef<SmallVector<int64_t>> newIndices) {
  auto remap = [&](DenseI64ArrayAttr indices) {
    SmallVector<int64_t> result;
    for (int64_t index : indices.asArrayRef())
      llvm::append_range(result, newIndices[index]);
    return DenseI64ArrayAttr::get(indices.getContext(), result);
  };
  SmallVector<DenseI64ArrayAttr> dims;
  for (DenseI64ArrayAttr dim : attr.getDimPartitioningAxes())
    dims.push_back(remap(dim));
  return IndexedTensorShardingAttr::get(attr.getContext(), dims,
                                        remap(attr.getUnreducedAxes()));
}

IndexedTensorShardingPerValueAttr
remapShardings(IndexedTensorShardingPerValueAttr attr,
               ArrayRef<SmallVector<int64_t>> newIndices) {
  SmallVector<IndexedTensorShardingAttr> shardings;
  for (IndexedTensorShardingAttr sharding : attr.getShardings())
    shardings.push_back(remapSharding(sharding, newIndices));
  return IndexedTensorShardingPerValueAttr::get(attr.getContext(), shardings);
}

// Splits every slot of `kernelOp` whose factors mix sharded and local kinds
// into one slot per maximal same-kind run, and rewrites all references.
void refineKernel(DistributedKernelOp kernelOp) {
  OpBuilder builder(kernelOp.getContext());
  builder.setInsertionPoint(kernelOp);
  axis::ModuleScopeGuard moduleScope(builder);

  SmallVector<Value> newSlots;
  SmallVector<SmallVector<int64_t>> newIndices;
  bool changed = false;
  for (Value slot : kernelOp.getPartitioningAxes()) {
    auto factors = axis::getProductProvenanceFactors(
        cast<TypedValue<axis::FactorGroupType>>(slot));
    assert(succeeded(factors) && "slot is not an axis.product");

    SmallVector<SmallVector<TypedValue<axis::AxisFactorType>>> runs;
    for (auto factor : *factors) {
      if (runs.empty() ||
          isShardedFactor(runs.back().front()) != isShardedFactor(factor))
        runs.emplace_back();
      runs.back().push_back(factor);
    }

    auto &indices = newIndices.emplace_back();
    if (runs.size() <= 1) {
      indices.push_back(newSlots.size());
      newSlots.push_back(slot);
      continue;
    }
    changed = true;
    for (auto &run : runs) {
      indices.push_back(newSlots.size());
      newSlots.push_back(
          axis::viewFactorsAsProduct(run, builder, kernelOp.getLoc()));
    }
  }
  if (!changed)
    return;

  kernelOp.getPartitioningAxesMutable().assign(newSlots);
  kernelOp.setArgumentShardingsAttr(
      remapShardings(kernelOp.getArgumentShardings(), newIndices));
  kernelOp.setOutputShardingsAttr(
      remapShardings(kernelOp.getOutputShardings(), newIndices));
  kernelOp.getBody().walk([&](Operation *op) {
    for (StringRef name :
         {"distributed.argument_shardings", "distributed.output_shardings"}) {
      if (auto attr =
              op->getAttrOfType<IndexedTensorShardingPerValueAttr>(name))
        op->setAttr(name, remapShardings(attr, newIndices));
    }
  });
}

struct RefinePartitioningSlotsPass
    : public impl::RefinePartitioningSlotsPassBase<
          RefinePartitioningSlotsPass> {
  using RefinePartitioningSlotsPassBase::RefinePartitioningSlotsPassBase;

  void runOnOperation() override {
    getOperation().walk(
        [](DistributedKernelOp kernelOp) { refineKernel(kernelOp); });
  }
};

} // namespace

} // namespace mlir::enzyme::distributed
