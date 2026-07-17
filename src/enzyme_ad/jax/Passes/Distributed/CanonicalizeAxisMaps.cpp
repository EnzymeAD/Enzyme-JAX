#include "src/enzyme_ad/jax/Dialect/Axis/Utilities.h"
#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h"

#include "mlir/IR/Builders.h"

namespace mlir::enzyme::distributed {

#define GEN_PASS_DEF_CANONICALIZEAXISMAPSPASS
#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h.inc"

namespace {

struct CanonicalizeAxisMapsPass
    : public impl::CanonicalizeAxisMapsPassBase<CanonicalizeAxisMapsPass> {
  using CanonicalizeAxisMapsPassBase::CanonicalizeAxisMapsPassBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    OpBuilder builder(&getContext());

    SmallVector<axis::AxisMapOp> mapOps;
    module.walk([&](axis::AxisMapOp mapOp) { mapOps.push_back(mapOp); });

    for (axis::AxisMapOp mapOp : mapOps) {
      auto lhsFactors = axis::castTypedValueList<axis::FactorGroupType>(
          mapOp.getMappingLhs(), "FactorGroupType");
      auto rhsFactors = axis::castTypedValueList<axis::FactorGroupType>(
          mapOp.getMappingRhs(), "FactorGroupType");

      SmallVector<TypedValue<axis::FactorGroupType>> lhsOut;
      SmallVector<TypedValue<axis::FactorGroupType>> rhsOut;

      builder.setInsertionPoint(mapOp);
      axis::split_divisible(lhsFactors, rhsFactors, lhsOut, rhsOut, builder);

      Block &targetBlock = *mapOp->getBlock();
      auto materializeGroup = [&](TypedValue<axis::FactorGroupType> group) {
        Operation *groupOp = group.getDefiningOp();
        if (!groupOp) {
          return;
        }
        auto maybeTemporary = cast<axis::MaybeTemporaryInterface>(groupOp);
        maybeTemporary.materialize(targetBlock);
      };

      for (TypedValue<axis::FactorGroupType> group : lhsOut) {
        materializeGroup(group);
      }
      for (TypedValue<axis::FactorGroupType> group : rhsOut) {
        materializeGroup(group);
      }

      SmallVector<Value> lhsValues;
      lhsValues.reserve(lhsOut.size());
      for (TypedValue<axis::FactorGroupType> value : lhsOut) {
        lhsValues.push_back(value);
      }

      SmallVector<Value> rhsValues;
      rhsValues.reserve(rhsOut.size());
      for (TypedValue<axis::FactorGroupType> value : rhsOut) {
        rhsValues.push_back(value);
      }

      builder.setInsertionPoint(mapOp);
      auto newMap = builder.create<axis::AxisMapOp>(
          mapOp.getLoc(), ValueRange(lhsValues), ValueRange(rhsValues));
      mapOp.getMap().replaceAllUsesWith(newMap.getMap());
      mapOp.erase();
    }
  }
};

} // namespace

} // namespace mlir::enzyme::distributed
