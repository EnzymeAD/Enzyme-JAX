#ifndef ENZYME_AD_JAX_DIALECT_DISTRIBUTED_UTILITIES_H
#define ENZYME_AD_JAX_DIALECT_DISTRIBUTED_UTILITIES_H

#include "Dialect.h"
#include "src/enzyme_ad/jax/Dialect/Axis/Utilities.h"
#include "src/enzyme_ad/jax/Utils.h"

#include <memory>

namespace mlir::enzyme::distributed {

using ::mlir::enzyme::axis::castTypedValue;
using ::mlir::enzyme::axis::castTypedValueList;
using ::mlir::enzyme::axis::TypedValueArrayRef;

template <typename T>
::llvm::SmallVector<T>
flattenNested(const ::llvm::SmallVector<::llvm::SmallVector<T>> &nested) {
  ::llvm::SmallVector<T> flat;
  for (const auto &inner : nested) {
    flat.append(inner.begin(), inner.end());
  }
  return flat;
}

// Walks parent operations and checks each symbol table scope for a flat symbol.
::mlir::Operation *
lookupSymbolInEnclosingScopes(::mlir::Operation *from,
                              ::mlir::FlatSymbolRefAttr symRef);

// Finds the unique distributed physical mesh in the module.
::mlir::FailureOr<::mlir::enzyme::distributed::PhysicalMeshOp>
findUniquePhysicalMesh(::mlir::ModuleOp moduleOp);

template <typename OpTy>
::mlir::FailureOr<OpTy> resolveSymbolOpFromAttr(::mlir::Operation *from,
                                                ::mlir::Attribute opAttr) {
  auto symRef = ::mlir::dyn_cast_or_null<::mlir::FlatSymbolRefAttr>(opAttr);
  if (!symRef) {
    return ::mlir::failure();
  }

  if (auto *op = lookupSymbolInEnclosingScopes(from, symRef)) {
    if (auto typedOp = llvm::dyn_cast<OpTy>(op)) {
      return typedOp;
    }
    return ::mlir::failure();
  }

  return ::mlir::failure();
}

// Creates a new range with all replication axes removed from the input range.
::llvm::SmallVector<TypedValue<::mlir::enzyme::axis::AxisFactorType>>
filterOutReplicationFactors(
    TypedValueArrayRef<::mlir::enzyme::axis::AxisFactorType> factors);

// Try the normal Shardy sharding rule registry first, and fall back to a
// synthetic rule for ops that do not have an explicit sharding rule.
struct OpShardingRuleAndReductionKind {
  OpShardingRuleAndReductionKind() = default;
  OpShardingRuleAndReductionKind(::mlir::sdy::OpShardingRuleAttr rule,
                                 ::mlir::stablehlo::ReduceOpKind reductionKind,
                                 std::shared_ptr<::mlir::Region> reductionBody =
                                     nullptr)
      : rule(rule), reductionKind(reductionKind),
        reductionBody(std::move(reductionBody)) {}

  ::mlir::sdy::OpShardingRuleAttr rule;
  ::mlir::stablehlo::ReduceOpKind reductionKind =
      ::mlir::stablehlo::ReduceOpKind::Add;

  ::mlir::Region *getReductionBody(::mlir::Type elementType) const;

private:
  mutable std::shared_ptr<::mlir::Region> reductionBody;
};

OpShardingRuleAndReductionKind
getOrSynthesizeOpShardingRule(::mlir::Operation *op);

struct CollectiveAndAwait {
  ::mlir::enzyme::distributed::DistributedCollectiveOp collective;
  ::mlir::enzyme::distributed::DistributedAwait await;
};

CollectiveAndAwait
createCollectiveAndAwait(::mlir::OpBuilder &builder, ::mlir::Location loc,
                         ::mlir::Value inputObject, ::mlir::Value inputMesh,
                         ::mlir::Value outputMesh,
                         ::mlir::ValueRange reductionGroups,
                         ::mlir::Value mapping, ::mlir::Type outputType);

} // namespace mlir::enzyme::distributed

#endif // ENZYME_AD_JAX_DIALECT_DISTRIBUTED_UTILITIES_H
