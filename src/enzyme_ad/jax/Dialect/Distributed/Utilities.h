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
  OpShardingRuleAndReductionKind(
      ::mlir::sdy::OpShardingRuleAttr rule,
      ::mlir::stablehlo::ReduceOpKind reductionKind,
      std::shared_ptr<::mlir::Region> reductionBody = nullptr)
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

// A kernel is "trivially local" if it has no sharding attributes- aka,
// it has been lowered into its single-device form.
bool isTriviallyLocalKernel(DistributedKernelOp kernelOp);

// A sharding attribute declaring no axis assigned to any dimension
IndexedTensorShardingAttr buildEmptyShardingForType(::mlir::MLIRContext *ctx,
                                                    ::mlir::Type type);

// Builds a standalone module containing one func.func ("kernel") whose
// signature and body mirror kernelOp's body block: argument types from the
// block's own arguments, result types from its DistributedYield operands,
// and each body op cloned in verbatim (attributes included) so per-op
// metadata like distributed.argument_shardings/output_shardings survives
// unchanged. kernelOp itself is left untouched. Two lowering stages need
// exactly this kernel-body-as-a-function view for different reasons --
// LowerKernelsPass hands it to Shardy after layering on sdy sharding
// attributes, while a later executable-lowering stage hands it to an
// external compiler as-is and must NOT expect or add any sharding
// attributes of its own -- so the shape of the module is shared here and
// each caller only adds what's specific to its own destination.
//
// A kernel body need not be isolated from above (e.g. CSE can common up a
// constant used by several sibling kernels and hoist the single copy just
// outside all of them), so any such captured value is pulled in by cloning
// its defining op, recursively, making the returned module self-contained.
// Fails (with a diagnostic already emitted on kernelOp) if a capture has no
// defining op to clone -- an outer block argument -- which should not occur
// for a well-formed kernel body.
//
// On success the caller owns the returned module and is responsible for
// erasing it.
::mlir::FailureOr<::mlir::ModuleOp>
buildKernelBodyModule(DistributedKernelOp kernelOp);

// The inverse of buildKernelBodyModule: splices an external compiler's
// lowering of a kernel's body back into kernelOp. patchedModule is expected
// to follow the same "kernel" ABI buildKernelBodyModule produces -- a
// func.func named "kernel" with the same signature -- but with its body
// rewritten by the external compiler, plus any number of sibling
// func.func's the rewritten body calls into (e.g. ACT's own convention:
// bodyless private declarations carrying a cost attribute on the launch and
// a role attribute on the stage/fetch helpers around it). Concretely:
//   1. kernelOp's region is replaced with "kernel"'s body: the patched
//      function's own block arguments are remapped 1:1 (by position) onto
//      kernelOp's existing block arguments, and its func.return becomes a
//      DistributedYieldOp with the same (remapped) operands.
//   2. Every other top-level function in patchedModule is copied into the
//      module containing kernelOp, as a new top-level sibling.
//
// patchedModule is untouched by this call (its functions are cloned, not
// moved) -- the caller still owns it and is responsible for erasing it.
//
// Since an external compiler names these helper functions the same way for
// every kernel it compiles, splicing more than one kernel into the same
// module collides on those names; each sibling is inserted through the
// containing module's SymbolTable, which auto-renames on conflict, and any
// reference to a renamed sibling within the body just spliced in step 1 is
// rewritten to match. Renaming/rewriting is scoped to kernelOp's own region,
// never module-wide, so an unrelated pre-existing symbol of the same name
// elsewhere in the module is never touched.
//
// patchedModule is a boundary input (an external tool's real output, or a
// literal file fed in for testing) that MLIR's own verifier does not shape-
// check against this ABI, so this validates it and fails cleanly -- with a
// diagnostic already emitted on kernelOp -- if there is no defined "kernel"
// function or its signature does not match kernelOp's body.
::mlir::LogicalResult spliceKernelBodyModule(DistributedKernelOp kernelOp,
                                             ::mlir::ModuleOp patchedModule);

} // namespace mlir::enzyme::distributed

#endif // ENZYME_AD_JAX_DIALECT_DISTRIBUTED_UTILITIES_H
