#ifndef ENZYME_AD_JAX_PASSES_DISTRIBUTED_LOGICALAXISOVERLAP_H
#define ENZYME_AD_JAX_PASSES_DISTRIBUTED_LOGICALAXISOVERLAP_H

#include "src/enzyme_ad/jax/Dialect/Distributed/Dialect.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"

#include <optional>

namespace mlir::enzyme::distributed {

/**
 * For a module already in the Distributed dialect, builds a
 * map of logical axes that are "overlapping": they are assigned
 * to the same operation or collective, and should not be assigned
 * to the same axis of the physical device mesh.
 *
 * Overlapping is a symmetric but not necessarily transitive relation.
 */
class LogicalAxisOverlap {
private:
  using LogicalAxisSet =
      llvm::SmallSetVector<TypedValue<LogicalMeshAxisType>, 4>;

  // Map from a logical axis to the set of logical axes it overlaps with
  llvm::DenseMap<TypedValue<LogicalMeshAxisType>, LogicalAxisSet> overlaps;

  void addOverlap(TypedValue<LogicalMeshAxisType> axis1,
                  TypedValue<LogicalMeshAxisType> axis2) {
    if (axis1 == axis2)
      return;
    overlaps[axis1].insert(axis2);
    overlaps[axis2].insert(axis1);
  }

  // Marks every pair within the group as mutually overlapping.
  void addMutualOverlaps(llvm::ArrayRef<TypedValue<LogicalMeshAxisType>> axes);

public:
  LogicalAxisOverlap(ModuleOp moduleOp);
  std::optional<const llvm::ArrayRef<TypedValue<LogicalMeshAxisType>>>
  getOverlaps(TypedValue<LogicalMeshAxisType> axis) const;
};

} // namespace mlir::enzyme::distributed
#endif // ENZYME_AD_JAX_PASSES_DISTRIBUTED_LOGICALAXISOVERLAP_H