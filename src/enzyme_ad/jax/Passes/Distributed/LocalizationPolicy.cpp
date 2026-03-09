#include "src/enzyme_ad/jax/Passes/Distributed/LocalizationPolicy.h"
#include "src/enzyme_ad/jax/Dialect/Distributed/Dialect.h"
#include "mlir/IR/Value.h"

#include <algorithm>


namespace mlir {
namespace enzyme {
namespace distributed {

NaiveLocalMostPolicy::NaiveLocalMostPolicy(int budget) : budget(budget) {}

/**
 * A factor is less significant than another factor if:
 * - It has a different physical axis, which is less significant (leftmost)
 * - It has the same physical axis but is a less significant factor (leftmost)
 * This function requires both values to be factor axis and not product axis.
 *
 * Returns a < b.
 */
bool compare_factor_significance(PhysicalMeshOp mesh, Value a, Value b) {
  auto a_parent = a.getDefiningOp<AxisFactorOp>();
  auto b_parent = b.getDefiningOp<AxisFactorOp>();
  assert(a_parent && b_parent &&
         "Both values must be defined by AxisFactorOp");

  // If the physical axes are the same, assert that the parents are the same
  // and compare the factor indices
  if (a_parent.getPhysicalAxisAttr() == b_parent.getPhysicalAxisAttr()) {
    assert(
        a_parent == b_parent &&
        "Factors with the same physical axis must be defined by the same op");
    auto a_result = cast<OpResult>(a);
    auto b_result = cast<OpResult>(b);
    return a_result.getResultNumber() < b_result.getResultNumber();
  }

  // Otherwise, compare the physical axes in the mesh
  auto a_axis = a_parent.getPhysicalAxisAttr();
  auto b_axis = b_parent.getPhysicalAxisAttr();
  auto a_axis_index = mesh.getPhysicalAxisPosition(a_axis);
  auto b_axis_index = mesh.getPhysicalAxisPosition(b_axis);
  return a_axis_index < b_axis_index;
}

llvm::SmallVector<TypedValue<LogicalCommAxisType>>
NaiveLocalMostPolicy::suggestLocalization(MeshComputationOp mesh_op,
                                          OpIterator range_begin,
                                          OpIterator range_end) {
  (void)range_begin;
  (void)range_end;

  auto logicalMesh = mesh_op.getMesh().getDefiningOp<LogicalMeshOp>();
  assert(logicalMesh && "mesh operand must be defined by LogicalMeshOp");
  auto physicalMeshOr = logicalMesh.resolvePhysicalMesh();
  assert(succeeded(physicalMeshOr) &&
      "logical mesh must resolve to a physical mesh");
  PhysicalMeshOp physicalMesh = *physicalMeshOr;

  // Use utility functions to decompose the logical mesh into atomic factors,
  // then sort by significance and greedily consume the least significant
  // factors until the budget is met.
  llvm::SmallVector<Value> atomicFactors;
  (void)logicalMesh.resolveToAtomicFactors(atomicFactors);

  std::sort(atomicFactors.begin(), atomicFactors.end(),
            [&](Value a, Value b) {
              return compare_factor_significance(physicalMesh, a, b);
            });

  llvm::SmallVector<TypedValue<LogicalCommAxisType>> suggested;
  int budget_used = 1;
  for (Value factor : atomicFactors) {
    if (budget_used >= budget) {
      break;
    }

    auto factorOp =
        dyn_cast_or_null<LogicalCommAxisOpInterface>(factor.getDefiningOp());
    assert(factorOp && "factor must be defined by LogicalCommAxisOpInterface");
    auto factor_size = factorOp.getAxisSize(factor);
    if (factor_size <= budget / budget_used) {
      budget_used *= factor_size;
      suggested.push_back(cast<TypedValue<LogicalCommAxisType>>(factor));
    }
  }

  return suggested;
}

} // namespace distributed
} // namespace enzyme
} // namespace mlir