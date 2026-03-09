#ifndef ENZYME_AD_JAX_PASSES_DISTRIBUTED_LOCALIZATIONPOLICY_H
#define ENZYME_AD_JAX_PASSES_DISTRIBUTED_LOCALIZATIONPOLICY_H

#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Block.h"

#include "src/enzyme_ad/jax/Dialect/Distributed/Dialect.h"

namespace mlir {
namespace enzyme {
namespace distributed {

using OpIterator = mlir::Block::iterator;

/**
 * A localization policy takes mesh region, a range within the region to
 * localize (for instance, suggested by the partitioner), and suggests a submesh
 * to localize to. Currently, this class returns a single submesh, but we may
 * change this to returning multiple candidate submeshes with scores in the
 * future.
 */
class LocalizationPolicy {
public:
  virtual ~LocalizationPolicy() = default;

  virtual llvm::SmallVector<TypedValue<LogicalCommAxisType>>
  suggestLocalization(MeshComputationOp mesh_op, OpIterator range_begin,
                      OpIterator range_end) = 0;
};


/**
 * Greedily consumes the least significant factors of the logical mesh axes until the budget is met.
 * Does not introduce subfactorization, so results are fragile depending on the input factorization.
 * May skip factors too large for the remaining budget. 
 */
class NaiveLocalMostPolicy : public LocalizationPolicy {

  int budget;

public:
  NaiveLocalMostPolicy(int budget);
  ~NaiveLocalMostPolicy() override = default;

  llvm::SmallVector<TypedValue<LogicalCommAxisType>>
  suggestLocalization(MeshComputationOp mesh_op, OpIterator range_begin,
                      OpIterator range_end) override;
};

} // namespace distributed
} // namespace enzyme
} // namespace mlir

#endif // ENZYME_AD_JAX_PASSES_DISTRIBUTED_LOCALIZATIONPOLICY_H