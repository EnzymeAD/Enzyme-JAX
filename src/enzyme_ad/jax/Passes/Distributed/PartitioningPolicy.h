#ifndef ENZYME_AD_JAX_PASSES_DISTRIBUTED_PARTITIONINGPOLICY_H
#define ENZYME_AD_JAX_PASSES_DISTRIBUTED_PARTITIONINGPOLICY_H

#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Block.h"

#include "src/enzyme_ad/jax/Dialect/Distributed/Dialect.h"

namespace mlir {
namespace enzyme {
namespace distributed {

using OpIterator = mlir::Block::iterator;

struct OpRange {
	OpIterator begin;
	OpIterator end;
};

class PartitioningPolicy {
public:
	virtual ~PartitioningPolicy() = default;

	virtual llvm::SmallVector<OpRange>
	suggestSpecializationRanges(MeshComputationOp meshOp) = 0;
};

class TrivialWholeOpPartitioningPolicy : public PartitioningPolicy {
public:
	llvm::SmallVector<OpRange>
	suggestSpecializationRanges(MeshComputationOp meshOp) override;
	~TrivialWholeOpPartitioningPolicy() override = default;
};

} // namespace distributed
} // namespace enzyme
} // namespace mlir

#endif // ENZYME_AD_JAX_PASSES_DISTRIBUTED_PARTITIONINGPOLICY_H
