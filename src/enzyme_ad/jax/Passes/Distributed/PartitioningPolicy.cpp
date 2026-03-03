#include "src/enzyme_ad/jax/Passes/Distributed/PartitioningPolicy.h"

namespace mlir {
namespace enzyme {
namespace distributed {

llvm::SmallVector<OpRange>
TrivialWholeOpPartitioningPolicy::suggestSpecializationRanges(
    MeshComputationOp meshOp) {
  llvm::SmallVector<OpRange> ranges;
  Block &body = meshOp.getBody().front();
  ranges.push_back(OpRange{body.begin(), body.end()});
  return ranges;
}

} // namespace distributed
} // namespace enzyme
} // namespace mlir
