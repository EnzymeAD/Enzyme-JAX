#ifndef DISTRIBUTED_LOCALIZATION_TYPES_H
#define DISTRIBUTED_LOCALIZATION_TYPES_H

#include "llvm/ADT/DenseMap.h"
#include "mlir/IR/Value.h"

namespace mlir {
namespace enzyme {
namespace distributed {

/// Sharding modes for tensor placement strategies during localization.
enum class ShardingMode {
  /// Device-specific compute with index-aware pruning - each op assigned to
  /// a specific device, wrong-index clones pruned from other device bodies.
  IndexBased,

  /// Distributed computation across all devices - ops replicated on all
  /// devices, communicate results between them.
  Sharded,

  /// Replicated compute on all devices - all ops execute identically on every
  /// device, no cross-device communication needed.
  Replicated
};

/// Policy for how a tensor operation should be localized to the mesh.
struct TensorBindingChoice {
  /// The localized axis along which this tensor is distributed.
  ::mlir::Value localizedAxis;

  /// The axis within the tensor's shape being distributed (0 = first axis).
  int64_t tensorAxis = 0;

  /// For IndexBased mode: which device index gets this operation.
  int64_t chosenDeviceIndex = 0;

  /// Sharding mode determining how this op is placed across devices.
  ShardingMode shardingMode = ShardingMode::IndexBased;
};

/// Map from tensor SSA values to their localization binding choices.
using TensorBindingMap = ::llvm::DenseMap<::mlir::Value, TensorBindingChoice>;

} // namespace distributed
} // namespace enzyme
} // namespace mlir

#endif
