#ifndef ENZYME_AD_JAX_DIALECT_DISTRIBUTED_UTILS_H
#define ENZYME_AD_JAX_DIALECT_DISTRIBUTED_UTILS_H

#include "Dialect.h"

namespace mlir::enzyme::distributed {

/** Get the enclosing device parallel branch for a given operation, or nullptr
 * if the provided deviceParallelOp is not an ancestor of op.
 */
Region *getEnclosingDeviceParallelBranch(DeviceParallelOp parent,
                                         Operation *op);

/** Get the index of a device parallel branch within its parent operation.
 * Parent op must be the direct parent of the branch region.
 */
int getDeviceParallelBranchIndex(DeviceParallelOp parent, Region *branch);

/**
 * Returns the defining op of the enclosing device of a given computational op
 * (e.g. not the parent of a device defintion op). Returns nullptr if no such
 * device can be found (not inside a device parallel region).
 */
mlir::Operation *getExecutingDevice(mlir::Operation *op);

/**
 * Returns all block arguments in the same device parallel region corresponding
 * to the provided token, including the provided token itself. Will be provided
 * in the same order as the branch assignments of the parent device parallel op.
 */
llvm::SmallVector<mlir::BlockArgument>
getCorrespondingTokens(mlir::BlockArgument token);
llvm::SmallVector<mlir::Operation *> getTokenUsers(mlir::BlockArgument token);
} // namespace mlir::enzyme::distributed

#endif // ENZYME_AD_JAX_DIALECT_DISTRIBUTED_UTILS_H