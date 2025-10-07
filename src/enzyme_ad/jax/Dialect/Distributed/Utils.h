#ifndef ENZYME_AD_JAX_DIALECT_DISTRIBUTED_UTILS_H
#define ENZYME_AD_JAX_DIALECT_DISTRIBUTED_UTILS_H

#include "Dialect.h"
#include "Traits.h"

namespace mlir::enzyme::distributed {

/**
 * Get the enclosing device parallel branch for a given operation, or nullptr
 * if the provided deviceParallelOp is not an ancestor of op.
 */
Region *getEnclosingDeviceParallelBranch(DeviceParallelOp parent,
                                         Operation *op);

/**
 * Get the index of a device parallel branch within its parent operation.
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
 * Returns the counterpart tokens across all branches for the provided token.
 * Each token here corresponds to the same logical token, but passed as a
 * different block argument to each branch. Tokens are ordered in the same order
 * as the branches of the parent DeviceParallelOp. Includes token itself.
 */
llvm::SmallVector<Token> getCorrespondingTokens(Token token);

/**
 * Returns all users of the provided token or its counterpart across all
 * branches, including readers, writers, and any other op that takes the token
 * as an operand.
 */
llvm::SmallVector<mlir::Operation *> getTokenUsers(Token token);

/**
 * Returns true if no other ops ever write to any token written by the
 * provided op.
 */
bool isSoleSender(TokenWriterOpInterface writer);

/**
 * Returns true if no other ops in the provided list send on the same channel.
 */
bool isSoleSender(TokenWriterOpInterface writer, Token token,
                  llvm::ArrayRef<Operation *> others);
} // namespace mlir::enzyme::distributed

#endif // ENZYME_AD_JAX_DIALECT_DISTRIBUTED_UTILS_H