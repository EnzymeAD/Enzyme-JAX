#ifndef ENZYME_AD_JAX_DIALECTS_COMM_COMM_H
#define ENZYME_AD_JAX_DIALECTS_COMM_COMM_H

#include "src/enzyme_ad/jax/Dialect/Comm/CommDialect.h"
#include "src/enzyme_ad/jax/Dialect/Comm/CommOps.h"
#include "src/enzyme_ad/jax/Dialect/Comm/CommTypes.h"

// Utility functions

namespace mlir::comm {

/**
 * Returns the device set of a given op. Should only be called on an op
 * located within a branch.
 */
llvm::ArrayRef<int32_t> getOpDevices(mlir::Operation &op);

}

#endif