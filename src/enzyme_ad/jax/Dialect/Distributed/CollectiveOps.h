#ifndef ENZYME_AD_JAX_DIALECT_DISTRIBUTED_COLLECTIVE_OPS_H
#define ENZYME_AD_JAX_DIALECT_DISTRIBUTED_COLLECTIVE_OPS_H

#include "Dialect.h"
#include "Utilities.h"

namespace mlir::enzyme::distributed {

// Shared with the axis dialect: parses/prints a type-annotated variadic as
// "(%arg1 : type1, ..., %argN : typeN)" or "()".
using ::mlir::enzyme::axis::parseVariadicWithTypes;
using ::mlir::enzyme::axis::printVariadicWithTypes;

} // namespace mlir::enzyme::distributed

#endif // ENZYME_AD_JAX_DIALECT_DISTRIBUTED_COLLECTIVE_OPS_H