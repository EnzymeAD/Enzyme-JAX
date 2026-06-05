#ifndef ENZYMEXLA_COMM_DIALECT_H
#define ENZYMEXLA_COMM_DIALECT_H

#include "mlir/IR/Dialect.h"

#include "src/enzyme_ad/jax/Dialect/Comm/CommDialect.h.inc"

#include "src/enzyme_ad/jax/Dialect/Comm/MPIAttrEnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "src/enzyme_ad/jax/Dialect/Comm/MPIAttrDefs.h.inc"

#define GET_TYPEDEF_CLASSES
#include "src/enzyme_ad/jax/Dialect/Comm/Types.h.inc"

#endif // ENZYMEXLA_COMM_DIALECT_H
