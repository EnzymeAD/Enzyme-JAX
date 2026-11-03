#ifndef ENZYME_AD_JAX_DIALECT_JULIA_DIALECT_H
#define ENZYME_AD_JAX_DIALECT_JULIA_DIALECT_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Types.h"

#include "src/enzyme_ad/jax/Dialect/Julia/JuliaDialect.h.inc"

#include "src/enzyme_ad/jax/Dialect/Julia/JuliaAttrEnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "src/enzyme_ad/jax/Dialect/Julia/JuliaAttrDefs.h.inc"
#undef GET_ATTRDEF_CLASSES

#define GET_TYPEDEF_CLASSES
#include "src/enzyme_ad/jax/Dialect/Julia/JuliaTypes.h.inc"
#undef GET_TYPEDEF_CLASSES

namespace mlir::enzymexla::julia {

inline bool isJuliaBufferType(Type type) {
  return llvm::isa<ArrayType, ViewType>(type);
}

} // namespace mlir::enzymexla::julia

#endif // ENZYME_AD_JAX_DIALECT_JULIA_DIALECT_H
