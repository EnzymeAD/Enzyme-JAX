#ifndef ENZYME_AD_JAX_DIALECTS_COMM_COMMDIALECT_H
#define ENZYME_AD_JAX_DIALECTS_COMM_COMMDIALECT_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Support/TypeID.h"
#include "mlir/include/mlir/IR/DialectImplementation.h"
#include "src/enzyme_ad/jax/Dialect/Comm/CommDialect.h.inc"

#include "mlir/include/mlir/IR/BuiltinOps.h"
#include "mlir/include/mlir/IR/BuiltinTypes.h"
#include "mlir/include/mlir/IR/Dialect.h"

namespace mlir::comm {
template <typename ConcreteType>
class SplitMemberOp : public OpTrait::TraitBase<ConcreteType, SplitMemberOp> {
public:
  static LogicalResult verifyTrait(Operation *op);
};
} // namespace mlir::comm

#define GET_TYPEDEF_CLASSES
#include "src/enzyme_ad/jax/Dialect/Comm/CommTypes.h.inc"

#include "src/enzyme_ad/jax/Dialect/Comm/CommInterfaces.h.inc"

#define GET_ATTRDEF_CLASSES
#include "src/enzyme_ad/jax/Dialect/Comm/CommAttrs.h.inc"

#define GET_OP_CLASSES
#include "src/enzyme_ad/jax/Dialect/Comm/CommOps.h.inc"

template <typename ConcreteType>
mlir::LogicalResult mlir::comm::SplitMemberOp<ConcreteType>::verifyTrait(Operation *op) {
  if (!isa<CommSplit>(op->getParentOp())) {
    return op->emitOpError("must be located as immediate child of split op");
  }
  return success();
}

#endif