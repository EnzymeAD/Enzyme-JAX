/**
 * Contains the includes for comm ops and interfaces.
 *
 * These includes are difficult to separate in this case due to template
 * dependencies closely interlink the definitions of the SplitMemberOp trait,
 * MessageOp interface, and CommSplit ops, so they are handled in the same file.
 */
#ifndef ENZYME_AD_JAX_DIALECTS_COMM_COMMOPS_H
#define ENZYME_AD_JAX_DIALECTS_COMM_COMMOPS_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"


#include "src/enzyme_ad/jax/Dialect/Comm/CommDialect.h"
#include "src/enzyme_ad/jax/Dialect/Comm/CommTypes.h"

namespace mlir::comm {
template <typename ConcreteType>
class SplitMemberOp : public OpTrait::TraitBase<ConcreteType, SplitMemberOp> {
public:
  static LogicalResult verifyTrait(Operation *op);
};
} // namespace mlir::comm

#include "src/enzyme_ad/jax/Dialect/Comm/CommInterfaces.h.inc"

#define GET_OP_CLASSES
#include "src/enzyme_ad/jax/Dialect/Comm/CommOps.h.inc"

template <typename ConcreteType>
mlir::LogicalResult
mlir::comm::SplitMemberOp<ConcreteType>::verifyTrait(Operation *op) {
  if (!isa<CommSplit>(op->getParentOp())) {
    return op->emitOpError("must be located as immediate child of split op");
  }
  return success();
}

#endif