#ifndef ENZYME_AD_JAX_DIALECT_AXIS_TRAITS_H
#define ENZYME_AD_JAX_DIALECT_AXIS_TRAITS_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OpDefinition.h"

namespace mlir::OpTrait::enzyme::axis {

// Axis-algebra ops are pure compile-time metadata derived from module-level
// provenance (a physical mesh, a static shape), independent of any specific
// function invocation, so every one belongs directly in the module body.
template <typename ConcreteType>
class MetadataTrait : public OpTrait::TraitBase<ConcreteType, MetadataTrait> {
public:
  static LogicalResult verifyTrait(Operation *op) {
    if (!isa<ModuleOp>(op->getParentOp())) {
      return op->emitOpError()
             << "is axis-algebra metadata and must be declared directly in "
                "a module body";
    }
    return success();
  }
};

} // namespace mlir::OpTrait::enzyme::axis

#endif // ENZYME_AD_JAX_DIALECT_AXIS_TRAITS_H
