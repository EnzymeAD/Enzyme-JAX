#ifndef ENZYME_AD_JAX_DIALECT_AXIS_TRAITS_H
#define ENZYME_AD_JAX_DIALECT_AXIS_TRAITS_H

#include "mlir/IR/OpDefinition.h"

namespace mlir::OpTrait::enzyme::axis {

template <typename ConcreteType>
class MetadataTrait : public OpTrait::TraitBase<ConcreteType, MetadataTrait> {};

} // namespace mlir::OpTrait::enzyme::axis

#endif // ENZYME_AD_JAX_DIALECT_AXIS_TRAITS_H
