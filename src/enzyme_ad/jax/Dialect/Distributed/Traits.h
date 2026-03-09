#ifndef ENZYME_AD_JAX_DIALECT_DISTRIBUTED_TRAITS_H
#define ENZYME_AD_JAX_DIALECT_DISTRIBUTED_TRAITS_H

#include "mlir/IR/OpDefinition.h"

namespace mlir::OpTrait::enzyme::distributed {


template <typename ConcreteType>
class MetadataTrait : public OpTrait::TraitBase<ConcreteType, MetadataTrait> {
};
} // namespace mlir::OpTrait::enzyme::distributed

#endif // ENZYME_AD_JAX_DIALECT_DISTRIBUTED_TRAITS_H