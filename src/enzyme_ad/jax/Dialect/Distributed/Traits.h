#ifndef ENZYME_AD_JAX_DIALECT_DISTRIBUTED_TRAITS_H
#define ENZYME_AD_JAX_DIALECT_DISTRIBUTED_TRAITS_H

#include "mlir/IR/OpDefinition.h"

namespace mlir::OpTrait::enzyme::distributed {
template <typename ConcreteType>
class DeviceDefTrait : public OpTrait::TraitBase<ConcreteType, DeviceDefTrait> {
};

template <typename ConcreteType>
class ChannelDefTrait
    : public OpTrait::TraitBase<ConcreteType, DeviceDefTrait> {};
} // namespace mlir::OpTrait::enzyme::distributed

#endif // ENZYME_AD_JAX_DIALECT_DISTRIBUTED_TRAITS_H