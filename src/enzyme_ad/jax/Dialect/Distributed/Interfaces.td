#ifndef ENZYME_AD_JAX_DIALECT_DISTRIBUTED_INTERFACES
#define ENZYME_AD_JAX_DIALECT_DISTRIBUTED_INTERFACES

include "mlir/IR/OpBase.td"

def DeviceDefTrait : NativeOpTrait<"enzyme::distributed::DeviceDefTrait">;
def ChannelDefTrait : NativeOpTrait<"enzyme::distributed::ChannelDefTrait">;

#endif // ENZYME_AD_JAX_DIALECT_DISTRIBUTED_INTERFACES