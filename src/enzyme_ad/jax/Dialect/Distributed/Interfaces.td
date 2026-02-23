#ifndef ENZYME_AD_JAX_DIALECT_DISTRIBUTED_INTERFACES
#define ENZYME_AD_JAX_DIALECT_DISTRIBUTED_INTERFACES

include "mlir/IR/OpBase.td"

def PhysicalCommAxisOpInterface :
        OpInterface<"PhysicalCommAxisOpInterface"> {
    let description = [{
        Interface for ops that define a physical communication axis.
    }];

    let methods = [
        InterfaceMethod<
            [{Returns the extent/size of this physical axis in number of devices.}],
            "int64_t", "getPhysicalAxisSize", (ins)>
    ];
}

def LogicalCommAxisOpInterface :
        OpInterface<"LogicalCommAxisOpInterface"> {
    let description = [{
        Interface for ops that define logical communication axes. Since one
        op may define multiple logical axes, methods are keyed by the queried
        SSA axis value.
    }];

    let methods = [
        InterfaceMethod<
            [{Returns the extent/size of the queried logical axis in number of devices.}],
            "int64_t", "getAxisSize", (ins "::mlir::Value":$axis)>
    ];
}


#endif // ENZYME_AD_JAX_DIALECT_DISTRIBUTED_INTERFACES