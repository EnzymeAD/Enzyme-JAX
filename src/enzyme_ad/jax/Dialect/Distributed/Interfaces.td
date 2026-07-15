#ifndef ENZYME_AD_JAX_DIALECT_DISTRIBUTED_INTERFACES
#define ENZYME_AD_JAX_DIALECT_DISTRIBUTED_INTERFACES

include "mlir/IR/OpBase.td"

// PhysicalCommAxisInterface: require physical axes to provide size and
// cost modeling information.
def CommAxisOpInterface :
		OpInterface<"CommAxisOpInterface"> {
	let description = [{
		Interface for ops that define a communication axis. Contains the shared
        components of both physical and logical axes interfaces.
	}];

	let methods = [

        // Since an op may define multiple axes (see factor), we need to be able to query the size of each axis separately,
        // so this takes in the mlir Value corresponding to the axis in question.
		InterfaceMethod<
			[{Returns the extent/size of the axis in number of devices. Takes in the 
            corresponding axis in question, since an op may define multiple axes (e.g. factor).}],
            "int64_t", "getAxisSize", (ins "::mlir::Value":$axis)>,
	];
}

def PhysicalCommAxisOpInterface :
        OpInterface<"PhysicalCommAxisOpInterface", [CommAxisOpInterface]> {
    let description = [{
        Interface for ops that define a physical communication axis. Requires the methods of CommAxisOpInterface.
    }];

    let methods = [
       // TODO: determine queries as needed
    ];

    // Assert that any op implementing a physical axis also implements the base CommAxisOpInterface.
    let verify = [{
        if (!::llvm::isa<CommAxisOpInterface>($_op))
            return $_op->emitOpError()
                            << "must also implement CommAxisOpInterface";
        return ::mlir::success();
    }];
}

def LogicalCommAxisOpInterface :
        OpInterface<"LogicalCommAxisOpInterface", [CommAxisOpInterface]> {
    let description = [{
        Interface for ops that define a logical communication axis. Requires
        the methods of CommAxisOpInterface.
    }];

    let methods = [
       // TODO: determine queries as needed
    ];

    // Assert that any op implementing a logical axis also implements the base CommAxisOpInterface.
    let verify = [{
        if (!::llvm::isa<CommAxisOpInterface>($_op))
            return $_op->emitOpError()
                            << "must also implement CommAxisOpInterface";
        return ::mlir::success();
    }];
}


#endif // ENZYME_AD_JAX_DIALECT_DISTRIBUTED_INTERFACES