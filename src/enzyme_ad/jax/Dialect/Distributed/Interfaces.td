#ifndef ENZYME_AD_JAX_DIALECT_DISTRIBUTED_INTERFACES
#define ENZYME_AD_JAX_DIALECT_DISTRIBUTED_INTERFACES

include "mlir/IR/OpBase.td"

def AxisTypeInterface : TypeInterface<"AxisTypeInterface"> {
    let cppNamespace = "::mlir::enzyme::distributed";
    let description = [{
        Interface for axis types with a statically encoded extent.
    }];

    let methods = [
        InterfaceMethod<
            [{Returns the compile-time extent of this axis.}],
            "unsigned", "extent", (ins), "",
            [{
                return $_type.getExtent();
            }]>,
        InterfaceMethod<
            [{Returns whether two axis SSA values refer to the same underlying axis under this axis type's semantics. 
              Axis are allowed to either be fully equivalent or disjoint: canonical axes may not partially alias.}],
            "bool", "aliases",
            (ins "::mlir::Value":$selfAxis, "::mlir::Value":$otherAxis)>
    ];
}

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

// Interfaces for timing analysis
def LaneOpInterface :
        OpInterface<"LaneOpInterface"> {
    let methods = [
        InterfaceMethod<
            [{Returns the regions corresponding to the different lanes of this operation}],
            "llvm::SmallVector<mlir::Region *>", "getLanes", (ins)
        >
    ];
}

def HandleDeclarationOpInterface :
        OpInterface<"HandleDeclarationOpInterface"> {
        let methods = [
            InterfaceMethod<
                [{Returns the SSA value representing an event handle}],
                "mlir::Value", "getHandle", (ins)
            >
        ];
}

def EventOpInterface :
        OpInterface<"EventOpInterface"> {
    let description = [{
        An op representing part or all of an event. Not all ops in a lane
        need to be events, but any op spanning multiple lanes must be. 
        Events are ordered by a happens-before relationship, determined by:
        - Explicit happensAfter(): a list of handles whose events must complete before this event can start.
        - Implicit happens after: No interface needed, this is determined by the order of operations within a
          basic block and by SSA dependencies.
                - simultaneousWith(): a handle that this event happens simultaneously with. Multiple events
                    (including those on the same lane) may be simultaneous with the same handle, forming an
                    equivalence relation. From a happens-before view, these events are merged into the same
                    event and share predecessors and successors. Simultaneous events must not have a happens-before
                    relationship between each other, with one exception: concurrent ops in the same lane whose only
                    dependency is from implicit op order, with no intervening non-simultaneous ops.
        - concurrentWith(Op other): A "must be concurrent" check allowing an underapproximation of 
                    concurrency. Currently this is only used when checking the legality of simultaneousWith().
    }];
    let methods = [
        InterfaceMethod<
            [{Returns the SSA value handles whose events must complete before this event can start}],
            "::llvm::SmallVector<::mlir::Value>", "happensAfter", (ins)
        >,
        InterfaceMethod<
            [{Returns the SSA value handles whose events must happen after this event.}],
            "::llvm::SmallVector<::mlir::Value>", "happensBefore", (ins)
        >,
        InterfaceMethod<
            [{Returns the SSA value handles whose events this event happens simultaneously with}],
            "::llvm::SmallVector<::mlir::Value>", "simultaneousWith", (ins)
        >,
        InterfaceMethod<
            [{An underapproximating relation allowing concurrency between two ops. }],
            "bool", "concurrentWith", (ins "mlir::Operation *":$other)
        >
    ];
}


#endif // ENZYME_AD_JAX_DIALECT_DISTRIBUTED_INTERFACES