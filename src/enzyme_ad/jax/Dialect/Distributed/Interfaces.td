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
            "int64_t", "getAxisSize", (ins "::mlir::enzyme::distributed::TypedOpResult<::mlir::enzyme::distributed::LogicalCommAxisType>":$axis)>,
        InterfaceMethod<
            [{Decomposes the queried logical axis into atomic factors and appends them to `atomicFactors`.}],
            "void", "resolveToAtomicFactors",
            (ins "::mlir::enzyme::distributed::TypedOpResult<::mlir::enzyme::distributed::LogicalCommAxisType>":$axis,
                 "::llvm::SmallVectorImpl<::mlir::enzyme::distributed::TypedOpResult<::mlir::enzyme::distributed::LogicalCommAxisType>> &":$atomicFactors)>
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

def HandleDeclarationOp :
        OpInterface<"HandleDeclarationOp"> {
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
          (including those on the same lane) may be simultanous with the same handle, forming an
          equivalence relation. From a happens-before view, these events are merged into the same
          event and share predecessors and successors. Simulatenous events must not have a happens-before
          relationship between each other, with one exception: concurrent ops in the same lane whose only
          dependency is from implicit op order, with no intervening non-simultaenous ops.
        - concurrentWith(Op other): A "must be concurrent" check allowing an underapproximation of 
          concurrency. Currently this is only used when checking the legality of simultanousWith().
    }];
    let methods = [
        InterfaceMethod<
            [{Returns the SSA value handles whose events must complete before this event can start}],
            "::llvm::SmallVector<::mlir::Value>", "happensAfter", (ins)
        >,
        InterfaceMethod<
            [{Returns the SSA value handles whose events this event happens simultanously with}],
            "::llvm::SmallVector<::mlir::Value>", "simulatenousWith", (ins)
        >,
        InterfaceMethod<
            [{An underapproximating relation allowing concurrency between two ops. }],
            "bool", "concurrentWith", (ins "mlir::Operation *":$other)
        >
    ];
}


#endif // ENZYME_AD_JAX_DIALECT_DISTRIBUTED_INTERFACES