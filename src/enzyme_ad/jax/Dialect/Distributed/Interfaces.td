#ifndef ENZYME_AD_JAX_DIALECT_DISTRIBUTED_INTERFACES
#define ENZYME_AD_JAX_DIALECT_DISTRIBUTED_INTERFACES

include "mlir/IR/OpBase.td"

def PhysicalCommAxisOpInterface : OpInterface<"PhysicalCommAxisOpInterface"> {
    let description = [{
        Interface for ops that define a physical communication axis.
    }];

    let methods = [
        InterfaceMethod<
            [{Returns the extent of this physical axis in number of devices.}],
            "int64_t", "getPhysicalAxisExtent", (ins)>
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

def PartitioningAnchorOpInterface :
        OpInterface<"PartitioningAnchorOpInterface"> {
    let description = [{
        An op that can report, for one of its own tensor-typed operands or
        results, the per-tensor-dimension axis-algebra factor groups
        describing how that value's dimensions are currently partitioned --
        the same information a DistributedCastGlobalToLocalOp/
        CastLocalToGlobalOp's own positional `partitioning_axes` operand
        list carries, generalized to ops (like DistributedKernelOp) that
        address this information by index into a shared list instead of by
        argument position.

        Callers needing "what's the current local/global binding of this
        value" should look for the nearest op along its def-use chain that
        implements this interface and ask it directly, rather than assuming
        a specific op kind (a cast, a kernel, ...) is present.
    }];

    let methods = [
        InterfaceMethod<
            [{Returns, one entry per tensor dimension of `operand`, the list
              of factor-group values currently contributing to that
              dimension's partitioning -- usually a single value (a Cast
              op's own partitioning_axes[d] is already one combined
              FactorGroupType), but possibly several for an op (like
              DistributedKernelOp) whose dim_partitioning_axes can name a
              composite decomposition without pre-combining it into one
              product. Deliberately never synthesizes a combined product
              itself (a pure query, no builder involved) -- callers needing
              one combined value for a dimension build it themselves.
              Failure if not resolvable (e.g. a non-ranked-tensor operand,
              or an index this op doesn't have binding info for).}],
            "::mlir::FailureOr<::llvm::SmallVector<::llvm::SmallVector<::mlir::Value>>>",
            "getBindingInfoForOperand", (ins "::mlir::OpOperand &":$operand)>,
        InterfaceMethod<
            [{Symmetric with getBindingInfoForOperand, for one of this op's
              own results.}],
            "::mlir::FailureOr<::llvm::SmallVector<::llvm::SmallVector<::mlir::Value>>>",
            "getBindingInfoForResult", (ins "::mlir::OpResult":$result)>
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