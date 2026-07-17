#ifndef ENZYME_AD_JAX_DIALECT_DISTRIBUTED_OPS_TD
#define ENZYME_AD_JAX_DIALECT_DISTRIBUTED_OPS_TD

include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "stablehlo/dialect/Base.td"
include "shardy/dialect/sdy/ir/attrs.td"

include "Dialect.td"
include "Types.td"
include "Interfaces.td"
include "src/enzyme_ad/jax/Dialect/Axis/Dialect.td"
include "src/enzyme_ad/jax/Dialect/Axis/Types.td"

// Marker for distributed metadata ops. Includes builtin Pure.
def PureDistributedMetadata : TraitList<[MetadataOpTrait]>;

def PhysicalMeshOp
    : DistributedOp<"PhysicalMesh", [PureDistributedMetadata, Symbol]> {
  let description = [{
        Declares a mesh of device targets along with their local communication hierarchy.
        }];
    let arguments = (ins
        SymbolNameAttr:$sym_name,
        StrAttr:$device_target,
        ArrayAttr:$axes // array of physical axis types.
    );
    let assemblyFormat = "$sym_name `device_target` $device_target `axes` $axes attr-dict";
    let hasVerifier = 1;
}

def GetPhysicalMeshAxesOp : DistributedOp<"GetPhysicalMeshAxes", [PureDistributedMetadata]> {
    let description = [{
        Resolves a module-level physical mesh symbol into one SSA axis value per mesh axis.
     }];
    let arguments = (ins FlatSymbolRefAttr:$physical_mesh);
    let results = (outs Variadic<PhysicalCommAxisType>:$axes);
    let assemblyFormat = "$physical_mesh attr-dict `:` type($axes)";
    let hasVerifier = 1;
}

def LogicalMeshAxesOp : Distr ibutedOp<"LogicalMeshAxes", [
    PureDistributedMetadata, DeclareOpInterfaceMethods<InferTypeOpInterface>
  ]> {
    let description = [{
        Declares a new logical mesh of axes, which may or may not correspond to a
        physical mesh. 
    }];
    let arguments = (ins
        DenseI32ArrayAttr:$axis_extents
    );
    let results = (outs Variadic<LogicalMeshAxisType>:$axes);
    let assemblyFormat = "$axis_extents attr-dict `:` type($axes)";
    let hasVerifier = 1;
}

def ReplicationAxisOp : DistributedOp<"ReplicationAxis", [PureDistributedMetadata, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
    let description = [{
        Declares a new replication axis, which is a logical mesh axis that is replicated
        across all devices in the physical mesh. 
    }];
    let arguments = (ins
        I32Attr:$extent
    );
    let results = (outs ReplicationAxisType:$axis);
    let assemblyFormat = "$extent attr-dict `:` type($axis)";
}


def MeshComputationOp : DistributedOp<"MeshComputation", [
    HasOnlyGraphRegion,
    NoTerminator,
    Symbol,
    SymbolTable,
    IsolatedFromAbove
]> {
    let description = [{
        Represents a closed distributed program. Contains a mix non-computational SSA values
        (operations that represent a static computation, like axis algebra) and distributed
        functions. Distributed functions can call into normal single-device MLIR functions
        outside of this region, but all functions with internal distributed logic (communication)
        must be fully contained in this region, allowing us to reason about the whole call graph
        for optimization decisions.
     }];
    let arguments = (ins
        SymbolNameAttr:$sym_name,
        FlatSymbolRefAttr:$physical_mesh
    );
    let regions = (region SizedRegion<1>:$body);
        let assemblyFormat = [{
            $sym_name `mesh` $physical_mesh $body attr-dict
        }];
    let hasVerifier = 1;
}

def DistributedFunctionO p : DistributedOp<"Function", [Symbol]> {
      let description = [{
        Represents a distributed function. Functions have an execution context indicating what subspace of
        the logical/physical mesh they are executed on, and can only be called from a matching enclosing 
        context. All arguments and return values are local values, rather than global tensors with
        a sharding annotation.
    }];
    let arguments = (ins
        SymbolNameAttr:$sym_name,
        FactorGroupType:$execution_context,
        ArrayAttr:$argument_types,
        ArrayAttr:$return_types
    );
    let regions = (region SizedRegion<1>:$body);
    let assemblyFormat = [{
        $sym_name `context` $execution_context
        `:` type($execution_context)
        `arg_types` $argument_types `ret_types` $return_types
        $body attr-dict
    }];
    let hasVerifier = 1;
}

def DistributedYieldOp : DistributedOp<"DistributedYield", [Terminator]> {
    let description = [{
        Terminator for distributed functions. Yields local values back to the caller.
     }];
    let arguments = (ins Variadic<AnyType>:$returns);
    let assemblyFormat = "$returns type($returns) attr-dict";
}

def DistributedAwait : DistributedOp<"Await", [DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
    let description = [{
        Awaits a value that is being computed asynchronously. Extracts the value
        from an AsynchHandleType. This is a no-op if the value is already available.
    }];
    let arguments = (ins AsynchHandleType:$async_handle);
    let results = (outs AnyType:$value);
    let assemblyFormat = "$async_handle attr-dict `:` type($async_handle) `->` type($value)";
    let hasVerifier = 1;
}

def DistributedCallOp : DistributedOp<"DistributedCall", []> {
    let description = [{
        Calls a distributed function. The execution context must match the enclosing context.
     }];
    let arguments = (ins
        FlatSymbolRefAttr:$callee,
        FactorGroupType:$replicate_over,
        Variadic<AnyType>:$arguments
    );
    let results = (outs Variadic<AnyType>:$returns);
    let assemblyFormat = "$callee `replicate_over` $replicate_over $arguments attr-dict `:` type($replicate_over) `,` type($arguments) `->` type($returns)";
    let hasVerifier = 1;
}

def DistributedCollectiveOp : DistributedOp<"Collective", [
    DeclareOpInterfaceMethods<InferTypeOpInterface>
]> {
    let description = [{
        An operation representing reductions and resharding along multiple axis. Does not 
        cover collective permutes and point-to-point communication. 
        
        A collective operation happens over the mesh and tile axis, and maps input mesh/tile 
        axis factors to output mesh/tile axis factors. Essentially, this is a 
        reduce-transpose-reshape operation. 

        Reductions are specified as different groups of axis factors to reduce over.
        Reductions take custom binary functions as input, which must be associative.
        Each group of reduction axes may have a different reduction function function, 
        but these must be commutative with each other: for now, use two distinct collectives 
        when reductions cannot be fused.

        The transpose/reshape is specified as a mapping between non-reduced input/output axis factors.
        This mapping space can also include replicate(N) factors on either side: mapping an input axis
        to replicate means that you are asserting each device on that dimension contributes identical values,
        while mapping replicate to an output axis means you are cloning the result over that axis.

        Collectives must be between the same mesh (verified via index space equality).

        Inputs:
        - input_object: a shaped tensor (tensor or memref).
        - input_mesh: the mesh axes (FactorGroup) that the input object is distributed over.
        - output_mesh: the mesh axes (FactorGroup) that the output object is distributed over.
                       Must have equal index space to input_mesh.-
                        reduction_groups
          : a list of FactorGroups to reduce over(pairwise disjoint)
                .-
                        reduction_functions
          : ArrayAttr of FlatSymbolRefAttr naming single -
            device reduction functions.Size must match reduction_groups.-
            mapping : an axis.map value whose lhs / rhs groups encode the post -
            reduction input / output mapping
                                  .

                              Output
          : -An AsynchHandleType wrapping the output tensor type.

            Custom format
          : -Reductions are specified as(% axisgroup @reduction_function),
          and are collected by the parser into the reduction_groups list and
                  reduction_functions attributes.-
              Mappings are specified via one axis.map SSA value.
    }];
    let arguments = (ins
        AnyTypeOf<[AnyTensor, AnyMemRef]>:$input_object,
        FactorGroupType:$input_mesh,
        FactorGroupType:$output_mesh,
        Variadic<FactorGroupType>:$reduction_groups,
        ArrayAttr:$reduction_functions,
        AxisMapType:$mapping,
        TypeAttrOf<AnyTensor>:$output_tensor_type
    );
    let results = (outs AsynchHandleType : $async_handle);
    let assemblyFormat = [{
        $input_object `:` type($input_object) `on` $input_mesh `:` type($input_mesh) 
        `to` $output_tensor_type `on` $output_mesh `:` type($output_mesh)
        `reduces` custom<ReductionGroups>($reduction_groups, $reduction_functions)
        `:` type($reduction_groups)
        `maps` $mapping `:` type($mapping)
        attr-dict
    }];
    let hasVerifier = 1;
}
#endif // ENZYME_AD_JAX_DIALECT_DISTRIBUTED_OPS_TD