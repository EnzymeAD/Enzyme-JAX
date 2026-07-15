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

defvar TensorType = HLO_Tensor;
defvar ShardedTensorType = HLO_Tensor; // Subject to change, for now shardy just adds attributes on top of a regular tensor.
defvar ShardingAttribute = Sdy_TensorSharding;

// Marker for distributed metadata ops. Includes builtin Pure.
def PureDistributedMetadata : TraitList<[MetadataOpTrait]>;

def PhysicalMeshOp : DistributedOp<"PhysicalMesh", [PureDistributedMetadata, Symbol]> {
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

def MeshComputationOp : DistributedOp<"MeshComputation", [
    HasOnlyGraphRegion,
    NoTerminator,
    Symbol,
    SymbolTable
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

def DistributedFunctionOp : DistributedOp<"Function", [Symbol]> {
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
#endif // ENZYME_AD_JAX_DIALECT_DISTRIBUTED_OPS_TD