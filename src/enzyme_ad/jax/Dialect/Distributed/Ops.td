#ifndef ENZYME_AD_JAX_DIALECT_DISTRIBUTED_OPS_TD
#define ENZYME_AD_JAX_DIALECT_DISTRIBUTED_OPS_TD

include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "stablehlo/dialect/Base.td"
include "shardy/dialect/sdy/ir/attrs.td"

include "Dialect.td"
include "Types.td"
include "Interfaces.td"

defvar TensorType = HLO_Tensor;
defvar ShardedTensorType = HLO_Tensor; // Subject to change, for now shardy just adds attributes on top of a regular tensor.
defvar ShardingAttribute = Sdy_TensorSharding;

// Mesh definition ops
def PhysicalMeshOp : DistributedOp<"PhysicalMesh", [Symbol]> {
    let description = [{Defines a physical mesh of devices based on the hardware topology.
    A physical mesh is a Cartesian product of zero or more physical axes. Axes should be distinct!
    Two copies of the same axis is permitted (but may better be represented as a larger axis), but
    passing the same axis twice is an error.}];
    let arguments = (ins
        SymbolNameAttr:$sym_name,
        FlatSymbolRefArrayAttr:$axes,
        OptionalAttr<StrAttr>:$sym_visibility
    );
    let assemblyFormat = "$sym_name $axes attr-dict";
    let hasVerifier = 1;
}

def LogicalMeshOp : DistributedOp<"LogicalMesh"> {
    let description = [{Defines a logical mesh of devices, intended to represent 
    the mesh used by the program. Defined in terms of its projection
    onto a physical mesh via a product of logical axes.}];
    let arguments = (ins
        Variadic<LogicalCommAxisType>:$axes
    );
    let results = (outs LogicalMeshType:$mesh);
    let assemblyFormat = "$axes attr-dict";
    let hasVerifier = 1;
}

def AxisAllToAllOp : DistributedOp<"AxisAllToAll", [
    Symbol,
    DeclareOpInterfaceMethods<PhysicalCommAxisOpInterface>
]> {
    let description = [{
       A communication channel where any device can communicate to any other device without
       bandwidth conflict or latency scaling with index distance. 
     }];
    let arguments = (ins
        SymbolNameAttr:$sym_name,
        UI32Attr:$axis_size,
        // TODO should these be floats? Units library?
        UI64Attr:$bandwidth, // in bytes per second
        UI64Attr:$latency, // in picoseconds.
        OptionalAttr<StrAttr>:$sym_visibility
    );
    let assemblyFormat = "$sym_name $axis_size $bandwidth $latency attr-dict";
}

def AxisFactorOp : DistributedOp<"AxisFactor", [
    DeclareOpInterfaceMethods<InferTypeOpInterface>,
    DeclareOpInterfaceMethods<LogicalCommAxisOpInterface>
]> {
    let description = [{
       Factors a physical axis into multiple logical axes, according to a static array of factor sizes.
       The lowest index of factors corresponds to the lowest-significance / most contiguous new axis.
       Factors must multiply into the original physical axis size.
       Returns a logical axis per factor.
     }];
    let arguments = (ins
        FlatSymbolRefAttr:$physical_axis,
        I32ArrayAttr:$factors
    );
    let results = (outs Variadic<LogicalCommAxisType>:$logical_axes);
    let assemblyFormat = "$physical_axis $factors attr-dict";
    let hasVerifier = 1; // TODO: verify factors multiply to physical axis size, and correct number of outputs.
}


def AxisProductOp : DistributedOp<"AxisProduct", [
    DeclareOpInterfaceMethods<LogicalCommAxisOpInterface>
]> {
    let description = [{
       Combines multiple logical axes into a single logical axis. Logical axis must be disjoint,
       but may come from any mix of physical axes, including different strides over the same physical
       axis. The order encodes the significance, with the lowest-index/first logical axis being
       index as the least-significant/most contiguous dimension. 
       For simplicity for implementation, we require any factors of the same physical axis to be
       defined from the same factorization op. (For instance, this makes disjointness easier)
     }];
    let arguments = (ins
        Variadic<LogicalCommAxisType>:$logical_axes
    );
    let results = (outs LogicalCommAxisType:$logical_axis);
    let assemblyFormat = "$logical_axes attr-dict";
    let hasVerifier = 1; // TODO: axis disjointness.
}

def MeshComputationOp : DistributedOp<"MeshComputation"> {
    let description = [{
       A computation performed across a mesh. SPMD style with collectives.
       (Multiple) "Region" computations can be inserted into a mesh computation
       to break SPMD and perform different computations on devices within the region.
     }];
    let arguments = (ins
        LogicalMeshType:$mesh, Variadic<ShardedTensorType>:$input_tensors
    );
    let results = (outs Variadic<ShardedTensorType>:$output_tensors);
    let regions = (region SizedRegion<1>:$body);
    let assemblyFormat = "$mesh $input_tensors type($input_tensors) `->` type($output_tensors) $body attr-dict";
}

def RegionComputationOp : DistributedOp<"RegionComputation"> {
    let description = [{
       A computation performed across a submesh region of a parent MeshComputation. Any axis (factors) within the 
       region are decomposed into individual programs, while axes outside the region replicate the same program. 
     }];
    let arguments = (ins
        LogicalMeshType:$submesh, Variadic<ShardedTensorType>:$input_tensors
    );
    let results = (outs Variadic<ShardedTensorType>:$output_tensors);
    let regions = (region VariadicRegion<SizedRegion<1>>:$device_bodies);
    let assemblyFormat = "$submesh $input_tensors type($input_tensors) `->` type($output_tensors) $device_bodies attr-dict";
    // TODO: verify number of regoins matches number of devices in submesh,
    // each region has only one block, the block arguments match the shard componets
    // of the input tensors, and the region outputs match the output tensors.
    // Additionally, this op can only be called within a MeshComputationOp, and the submesh
    // must be subset or equal to the parent mesh.
    let hasVerifier = 1;
}


// Need a "return" terminator for both mesh and region computations.
def DistributedYieldOp : DistributedOp<"DistributedYield", [Terminator]> {
    let description = [{
       Terminator for MeshComputation and RegionComputation ops. Returns values from the computation. 
     }];
    let arguments = (ins Variadic<ShardedTensorType>:$output_tensors);
    let assemblyFormat = "$output_tensors type($output_tensors) attr-dict";
}

// Communication Ops
def CollectiveReshardOp : DistributedOp<"CollectiveReshard"> {
    let description = [{
       A resharding collective communication where the resharding pattern dictates the 
       communication pattern.
     }];
    let arguments = (ins
        LogicalMeshType:$mesh,
        ShardedTensorType:$input_tensor,
        ShardingAttribute:$input_sharding,
        ShardingAttribute:$output_sharding
    );
    let results = (outs CollectiveTokenType:$token, ShardedTensorType:$output_tensor);
    let assemblyFormat = "$mesh $input_tensor type($input_tensor) `->` type($token) `,` type($output_tensor) attr-dict";
}

#endif // ENZYME_AD_JAX_DIALECT_DISTRIBUTED_OPS_TD