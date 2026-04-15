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

def DistributedMetadataTrait
    : NativeOpTrait<"enzyme::distributed::MetadataTrait">;

// Marker for distributed metadata ops. Includes builtin Pure.
def PureDistributedMetadata : TraitList<[Pure, DistributedMetadataTrait]>;

// Mesh definition ops
def PhysicalMeshOp : DistributedOp<"PhysicalMesh", [Symbol, PureDistributedMetadata]> {
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
        let extraClassDeclaration = [{
            unsigned getPhysicalAxisPosition(FlatSymbolRefAttr axis);
        }];
}

def LogicalMeshOp : DistributedOp<"LogicalMesh", [PureDistributedMetadata]> {
    let description = [{Defines a logical mesh of devices, intended to represent 
    the mesh used by the program. Defined in terms of its projection
    onto a physical mesh via a product of logical axes.}];
    let arguments = (ins
        FlatSymbolRefAttr:$physical_mesh,
        Variadic<LogicalCommAxisType>:$axes
    );
    let results = (outs LogicalMeshType:$mesh);
    let assemblyFormat = "$axes attr-dict";
    let hasVerifier = 1;
        let extraClassDeclaration = [{
            ::mlir::FailureOr<PhysicalMeshOp> resolvePhysicalMesh();
            ::mlir::LogicalResult
            resolveToAtomicFactors(::llvm::SmallVectorImpl<
                ::mlir::enzyme::distributed::TypedOpResult<
                    ::mlir::enzyme::distributed::LogicalCommAxisType>> &atomicFactors);
            bool isDisjoint();
            bool isSubmesh(LogicalMeshOp submesh);
            ::mlir::FailureOr<int64_t> getMeshSize();
        }];
}

def AxisAllToAllOp : DistributedOp<"AxisAllToAll", [
    Symbol,
    PureDistributedMetadata,
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
    PureDistributedMetadata,
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
    PureDistributedMetadata,
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

def MeshComputationOp : DistributedOp<"MeshComputation", [
    AttrSizedOperandSegments,
    DeclareOpInterfaceMethods<LaneOpInterface>
]> {
    let description = [{
       Unified mesh computation op with explicit SPMD and MPMD decomposition.
       SPMD axes define the replicated program space, while MPMD axes define per-device programs.
       Device computation regions are indexed by the MPMD device space and implicitly replicated
         over the SPMD axes. Communication regions define cross-device transfer/collective behavior.
         All bodies are stored in a single variadic region list. The first num_device_bodies regions
         are device programs. The remaining regions are communication programs, ordered with SPMD
         communication bodies first and MPMD communication bodies after that.
     }];
    let arguments = (ins
        Variadic<LogicalCommAxisType>:$spmd_axes,
        Variadic<LogicalCommAxisType>:$mpmd_axes,
        ArrayAttr:$shardy_axis_names,
        Variadic<ShardedTensorType>:$input_tensors,
        I32Attr:$num_device_bodies,
        I32Attr:$num_communication_bodies
    );
    let results = (outs Variadic<ShardedTensorType>:$output_tensors);
    let regions = (region VariadicRegion<SizedRegion<1>>:$bodies);
        let assemblyFormat = [{
            `spmd` `(` $spmd_axes `)` `mpmd` `(` $mpmd_axes `)`
            `shardy_axis_names` $shardy_axis_names
            $input_tensors type($input_tensors)
            `->` type($output_tensors)
            `num_device_bodies` $num_device_bodies
            `num_communication_bodies` $num_communication_bodies
            `regions` $bodies
            attr-dict
        }];
    let extraClassDeclaration = [{
        // Multi-axis MPMD device coordinates. One coordinate per MPMD axis.
        using DeviceIndex = ::llvm::SmallVector<unsigned, 4>;

                // Const overloads for segmented operands/attrs generated as
                // mutable-only accessors by ODS.
            ::mlir::ValueRange getSpmdAxes() const;
            ::mlir::ValueRange getMpmdAxes() const;
            uint32_t getNumDeviceBodies() const;
            uint32_t getNumCommunicationBodies() const;

        // Axis-name/axis-value conversion helpers for mixed Shardy/distributed
        // lowering states.
        ::mlir::FailureOr<::mlir::Value>
        findLogicalAxisForShardyAxisName(::llvm::StringRef shardyAxisName) const;
        ::mlir::FailureOr<::mlir::StringAttr>
        findShardyAxisNameForLogicalAxis(::mlir::Value logicalAxis) const;

        // Looks up one of the input tensor operands by SSA value.
        ::mlir::FailureOr<unsigned>
        findInputTensorIndex(::mlir::Value tensor) const;

        // Rewrites the sharding attached to one of the input tensor operands.
        ::mlir::LogicalResult
        setInputTensorSharding(::mlir::Value tensor,
                                                        ::mlir::sdy::TensorShardingAttr sharding);

        // Indices are zero-based within the device-body partition.
      ::mlir::Region &getDeviceBody(unsigned idx);
      const ::mlir::Region &getDeviceBody(unsigned idx) const;

        // Converts a DeviceIndex into a flat index in the device-body partition.
        // Axis flattening treats higher-index MPMD axes as more significant.
        ::mlir::FailureOr<unsigned>
        findComputationBodyIndexByDeviceIndex(const DeviceIndex &deviceIndex) const;

        // Convenience helpers for direct DeviceIndex -> Region lookup.
        ::mlir::Region &getDeviceBodyByDeviceIndex(const DeviceIndex &deviceIndex);
        const ::mlir::Region &
        getDeviceBodyByDeviceIndex(const DeviceIndex &deviceIndex) const;

        // Indices are zero-based within the combined communication-body partition.
        // SPMD communication bodies occupy the lower indices, followed by MPMD
        // communication bodies.
      ::mlir::Region &getCommunicationBody(unsigned idx);
      const ::mlir::Region &getCommunicationBody(unsigned idx) const;

        // Returns the index of the communication body associated with the given
        // logical axis, relative to the combined communication-body partition.
        // SPMD axes map to lower indices than MPMD axes.
        ::mlir::FailureOr<unsigned>
        findCommunicationBodyIndexForAxis(::mlir::Value axis) const;

        // Convenience helper for resolving the communication body associated with
        // the given logical axis.
      ::mlir::Region &getCommunicationBodyForAxis(::mlir::Value axis);
      const ::mlir::Region &getCommunicationBodyForAxis(::mlir::Value axis) const;
    }];
    let hasVerifier = 1;
}


// Need a "return" terminator for both mesh and region computations.
def DistributedYieldOp : DistributedOp<"DistributedYield", [Terminator]> {
    let description = [{
        Terminator for MeshComputation regions. Yields tensor input shard values back to 
        a more global context.
     }];
    let arguments = (ins Variadic<ShardedTensorType>:$output_tensors);
    let assemblyFormat = "$output_tensors type($output_tensors) attr-dict";
}

def ShardingReduceOp : DistributedOp<"ShardingReduce", [PureDistributedMetadata]> {
    let description = [{
       An axis used exclusively to indicate a reduction axis in the right hand side of a collective
       communication. 
     }];
    let arguments = (ins
    );
    let results = (outs ShardingAxisType:$reduction_axis);
}

def ShardingOp : DistributedOp<"Sharding", [PureDistributedMetadata]> {
    let description = [{
       A product of logical axes and/or reduction axes used to indicate the sharding of a tensor.
       The presence of a reduction axis indicates that the tensor is replicated along that axis, and
       that dimension is reduced across in collectives. 
     }];
    let arguments = (ins
        Variadic<LogicalCommAxisType>:$axes
    );
    let results = (outs ShardingAxisType:$sharding);
}


// Communication Ops
def CollectiveOp : DistributedOp<"Collective", [
    DeclareOpInterfaceMethods<HandleDeclarationOpInterface>,
    DeclareOpInterfaceMethods<InferTypeOpInterface>,
    AttrSizedOperandSegments
]> {
    let description = [{
       A resharding collective communication where the resharding pattern dictates the 
       communication pattern. Not marked Pure to prevent CSE from combining multiple
       collectives, as each generates a distinct Transfer op in the communication body.
     }];
    let arguments = (ins
        Variadic<LogicalCommAxisType>:$axes,
        TypeAttr:$global_input_tensor_type,
        TypeAttr:$global_output_tensor_type,
        TypeAttr:$local_input_tensor_type,
        TypeAttr:$local_output_tensor_type,
        Variadic<ShardingAxisType>:$input_sharding,
        Variadic<ShardingAxisType>:$output_sharding
    );
    let results = (outs MessageTokenType:$token);
    let assemblyFormat = "$axes `(`$global_input_tensor_type `>` $local_input_tensor_type`,` $input_sharding `)` `->` `(`$global_output_tensor_type `>` $local_output_tensor_type`,` $output_sharding`)` attr-dict";
    // let hasVerifier = 1; // TODO verify sharding axes match the mesh, and that the input and output shardings are compatible with each other and the global/local tensor shapes.
}

def SubmeshCollectivePartsOp : DistributedOp<"SubmeshCollectiveParts", [PureDistributedMetadata, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
    let description = [{
       Decomposes a submesh collective into distinct send/recieve tokens 
       per submesh device, which can then be manipulated individually
       }];
    let arguments = (ins
    MessageTokenType:$collective,
    LogicalMeshType:$submesh,
    TypeAttr:$input_tensor_type,
    TypeAttr:$output_tensor_type
    );
    let results = (outs Variadic<MessageTokenType>:$device_specific_tokens);
    let assemblyFormat = "$collective $submesh $input_tensor_type`->` $output_tensor_type attr-dict";
}

def SendOp : DistributedOp<"Send", [DeclareOpInterfaceMethods<EventOpInterface>]> {
    let description = [{
         Marks a tensor as ready to send in a one-way communication asynchronously.
     }];
    let arguments = (ins
    MessageTokenType:$token,
    TensorType:$message
    );
    let assemblyFormat = "$token $message type($message) attr-dict";
}

def RecvOp : DistributedOp<"Recv", [DeclareOpInterfaceMethods<EventOpInterface>]> {
    let description = [{
       Receives the tensor value from an asynchronous one-way communication.
     }];
    let arguments = (ins MessageTokenType:$token);
    let results = (outs TensorType:$message);
    let assemblyFormat = "$token type($message) attr-dict";
}

def TransferOp : DistributedOp<"Transfer", [DeclareOpInterfaceMethods<EventOpInterface>]> {
    let description = [{
       Declares communication transfer work associated with a collective token.
       This op is intended only for communication regions of MeshComputation.
     }];
    let arguments = (ins MessageTokenType:$token);
    let assemblyFormat = "$token attr-dict";
    let hasVerifier = 1;
}

#endif // ENZYME_AD_JAX_DIALECT_DISTRIBUTED_OPS_TD