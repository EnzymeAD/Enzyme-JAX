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

def GetPhysicalAxisOp : DistributedOp<"GetPhysicalAxis", [PureDistributedMetadata]> {
    let description = [{
        Resolves a module-level physical axis symbol into an SSA axis value.
     }];
    let arguments = (ins FlatSymbolRefAttr:$physical_axis);
    let results = (outs PhysicalCommAxisType:$axis);
    let assemblyFormat = "$physical_axis attr-dict `:` type($axis)";
    let hasVerifier = 1;
}

def GenericAxisOp : DistributedOp<"GenericAxis", [PureDistributedMetadata]> {
    let description = [{
        Produces a generic SSA axis value of the requested axis type.
     }];
    let results = (outs AnyType:$axis);
    let assemblyFormat = "attr-dict `:` type($axis)";
    let hasVerifier = 1;
}

def AxisFactorOp : DistributedOp<"AxisFactor", [
    PureDistributedMetadata,
    DeclareOpInterfaceMethods<InferTypeOpInterface>
]> {
    let description = [{
       Factors an axis value into multiple axis factors according to a static
       array of extents.
       Leftmost factors are most major. For factor i:
         stride_i = product(extents_j for j > i)
       Factors must multiply to the extent of the source axis.
     }];
    let arguments = (ins
        AnyType:$axis,
        DenseI32ArrayAttr:$factor_extents
    );
    let results = (outs Variadic<AxisFactorType>:$axis_factors);
    let assemblyFormat = "$axis $factor_extents attr-dict `:` type($axis) `->` type($axis_factors)";
    let hasVerifier = 1;
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
        Variadic<LogicalMeshAxisType>:$spmd_axes,
        Variadic<LogicalMeshAxisType>:$mpmd_axes,
        ArrayAttr:$shardy_axis_names,
        Variadic<ShardedTensorType>:$input_tensors,
        I32Attr:$num_device_bodies,
        I32Attr:$num_communication_bodies
    );
    let results = (outs Variadic<ShardedTensorType>:$output_tensors);
    let regions = (region VariadicRegion<SizedRegion<1>>:$bodies);
        let assemblyFormat = [{
            `spmd` `(` $spmd_axes `:` type($spmd_axes) `)`
            `mpmd` `(` $mpmd_axes `:` type($mpmd_axes) `)`
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
        // Axis flattening treats lower-index (leftmost) MPMD axes as more
        // significant.
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

#endif // ENZYME_AD_JAX_DIALECT_DISTRIBUTED_OPS_TD