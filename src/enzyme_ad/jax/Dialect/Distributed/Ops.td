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

// Marker for distributed metadata ops that are semantically pure.
def PureDistributedMetadata : TraitList<[PureMetadataTrait]>;
// Marker for distributed metadata declarations that must be effectful.
def DeclarativeDistributedMetadata : TraitList<[DeclarativeMetadataTrait]>;

def PhysicalMeshOp
  : DistributedOp<"PhysicalMesh", [DeclarativeDistributedMetadata, Symbol]> {
  let description = [{Declares a mesh of device targets along with their local
                          communication hierarchy.}];
  let arguments = (ins SymbolNameAttr
                   : $sym_name, StrAttr
                   : $device_target,
                     ArrayAttr
                   : $axes // array of physical axis types.
  );
  let assemblyFormat =
      "$sym_name `device_target` $device_target `axes` $axes attr-dict";
  let hasVerifier = 1;
}

def GetPhysicalMeshAxesOp
    : DistributedOp<"GetPhysicalMeshAxes", [PureDistributedMetadata]> {
  let description =
      [{Resolves a module -
        level physical mesh symbol into one SSA axis value per mesh axis.}];
  let arguments = (ins FlatSymbolRefAttr : $physical_mesh);
  let results = (outs Variadic<PhysicalCommAxisType> : $axes);
  let assemblyFormat = "$physical_mesh attr-dict `:` type($axes)";
  let hasVerifier = 1;
}

def LogicalMeshAxesOp : DistributedOp<"LogicalMeshAxes", [
  DeclarativeDistributedMetadata, DeclareOpInterfaceMethods<InferTypeOpInterface>
]> {
  let description = [{Declares a new logical mesh of axes,
                      which may or may not correspond to a physical mesh.}];
  let arguments = (ins DenseI32ArrayAttr : $axis_extents);
  let results = (outs Variadic<LogicalMeshAxisType> : $axes);
  let assemblyFormat = "$axis_extents attr-dict `:` type($axes)";
  let hasVerifier = 1;
}

def ReplicationAxisOp : DistributedOp<"ReplicationAxis", [
  PureDistributedMetadata, DeclareOpInterfaceMethods<InferTypeOpInterface>
]> {
  let description = [{Declares a new replication axis,
                      which is a logical mesh axis that is replicated across all
                          devices in the physical mesh.}];
  let arguments = (ins I32Attr : $extent);
  let results = (outs ReplicationAxisType : $axis);
  let assemblyFormat = "$extent attr-dict `:` type($axis)";
}

// Per-tensor sharding metadata keyed by partitioning-axis indices.
// - dim_partitioning_axes: one DenseI64ArrayAttr per tensor dim, where each
//   entry stores zero or more indices into the owning op's partitioning_axes.
// - unreduced_axes: tensor-dim indices that remain unreduced.
def IndexedTensorShardingAttr : AttrDef<DistributedDialect, "IndexedTensorSharding"> {
  let summary = "Tensor sharding with integer partitioning-axis indices";
  let cppNamespace = "::mlir::enzyme::distributed";
  let mnemonic = "indexed_tensor_sharding";

  let parameters = (ins
    ArrayRefParameter<"DenseI64ArrayAttr">:$dim_partitioning_axes,
    "DenseI64ArrayAttr":$unreduced_axes
  );

  let assemblyFormat = [{
    `<` `dim_partitioning_axes` `=` $dim_partitioning_axes `,`
    `unreduced_axes` `=` $unreduced_axes `>`
  }];
}

// One IndexedTensorShardingAttr per operand/result value.
def IndexedTensorShardingPerValueAttr : AttrDef<DistributedDialect, "IndexedTensorShardingPerValue"> {
  let summary = "Indexed tensor sharding for each value";
  let cppNamespace = "::mlir::enzyme::distributed";
  let mnemonic = "indexed_tensor_sharding_per_value";

  let parameters = (ins
    ArrayRefParameter<"IndexedTensorShardingAttr">:$shardings
  );

  let assemblyFormat = [{
    `<` $shardings `>`
  }];
}

def DistributedFunctionOp : DistributedOp<"DistributedFunction", [
  Symbol,
  OpAsmOpInterface,
  SingleBlock
]> {
  let description = [{
    Function-like distributed region that can capture enclosing module SSA
    metadata (for example axis declarations). Unlike func.func, this op is not
    isolated from above. The function carries explicit partitioning axis
    operands and indexed sharding metadata for its inputs and outputs.
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttrOf<FunctionType>:$function_type,
    Variadic<FactorGroupType>:$partitioning_axes,
    IndexedTensorShardingPerValueAttr:$argument_shardings,
    IndexedTensorShardingPerValueAttr:$output_shardings,
    OptionalAttr<StrAttr>:$sym_visibility,
    OptionalAttr<DictArrayAttr>:$arg_attrs,
    OptionalAttr<DictArrayAttr>:$res_attrs
  );
  let regions = (region AnyRegion:$body);
  let hasVerifier = 1;
}

def DistributedKernelOp : DistributedOp<"DistributedKernel", [SingleBlock, AttrSizedOperandSegments]> {
  let description = [{Contains a computation subgraph that will never internally 
    create collectives no matter how the partitioning axes are chosen. The results
    may require a reduction after, as indicated by the unreduced axes on the 
    output sharding attributes.}];
  let arguments = (ins Variadic<AnyType> : $arguments,
                   Variadic<FactorGroupType> : $partitioning_axes,
                   IndexedTensorShardingPerValueAttr : $argument_shardings,
                   IndexedTensorShardingPerValueAttr : $output_shardings);
  let results = (outs Variadic<AnyType> : $returns);
  let hasVerifier = 1;
}



def DistributedYieldOp : DistributedOp<"DistributedYield", [Terminator]> {
  let description = [{
        Terminator for distributed functions. Yields local values back to the caller.
  }];
  let arguments = (ins Variadic<AnyType> : $returns);
  let assemblyFormat = "$returns type($returns) attr-dict";
}

def DistributedAwait
    : DistributedOp<"Await",
                    [DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let description = [{
    Awaits a value that is being computed asynchronously
            .Extracts the value from an AsynchHandleType.This is a no -
        op if the value is already available.
  }];
  let arguments = (ins AsynchHandleType : $async_handle);
  let results = (outs AnyType : $value);
  let assemblyFormat =
      "$async_handle attr-dict `:` type($async_handle) `->` type($value)";
  let hasVerifier = 1;
}

def DistributedCollectiveOp
    : DistributedOp<"Collective",
                    [DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let description = [{
  An operation representing reductions and resharding along multiple axes. Does not
        cover collective permutes and point-to-point communication. 
        
        A collective operation happens over the mesh and tile axis, and maps input mesh/tile 
        axis factors to output mesh/tile axis factors. Essentially, this is a 
        reduce-transpose-reshape operation. 

        Reductions are specified as different groups of axis factors to reduce over.
        Reductions take custom binary bodies as inline regions, which must be associative.
        Each group of reduction axes may have a different reduction body,
        but these must be commutative with each other: for now, use two distinct collectives
        when reductions cannot be fused.

        The transpose/reshape is specified as a mapping between non-reduced input/output axis factors.
        This mapping space can also include replicate(N) factors on either side: mapping an input axis
        to replicate means that you are asserting each device on that dimension contributes identical values,
        while mapping replicate to an output axis means you are cloning the result over that axis.

        Input and output meshes may differ; the axis.map + reduction groups
        define the remapping between mesh/tensor index spaces.

        Inputs:
        - input_object: a shaped tensor (tensor or memref).
        - input_mesh: the mesh axes (FactorGroup) that the input object is distributed over.
        - output_mesh: the mesh axes (FactorGroup) that the output object is distributed over.
        - reduction_groups: a list of FactorGroups to reduce over (pairwise disjoint).
        - reduction bodies: one single-block region per reduction group.
        - mapping: an axis.map value whose lhs/rhs groups encode the
          post-reduction input/output mapping.
        - output_type: the collective output tensor type in local indexing
          space. This may differ from the pre-collective producer-local type.

        Output:
        - An AsynchHandleType wrapping the output tensor type.

        Custom format:
        - Reductions are specified as (%axisgroup @reduction_function), and are
          collected by the parser into the reduction_groups list and
          reduction_functions attributes.
        - Mappings are specified via one axis.map SSA value.
  }];
  let arguments = (ins
        AnyTypeOf<[AnyTensor, AnyMemRef]>:$input_object,
        FactorGroupType:$input_mesh,
        FactorGroupType:$output_mesh,
        Variadic<FactorGroupType>:$reduction_groups,
        AxisMapType:$mapping,
        TypeAttrOf<AnyTensor>:$output_type
    );
  let results = (outs AsynchHandleType : $async_handle);
  let regions = (region VariadicRegion<SizedRegion<1>>:$reduction_bodies);
  let assemblyFormat =
      "$input_object `:` type($input_object) `on` $input_mesh `:` "
      "type($input_mesh) `to` $output_type `on` $output_mesh `:` "
      "type($output_mesh) `reduces` custom<ReductionGroups>($reduction_groups) `:` type($reduction_groups) `maps` $mapping `:` "
      "type($mapping) $reduction_bodies attr-dict";
  let hasVerifier = 1;
}
#endif // ENZYME_AD_JAX_DIALECT_DISTRIBUTED_OPS_TD