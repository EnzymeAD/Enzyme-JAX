#ifndef ENZYME_AD_JAX_DIALECT_DISTRIBUTED_OPS_TD
#define ENZYME_AD_JAX_DIALECT_DISTRIBUTED_OPS_TD

include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "stablehlo/dialect/Base.td"
include "shardy/dialect/sdy/ir/attrs.td"

include "Dialect.td"
include "Types.td"
include "Interfaces.td"
include "Attributes.td"
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
  let arguments = (ins SymbolNameAttr : $sym_name, StrAttr : $device_target,
                     ArrayAttr : $axes // array of physical axis types.
  );
  let assemblyFormat =
      "$sym_name `device_target` $device_target `axes` $axes attr-dict";
  let hasVerifier = 1;
  let extraClassDeclaration = [{
    // Product of every physical axis's extent -- the total number of
    // devices this mesh spans.
    uint64_t getDeviceCount();
  }];
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
  DeclarativeDistributedMetadata, DeclareOpInterfaceMethods<InferTypeOpInterface>,
  DeclareOpInterfaceMethods<MaybeTemporaryInterface>
]> {
  let description = [{Declares a new logical mesh axis,
                      which may or may not correspond to a physical mesh.}];
  let arguments = (ins I64Attr : $extent);
  let results = (outs LogicalMeshAxisType : $axis);
  let assemblyFormat = "$extent attr-dict `:` type($axis)";
  let hasCanonicalizer = 1;
}

def ReplicationAxisOp : DistributedOp<"ReplicationAxis", [
  PureDistributedMetadata, DeclareOpInterfaceMethods<InferTypeOpInterface>
]> {
  let description = [{Declares a new replication axis,
                      which is a logical mesh axis that is replicated across all
                          devices in the physical mesh.}];
  let arguments = (ins I64Attr : $extent);
  let results = (outs ReplicationAxisType : $axis);
  let assemblyFormat = "$extent attr-dict `:` type($axis)";
}

def DeviceLocalAxisOp : DistributedOp<"DeviceLocalAxis", [
  DeclarativeDistributedMetadata, DeclareOpInterfaceMethods<InferTypeOpInterface>
]> {
  let description = [{Declares a new device-local axis, marking that part of
                      the parallelism should be placed entirely within a
                          local device instead of being sharded across the mesh.
                      Deliberately opts out of Pure (see DeclarativeMetadataTrait,
                      and LogicalMeshAxesOp's identical choice): we want to be
                      able to carry tensor / operator axis provenance through them,
                      so they must never be CSE'd together --
                      identity is the SSA value itself, not its extent.}];
  let arguments = (ins I64Attr : $extent);
  let results = (outs DeviceLocalAxisType : $axis);
  let assemblyFormat = "$extent attr-dict `:` type($axis)";
  let hasCanonicalizer = 1;
}

def DistributedFunctionOp : DistributedOp<"DistributedFunction", [
  Symbol,
  OpAsmOpInterface,
  SingleBlock,
  FunctionOpInterface
]> {
  let description = [{
    Function-like distributed region that can capture enclosing module SSA
    metadata (for example axis declarations). Unlike func.func, this op is not
    isolated from above. The function carries explicit partitioning axis
    operands and indexed sharding metadata for its inputs and outputs.

    Implements FunctionOpInterface so that Shardy utilities dispatching on
    that interface, such as looking up a block argument's sharding, treat
    this op's block arguments like a func.func's instead of falling back to
    the "unshardable argument" default meant for ops with no place to attach
    a per-argument sharding, like reduce bodies.

    Its arg_attrs/res_attrs/function_type fields already use the names
    FunctionOpInterface expects, so no renaming was needed to satisfy it.
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

  let extraClassDeclaration = [{
    //===------------------------------------------------------------------===//
    // FunctionOpInterface Methods
    //===------------------------------------------------------------------===//

    ::mlir::Region *getCallableRegion() { return &getBody(); }

    ::mlir::ArrayRef<::mlir::Type> getArgumentTypes() {
      return getFunctionType().getInputs();
    }

    ::mlir::ArrayRef<::mlir::Type> getResultTypes() {
      return getFunctionType().getResults();
    }
  }];
}

def DistributedCallOp : DistributedOp<"DistributedCall", [
  CallOpInterface,
  DeclareOpInterfaceMethods<SymbolUserOpInterface>,
  DeclareOpInterfaceMethods<PartitioningAnchorOpInterface>,
  AttrSizedOperandSegments
]> {
  let description = [{
    Calls a distributed.DistributedFunction. Like DistributedKernelOp, this op
    is the local/global boundary: $arguments/$returns are at LOCAL (per-device
    shard) scope, while the callee's block arguments and yielded values are at
    GLOBAL (full-tensor) scope. Each corresponding pair must differ by exactly
    the product of that dimension's declared partitioning-axis factor extents.

    Each partitioning_axes operand is one logical axis; the indexed shardings
    refer to this ordered list, exactly as in DistributedKernelOp. Every call
    site of one callee names the same axes, since the callee's own argument
    and result shardings are shared by all of them.

    A call is never part of a kernel, so any communication needed to reach the
    callee's argument shardings stays visible in the caller.
  }];
  let arguments = (ins FlatSymbolRefAttr:$callee,
                       Variadic<AnyType>:$arguments,
                       Variadic<FactorGroupType>:$partitioning_axes,
                       IndexedTensorShardingPerValueAttr:$argument_shardings,
                       IndexedTensorShardingPerValueAttr:$output_shardings,
                       OptionalAttr<DictArrayAttr>:$arg_attrs,
                       OptionalAttr<DictArrayAttr>:$res_attrs);
  let results = (outs Variadic<AnyType>:$returns);
  let hasVerifier = 1;
  let assemblyFormat = [{$callee custom<VariadicWithTypes>($arguments, type($arguments)) $argument_shardings `\n` ` ` ` ` `->` custom<ParenTypes>(type($returns)) $output_shardings `\n` ` ` ` ` `axes` custom<VariadicWithTypes>($partitioning_axes, type($partitioning_axes)) attr-dict}];
  let extraClassDeclaration = [{
    ::mlir::CallInterfaceCallable getCallableForCallee() {
      return (*this)->getAttrOfType<::mlir::SymbolRefAttr>("callee");
    }
    void setCalleeFromCallable(::mlir::CallInterfaceCallable callee) {
      (*this)->setAttr("callee", ::llvm::cast<::mlir::SymbolRefAttr>(callee));
    }
    ::mlir::Operation::operand_range getArgOperands() { return getArguments(); }
    ::mlir::MutableOperandRange getArgOperandsMutable() {
      return getArgumentsMutable();
    }
  }];
}

def DistributedKernelOp : DistributedOp<"DistributedKernel", [
  SingleBlock, AttrSizedOperandSegments,
  DeclareOpInterfaceMethods<PartitioningAnchorOpInterface>
]> {
  let description = [{Contains a computation subgraph that will never internally
    create collectives no matter how the partitioning axes are chosen. The results
    may require a reduction after, as indicated by the unreduced axes on the
    output sharding attributes. Each partitioning_axes operand represents a unique
    logical partitioning axis; indexed sharding metadata refers to this ordered,
    unique axis list. The cluster-distributed-kernels pass establishes this
    invariant by assigning one index to each logical AxisSymbol. Operations in the
    body carry distributed.argument_shardings and distributed.output_shardings
    attributes with the same index space.

    Scope: $arguments/$returns (this op's own operand/result lists) are at
    LOCAL (per-device shard) scope, while the region's own block arguments
    and yielded values are at GLOBAL (full-tensor) scope -- this op is
    itself the local/global boundary, matching PartitioningAnchorOpInterface's
    role elsewhere in this dialect. The two are not required to be identical
    in type; verify() checks the real relationship between them (a
    corresponding operand/block-arg or result/yield pair's shapes must
    differ by exactly the product of that dimension's currently-declared
    partitioning-axis factor extents, per argument_shardings/output_shardings)
    instead.}];
  let arguments = (ins Variadic<AnyType> : $arguments,
                   Variadic<FactorGroupType> : $partitioning_axes,
                   IndexedTensorShardingPerValueAttr : $argument_shardings,
                   IndexedTensorShardingPerValueAttr : $output_shardings);
  let results = (outs Variadic<AnyType> : $returns);
  let regions = (region AnyRegion:$body);
  let hasVerifier = 1;
  let assemblyFormat = [{custom<VariadicWithTypes>($arguments, type($arguments)) $argument_shardings `\n` ` ` ` ` `->` custom<ParenTypes>(type($returns)) $output_shardings `\n` ` ` ` ` `axes` custom<VariadicWithTypes>($partitioning_axes, type($partitioning_axes)) $body attr-dict}];
}



def DistributedYieldOp : DistributedOp<"DistributedYield", [Terminator]> {
  let description = [{
        Terminator for distributed functions and DistributedKernelOp bodies.
  }];
  let arguments = (ins Variadic<AnyType> : $returns);
  let assemblyFormat =
      "custom<VariadicWithTypes>($returns, type($returns)) attr-dict";
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

// Rule of thumb established across this dialect for how two ops/edges at
// possibly-different local/global scope should communicate that fact:
//  1. If an op already implements PartitioningAnchorOpInterface (owns a
//     partitioning_axes-indexable operand/result list -- DistributedKernelOp,
//     a Cast op, AnchorPartitioningOp), read/attach the binding there
//     directly. Never introduce a second, separately-attached copy.
//  2. If a real shape change is genuinely needed between two scopes, use
//     CastGlobalToLocal/CastLocalToGlobal.
//  3. If two ops/edges meet at the same scope/shape with no owning op to
//     hang a binding on, use AnchorPartitioning.
def DistributedCastGlobalToLocalOp : DistributedOp<
    "CastGlobalToLocal", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>,
                          DeclareOpInterfaceMethods<PartitioningAnchorOpInterface>]> {
  let description = [{Reinterprets one global tensor as its local shard:
    $input is at GLOBAL scope, $output is at LOCAL scope. The
    partitioning_axes operands are positional: operand i is the factor group
    partitioning tensor dimension i. The result type is inferred by dividing
    each static input dimension by the corresponding factor-group extent.

    Pure: safe because CSE only ever merges instances with literally
    identical operands (same input value, same partitioning_axes values in
    the same order), and axis propagation guarantees any two such instances
    always require the identical growth decision -- so in-place mutation of
    a result type by InlineDeviceLocalAxesPass (inlining a DeviceLocalAxis
    factor: grow this dimension, drop that factor from partitioning_axes --
    see that pass's own comment) never creates two structurally-identical
    instances with diverging meaning.}];
  let arguments = (ins
    AnyTensor:$input,
    Variadic<FactorGroupType>:$partitioning_axes
  );
  let results = (outs AnyTensor:$output);
  let assemblyFormat =
      "$input `axes` custom<VariadicWithTypes>($partitioning_axes, type($partitioning_axes)) attr-dict `:` type($input) `->` type($output)";
  let hasVerifier = 1;
  let hasCanonicalizer = 1;
}

def DistributedCastLocalToGlobalOp : DistributedOp<
    "CastLocalToGlobal", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>,
                          DeclareOpInterfaceMethods<PartitioningAnchorOpInterface>]> {
  let description = [{Reinterprets one local tensor shard as its global
    tensor: $input is at LOCAL scope, $output is at GLOBAL scope. The
    partitioning_axes operands are positional: operand i is the factor
    group partitioning tensor dimension i. The result type is inferred by
    multiplying each static input dimension by the corresponding factor-group
    extent.

    Pure: see DistributedCastGlobalToLocalOp's identical note.}];
  let arguments = (ins
    AnyTensor:$input,
    Variadic<FactorGroupType>:$partitioning_axes
  );
  let results = (outs AnyTensor:$output);
  let assemblyFormat =
      "$input `axes` custom<VariadicWithTypes>($partitioning_axes, type($partitioning_axes)) attr-dict `:` type($input) `->` type($output)";
  let hasVerifier = 1;
  let hasCanonicalizer = 1;
}

def AnchorPartitioningOp : DistributedOp<
    "AnchorPartitioning", [Pure, AllTypesMatch<["input", "output"]>,
                           DeclareOpInterfaceMethods<PartitioningAnchorOpInterface>]> {
  let description = [{Re-anchors a value's partitioning binding onto a new
    SSA edge with no shape change: $input and $output are always the SAME
    type (structurally enforced -- AllTypesMatch). This op does not itself
    distinguish local from global scope -- both sides are the identical
    scope; which scope that is comes from context (this value's own
    producer/consumers), not from this op. partitioning_axes here is pure
    metadata (which mesh/DeviceLocal factors this value's own dimensions
    would divide by if its never-materialized other-scope counterpart were
    ever needed) -- it never drives this op's own type the way it drives a
    Cast op's.

    Because $input/$output must always match, this op cannot grow itself
    unilaterally the way a Cast op grows its own result in place: growing
    one side without the other would violate AllTypesMatch, and growing
    $input directly isn't this op's call to make (it doesn't own that
    value). Growth here is always propagated: whatever pass grows $input's
    own defining op forward-propagates that change onto any
    AnchorPartitioningOp directly consuming it (setting its result type to
    match and dropping the same factor from its own partitioning_axes) --
    the same style of propagation InlineDeviceLocalAxes.cpp's
    applyCollectiveOutputType already does for DistributedAwait uses of a
    grown collective, extended to also cover this op.

    Exists for edges where two ops meet at the same local/global scope but
    neither one already implements PartitioningAnchorOpInterface to hang the
    binding on directly (a chained distributed.Collective's reused await
    result, feeding another collective's input_object with no real scope
    change in between, is the motivating case). Never introduce one where
    the producer or consumer already self-describes via
    PartitioningAnchorOpInterface -- read the binding from there instead.}];
  let arguments = (ins
    AnyTensor:$input,
    Variadic<FactorGroupType>:$partitioning_axes
  );
  let results = (outs AnyTensor:$output);
  let assemblyFormat =
      "$input `axes` custom<VariadicWithTypes>($partitioning_axes, type($partitioning_axes)) attr-dict `:` type($input)";
  let hasVerifier = 1;
  let hasCanonicalizer = 1;
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
      "type($output_mesh) `reduces` custom<VariadicWithTypes>($reduction_groups, type($reduction_groups)) `maps` $mapping `:` "
      "type($mapping) $reduction_bodies attr-dict";
  let hasVerifier = 1;
}

def DistributedManualComputationOp
    : DistributedOp<"ManualComputation", [SingleBlock]> {
  let description = [{This project's own stand-in for Shardy's
    sdy.manual_computation, which LowerKernels.cpp later translates it into:
    operands/results are at the enclosing kernel's declared shape, while the
    region's block arguments/yielded values are at the fully-divided-down
    local shape for whichever dimensions reference a manual axis.
    CanonicalizeShardedFactorOrderPass builds this op in place of a real
    transpose to fix a reshape whose dimension order doesn't fit neatly with
    shardy axis grammar.

    manual_axes indexes into the enclosing DistributedKernelOp's shared
    partitioning_axes list (this op has no partitioning_axes of its own),
    naming which slots the region computes with at already-divided-down local
    extent; none of their factors may be DeviceLocalAxisType/
    ReplicationAxisType, since those are never really divided by
    LowerKernels.cpp anyway.}];
  let arguments = (ins
    Variadic<AnyTensor>:$inputs,
    DenseI64ArrayAttr:$manual_axes,
    IndexedTensorShardingPerValueAttr:$argument_shardings,
    IndexedTensorShardingPerValueAttr:$output_shardings
  );
  let results = (outs Variadic<AnyTensor>:$outputs);
  let regions = (region AnyRegion:$body);
  let hasVerifier = 1;
  let assemblyFormat = [{custom<VariadicWithTypes>($inputs, type($inputs)) $argument_shardings `\n` ` ` ` ` `manual_axes` $manual_axes `\n` ` ` ` ` `->` custom<ParenTypes>(type($outputs)) $output_shardings $body attr-dict}];
}
#endif // ENZYME_AD_JAX_DIALECT_DISTRIBUTED_OPS_TD
