#ifndef ENZYME_AD_JAX_DISTRIBUTED_PASSES
#define ENZYME_AD_JAX_DISTRIBUTED_PASSES

include "mlir/Pass/PassBase.td"


def ConvertMainToDistributedFunctionPass
    : Pass<"convert-main-to-distributed-function", "::mlir::ModuleOp"> {
  let summary = "Convert the main func.func into distributed.function.";
  let options = [
    Option<"dumpValueAxes", "dump-value-axes", "bool", "false",
           "Print ShardyLogicalAxisAnalysis axis symbols for each SSA value "
           "in main, before conversion">,
    Option<"dumpOperationAxes", "dump-operation-axes", "bool", "false",
           "Print ShardyLogicalAxisAnalysis axis symbols for each operation "
           "in main, before conversion">,
  ];
  let dependentDialects = [
    "enzyme::axis::AxisDialect", "sdy::SdyDialect",
    "enzyme::distributed::DistributedDialect"
  ];
  let description = [{
    Expects a unique func.func @main with a single block. Uses the current
    Shardy logical-axis analysis to build distributed.function partitioning
    axes and indexed sharding metadata, then rewrites the function body to a
    distributed.function with a distributed.yield terminator. If main has
    already been converted, the pass leaves it unchanged.
  }];
}

def MaterializeDistributedCollectivesPass
    : Pass<"materialize-distributed-collectives", "::mlir::ModuleOp"> {
  let summary = "Replace Shardy reshards with distributed collectives.";
  let options = [
    Option<"dumpValueAxes", "dump-value-axes", "bool", "false",
           "Print ShardyLogicalAxisAnalysis axis symbols for each SSA value "
           "in main, before collective materialization">,
    Option<"dumpOperationAxes", "dump-operation-axes", "bool", "false",
           "Print ShardyLogicalAxisAnalysis axis symbols for each operation "
           "in main, before collective materialization">,
  ];
  let dependentDialects = [
    "enzyme::axis::AxisDialect", "sdy::SdyDialect",
    "enzyme::distributed::DistributedDialect"
  ];
  let description = [{
    Runs on the unique main function-like op, removes explicit sdy.reshard
    operations, then materializes distributed collectives for reduction and
    layout conflicts inferred from Shardy logical-axis analysis.
  }];
}

def ClusterDistributedKernelsPass
    : Pass<"cluster-distributed-kernels", "::mlir::ModuleOp"> {
  let summary = "Cluster main compute ops into distributed.kernel regions.";
  let options = [
    Option<"dumpLogicalAxes", "dump-logical-axes", "bool", "false",
           "Print ShardyLogicalAxisAnalysis axis symbols for each operation in main">,
  ];
  let dependentDialects = [
    "enzyme::axis::AxisDialect", "sdy::SdyDialect",
    "enzyme::distributed::DistributedDialect"
  ];
  let description = [{
    Runs on the unique main function-like op and outlines clusterable compute
    operations into distributed.kernel regions while preserving SSA/topological
    ordering constraints. This pass assumes conversion to distributed.function
    and distributed collective materialization have already run.
  }];
}

def StabilizeAxisOrderPass
    : Pass<"stabilize-axis-order", "::mlir::ModuleOp"> {
  let summary = "Canonicalize the textual placement of axis-algebra ops for "
                "determinism.";
  let dependentDialects = [
    "enzyme::axis::AxisDialect", "enzyme::distributed::DistributedDialect"
  ];
  let description = [{
    Sorts axis algebra ops into the order they appear as uses. This has no
    semantics meaning: mainly this pass is used for canonicalization / to 
    simplify testing of otherwise arbitrary axis orderings.

    Expects --canonicalize (or equivalent DCE) to have already run: any
    axis-algebra op left unreferenced by a real op afterward is unexpected
    and is reported with a remark before being erased.
  }];
}

def InsertPhysicalMeshPass : Pass<"insert-physical-mesh", "::mlir::ModuleOp"> {
  let summary = "Inserts a module-level distributed.PhysicalMesh and axes based on the passed option.";
  let dependentDialects =
      ["enzyme::distributed::DistributedDialect"];
  let options = [
    Option<"configurationString", "configuration-string", "std::string", "\"\"",
           "Raw distributed.PhysicalMesh op. Exactly one of configuration string or file must be specified.">,
    Option<"configurationFile", "configuration-file", "std::string", "\"\"",
           "Path to file containing a distributed.PhysicalMesh op. Exactly one of configuration string or file must be specified.">,
  ];
  let description = [{The configuration string / file is expected to contain a well-formed distributed.PhysicalMesh op.}];
}

def DistributedToHloPass : Pass<"distributed-to-hlo", "::mlir::ModuleOp"> {
  let summary = "Lower distributed dialect constructs to StableHLO.";
  let dependentDialects = [
    "enzyme::distributed::DistributedDialect", "stablehlo::StablehloDialect"
  ];
  let description = [{
        Skeleton pass for lowering the distributed dialect to StableHLO.
        This pass currently performs no rewrites.
  }];
}

def CanonicalizeAxisMapsPass
    : Pass<"canonicalize-axis-maps", "::mlir::ModuleOp"> {
  let summary = "Rewrite axis.map mappings into maximal indivisible form.";
  let dependentDialects = ["enzyme::axis::AxisDialect"];
  let description =
      [{Rewrites each axis.map pair using split_divisible,
        replacing mapping groups with maximal one - to - one split groups.}];
}

def CanonicalizeShardedFactorOrderPass
    : Pass<"canonicalize-sharded-factor-order", "::mlir::ModuleOp"> {
  let summary = "Rewrites tensors so sharded factors are always major and "
                "local factors always minor on every dimension.";
  let dependentDialects = [
    "enzyme::axis::AxisDialect", "enzyme::distributed::DistributedDialect",
    "sdy::SdyDialect", "stablehlo::StablehloDialect"
  ];
  let description = [{
    A tensor dimension composed of several independent factors (e.g. from a
    reshape merging previously-separate axes) can end up with a Sharded
    factor sandwiched between Local ones -- a pattern Shardy's
    DimensionShardingAttr cannot represent (it requires all sharded (sub-)axes
    to form a major-most prefix, with everything else an implicit minor-most
    remainder). This pass establishes that invariant everywhere: every op
    consumes and produces tensors with all sharded factors bubbled to the
    major position of their axis, on every dimension.

    Elementwise and associative-reduction ops need no rewrite: untouched
    dimensions inherit their input's canonical order for free, determined
    generically via Shardy's own per-op sharding rule rather than by
    hardcoding op identity. A distributed kernel's own block arguments and
    yielded results, and the cast ops bookending collectives
    (distributed.CastGlobalToLocal/CastLocalToGlobal), are true fixed points
    with no producer or consumer to defer the fix to, so they are rewritten
    directly: a clean split at the dimension's current boundary, a transpose
    into canonical order, and a merge back to the original shape. This is a
    cheap, local, per-device operation (the transpose only reorders which
    dimension index a factor occupies; it never changes which device holds
    which data), not a cross-device communication cost.
    distributed.Collective itself is never rewritten directly: its mesh
    operands are flat, mesh-space groupings rather than one-per-tensor-
    dimension, so it has no positional structure this pass can act on --
    whatever correctness is needed around a collective lives entirely in its
    bookending casts.

    LowerKernelsPass and InlineDeviceLocalAxesPass depend on this invariant
    already holding: they assume every DeviceLocalAxis is contiguous and
    minor-most on its dimension, and this pass is what is expected to
    establish that before they run.

    Current status: a reshape (or other op with a dimension mapped to more
    than one factor) whose operand or result is one of the fixed points above
    is handled as part of that fixed point's own rewrite. One fed by an
    arbitrary intermediate value elsewhere in a kernel body is not yet
    handled -- doing so correctly requires tracking that value's actual
    current factor order forward from the nearest already-fixed point,
    rather than trusting any op's own once-computed sharding attributes,
    which go stale as soon as an upstream fixed point is rewritten. Such ops
    are reported with a remark identifying them, and the pass fails, so the
    gap is visible rather than silently producing wrong results. Rewrite
    support is added incrementally as specific op patterns are actually hit,
    rather than guessed upfront.
  }];
}

def LowerKernelsPass : Pass<"distributed-lower-kernels", "::mlir::ModuleOp"> {
  let summary = "Applies sharding to distributed kernels to translate from global to sharded form.";
  let dependentDialects = [
    "enzyme::distributed::DistributedDialect", "sdy::SdyDialect",
    "mlir::func::FuncDialect"
  ];
  let options = [ 
    Option<"lowerLogicalAxes", "lower-logical-axes", "bool", "false", "Whether to lower sharding over logical axes. If false, logical axes will be left un-sharded. Default: false">,
    Option<"dumpShardyModules", "dump-shardy-modules", "bool", "false", "Print Shardy modules created for each distributed kernel">,
    Option<"dumpLoweredModules", "dump-lowered-modules", "bool", "false", "Print the per-kernel module after sharding has been localized (global-to-local conversion and Shardy lowering complete), for debugging">,
    Option<"verifyShardyLowering", "verify-shardy-lowering", "bool", "false", "Run the MLIR verifier between the internal Shardy lowering passes. Off by default: with maximal logical parallelism (one device id per tensor element), verification can be extremely slow. Default: false">,
  ];
  let description = [{
    Translates distributed kernels to their sharded form. By default, only shards over sharding factors on physical axes. Axes that have
    been sharded over are removed from the kernel's axis mapping. Axis factors succesfully sharded over are removed from
    the kernel's axis mapping.
  }];
}

def DropKernelBodyShardingAttrsPass
    : Pass<"drop-kernel-body-sharding-attrs", "::mlir::ModuleOp"> {
  let summary = "Drops the per-op distributed.argument_shardings/"
                "output_shardings attributes inside every kernel body once "
                "distributed-lower-kernels has consumed them.";
  let dependentDialects = ["enzyme::distributed::DistributedDialect"];
  let description = [{
    A DistributedKernelOp's body ops carry distributed.argument_shardings/
    output_shardings attributes indexing into that kernel's own
    partitioning_axes list (see DistributedKernelOp's doc comment).
    distributed-lower-kernels is their last real consumer; once it has run,
    they're stale bookkeeping that nothing downstream reads, and they become
    outright dangling (indices into a slot list a later pass, e.g.
    merge-adjacent-trivial-kernels, may shrink or empty) if left in place.
    This pass unconditionally removes them from every op nested in a kernel
    body, expected to run any time after distributed-lower-kernels.
  }];
}

def DropShardingRuleAttrsPass
    : Pass<"drop-sharding-rule-attrs", "::mlir::ModuleOp"> {
  let summary = "Drops stale sdy.sharding_rule attributes module-wide.";
  let dependentDialects = ["mlir::sdy::SdyDialect"];
  let description = [{
    Wraps mlir::sdy::removeShardingRules (the same logic behind Shardy's own
    sdy-drop-sharding-rules pass) but runs it directly on the whole module
    instead of nesting it under func::FuncOp: at this point in the pipeline
    the outer function is still a distributed.DistributedFunction, not a
    func.func, so the upstream, FuncOp-anchored pass would never find
    anything to run on. distributed-lower-kernels is the last real consumer
    of these attributes; once it's done they're stale.
  }];
}

def DropTrivialKernelShardingPass
    : Pass<"drop-trivial-kernel-sharding", "::mlir::ModuleOp"> {
  let summary = "Blanks a trivial kernel's own argument_shardings/"
                "output_shardings and empties its partitioning_axes.";
  let dependentDialects = ["enzyme::distributed::DistributedDialect"];
  let description = [{
    A kernel is trivial (see MergeAdjacentTrivialKernelsPass) once every
    operand/result's local-scope type already equals its body's
    block-argument/yielded-value type -- no partitioning axis divides its
    declared extent by anything but 1. Such a kernel's argument_shardings/
    output_shardings and partitioning_axes are then pure, unused bookkeeping
    (every dim_partitioning_axes entry could only ever resolve to an
    extent-1 slice), so this pass blanks them: empty dim_partitioning_axes
    per operand/result, and an empty partitioning_axes list. This is what
    lets the axis-algebra chain feeding those slots become dead code for a
    later cse/canonicalize to remove, for every trivial kernel -- not just
    ones merge-adjacent-trivial-kernels happens to fuse. Expected to run any
    time after distributed-lower-kernels, which is what establishes
    triviality in the first place.
  }];
}

def DropIdentityPartitioningAnchorsPass
    : Pass<"drop-identity-partitioning-anchors", "::mlir::ModuleOp"> {
  let summary = "Unconditionally drops partitioning-anchor metadata that is "
                "no longer needed once lowering has completed.";
  let dependentDialects = [
    "enzyme::axis::AxisDialect", "enzyme::distributed::DistributedDialect"
  ];
  let description = [{
    distributed.AnchorPartitioning's input/output types are always
    structurally identical, so every instance is a semantic no-op: it exists
    only to keep an axis-to-tensor-axis binding discoverable for earlier
    passes that consume PartitioningAnchorOpInterface (chiefly
    InlineDeviceLocalAxesPass). Once distributed-lower-kernels has run, that
    metadata has already been fully consumed and no further growth will
    occur, so every AnchorPartitioning op can simply be dropped in favor of
    its own input, and any Cast op whose input/output types happen to
    already be equal can be folded the same way. This pass is expected to
    run late in the pipeline (after distributed-lower-kernels), never
    earlier -- running it before InlineDeviceLocalAxesPass would incorrectly
    discard bindings that pass still needs.
  }];
}

def MergeAdjacentTrivialKernelsPass
    : Pass<"merge-adjacent-trivial-kernels", "::mlir::ModuleOp"> {
  let summary = "Merges physically-adjacent distributed.kernel ops that no "
                "longer carry any real sharding.";
  let dependentDialects = [
    "enzyme::axis::AxisDialect", "enzyme::distributed::DistributedDialect"
  ];
  let description = [{
    A distributed.kernel is "trivial" once every operand/result's local-scope
    type equals its body block-argument/yielded-value's global-scope type --
    i.e. distributed-lower-kernels has already divided every partitioning
    axis down to an extent of 1, so no real sharding is left for that kernel
    to describe. Two trivial kernels that are immediate block neighbors are
    pure serial computation split across an unnecessary op boundary, and are
    merged into a single kernel that returns everything either constituent
    kernel returned. Any direct SSA dependency between the pair (the second
    kernel consuming one of the first's results) is rewired to the shared
    internal value instead of round-tripping through the now-removed kernel
    boundary. Chains of more than two adjacent trivial kernels collapse to
    one op in a single run, since each merge produces a new kernel that is
    itself revisited. Leaves any now-unused kernel results in place --
    dropping those is a separate pass.

    A CastGlobalToLocal/CastLocalToGlobal is often left directly between two
    otherwise-mergeable kernels purely as an artifact of emission order, not
    a real dependency: a CastGlobalToLocal reading straight from a block
    argument doesn't depend on anything computed earlier in the block, and a
    CastLocalToGlobal whose only use is the block's terminator has nothing
    downstream depending on where it sits. This pass also sweeps such casts
    up to just past the last thing they actually depend on, or down to just
    before the terminator, respectively, so the merge above only has to
    reason about strict physical adjacency.
  }];
}

def LowerKernelsToExecutablePass
    : Pass<"distributed-lower-kernels-to-executable", "::mlir::ModuleOp"> {
  let summary = "Placeholder: will dispatch each local-size distributed.kernel "
                "body to an external compiler for executable lowering.";
  let dependentDialects = [
    "enzyme::distributed::DistributedDialect", "mlir::func::FuncDialect"
  ];
  let options = [
    Option<"dumpKernelModulesTo", "dump-kernel-modules-to", "std::string", "\"\"",
           "If non-empty, write each kernel's body as a standalone module "
           "(one func.func mirroring the kernel's local-scope ABI) under "
           "this directory, one kernel-<random>.mlir file per kernel -- the "
           "wire format the external compiler will eventually be dispatched "
           "on, for inspecting before real dispatch is wired up. Randomly "
           "suffixed rather than indexed since this pass runs once per "
           "search candidate, sharing one directory across many runs; "
           "nothing here needs to map a dump back to the candidate or "
           "kernel that produced it.">
  ];
  let description = [{
    By the time this pass runs, distributed-lower-kernels has already
    divided every kernel's operands/results down to local (per-device)
    sizes -- that pass is about sharding, not code generation. What's left
    inside each kernel body is a plain, unsharded HLO subgraph. This pass is
    a placeholder for the next lowering stage: handing each such subgraph to
    an external compiler for actual executable lowering. Compilation there
    is per-kernel, so this pass's job is the walk-and-dispatch boilerplate;
    for now it only performs that walk, with dump-kernel-modules-to emitting
    the standalone module each kernel would be dispatched as, since real
    dispatch isn't wired up yet.
  }];
}

def DropIdentityCollectivesPass
    : Pass<"drop-identity-collectives", "::mlir::ModuleOp"> {
  let summary = "Drops distributed.Collective ops that perform no real "
                "communication.";
  let dependentDialects = [
    "enzyme::axis::AxisDialect", "enzyme::distributed::DistributedDialect"
  ];
  let description = [{
    Runs canonicalize-axis-maps as an internal sub-pass to put every
    distributed.Collective's mapping into maximal one-to-one factor form,
    then drops any collective that does no reduction, maps its input mesh
    to a structurally identical output mesh with a fully identity mapping,
    and produces an output type identical to its own input -- i.e. one that
    performs no real communication. Each such collective's
    distributed.Await uses are replaced directly by its own input_object.
  }];
}

def InlineDeviceLocalAxesPass
    : Pass<"inline-device-local-axes", "::mlir::ModuleOp"> {
  let summary = "Inline distributed.DeviceLocalAxis markers into concrete local structure.";
  let dependentDialects = [
    "enzyme::axis::AxisDialect", "enzyme::distributed::DistributedDialect"
  ];
  let description = [{
    Materializes distributed.DeviceLocalAxis markers (axis-algebra factors the
    search process, SearchStrategies.cpp, decided to serialize entirely
    within a device rather than shard across the mesh) into concrete local
    structure: growing the local side of casts and a kernel's own external
    result types by the corresponding real tensor extent, and dropping the
    now-redundant factors from a collective's own mesh/reduction/mapping
    bookkeeping. See InlineDeviceLocalAxes.cpp's own doc comment for the full
    per-op breakdown.
  }];
}

def DistributedSearchStrategiesPass : Pass<"distributed-search-strategies", "::mlir::ModuleOp"> {
  let summary = "Searches for distribution strategies mapping logical axes to physical execution";
  // Every dialect any pass in buildDistributedSearchLoweringPipeline
  // (SearchStrategies.cpp's cloneAndApplyDecisions runs it in a nested
  // PassManager once per scored candidate) could ever need must be loaded
  // here, not left to that nested run's own dependent-dialect preloading:
  // this pass's runOnOperation() is already inside the outer PassManager's
  // multi-threaded execution context by the time it starts, and MLIR
  // disallows loading a not-yet-loaded dialect at that point (it either
  // hard-errors or deadlocks contending on the same thread pool).
  let dependentDialects = [
    "enzyme::distributed::DistributedDialect", "enzyme::axis::AxisDialect",
    "sdy::SdyDialect", "stablehlo::StablehloDialect", "mlir::func::FuncDialect"
  ];
  let options = [
    Option<"dumpCandidates", "dump-candidates", "bool", "false", "Print the cloned, partially-rewritten IR for each scored search candidate, for debugging. Default: false">,
    Option<"dumpFinalized", "dump-finalized", "bool", "false", "Print the cloned, partially-rewritten IR for each finalized search candidate, for debugging. Default: false">,
    Option<"dumpBest", "dump-best", "bool", "false", "Print the cloned, partially-rewritten IR for the best search candidate, for debugging. Default: false">,
    Option<"disableVerifier", "disable-verifier", "bool", "true", "Disable the MLIR verifier when running the internal lowering pipeline on each candidate clone during search, for runtime reasons. Default: true">,
    Option<"beamSize", "beam-size", "int", "100", "Maximum number of candidates retained per generation of the beam search. Default: 100">,
    Option<"logProgress", "log-progress", "bool", "false", "Log the deepest and shallowest axis-decision depth reached across the beam at each generation turnover, out of the total axis count. Useful for tracking how far a long-running search has gotten. Default: false">
  ];
}

#endif // ENZYME_AD_JAX_DISTRIBUTED_PASSES
