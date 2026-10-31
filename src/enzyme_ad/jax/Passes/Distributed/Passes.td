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

def DistributedVerifyKernelsPass
    : Pass<"distributed-verify-kernels", "::mlir::ModuleOp"> {
  let summary = "Explicitly run the verifiers of distributed kernel, function and call ops.";
  let description = [{
    Fails with the op's own diagnostic when any DistributedKernelOp,
    DistributedFunctionOp or DistributedCallOp is invalid (e.g. overlapping
    partitioning-axis slots, or a tensor sharded twice along one axis), even
    when the pass manager's per-pass verifier is disabled.
  }];
  let dependentDialects = ["enzyme::distributed::DistributedDialect"];
}

def ValidateShardyBaselineMeshPass
    : Pass<"validate-shardy-baseline-mesh", "::mlir::ModuleOp"> {
  let summary = "Check that every sdy.mesh projects onto the distributed.PhysicalMesh.";
  let description = [{
    Requires exactly one distributed.PhysicalMesh. Every sdy.mesh must use the
    default (iota) device order, have the physical mesh's device count, and each
    of its axes must be a contiguous device-id window over the physical axes'
    factorization. On success the module is stamped with
    `distributed.shardy_baseline`, which makes the Shardy logical axis analysis
    bind Shardy-sharded axes to physical factors.
  }];
  let dependentDialects =
      ["enzyme::distributed::DistributedDialect", "sdy::SdyDialect"];
}

def SetDistributedLoggingPass
    : Pass<"set-distributed-logging", "::mlir::ModuleOp"> {
  let summary = "Mints a per-run logging directory and stamps its location "
                "as discardable attributes on the module.";
  let options = [
    Option<"root", "root", "std::string", "\"\"",
           "Master logging directory. Empty (the default) turns logging off "
           "entirely: this pass becomes a no-op, creating no directory and "
           "stamping no attributes. Non-empty mints "
           "<root>/<module-name-or-\"run\">-<date>-<random suffix>/ (with "
           "empty Main/ and Candidates/ subdirectories) and stamps its "
           "absolute path as distributed.logging_root on the module.">,
    Option<"moduleName", "module-name", "std::string", "\"\"",
           "Optional name folded into the run directory and stamped as "
           "distributed.logging_module_name; omitted from both when "
           "empty (the run directory then falls back to a plain \"run\" "
           "prefix).">,
    Option<"commit", "commit", "std::string", "\"\"",
           "Optional commit/version string, stamped as "
           "distributed.logging_commit when non-empty.">,
  ];
  let description = [{
    A standalone config pass, meant to run once, early, ahead of
    distributed-search-strategies (or a bare distributed-search-lowering-
    pipeline invocation re-finalizing an already-searched winner). Later
    passes read distributed.logging_root (and the optional module-name/
    commit attributes) off their own module, so no logging option has to be
    threaded through intervening passes. Cloning a module, e.g. one search
    candidate's disposable clone, carries these attributes along.
  }];
}

def DumpDistributedLoggingCheckpointPass
    : Pass<"dump-distributed-logging-checkpoint", "::mlir::ModuleOp"> {
  let summary = "Dumps the module to Candidates/<uid>/ir/<checkpoint-name>"
                ".mlir when per-candidate logging is active.";
  let dependentDialects = ["enzyme::distributed::DistributedDialect"];
  let options = [
    Option<"checkpointName", "checkpoint-name", "std::string", "\"\"",
           "This instance's checkpoint filename (e.g. "
           "\"00-after-decisions\"), written as <checkpoint-name>.mlir.">,
    Option<"countCollectivesAs", "count-collectives-as", "std::string", "\"\"",
           "Attribute name to stamp the module's current collective count "
           "under: abstract distributed.Collective ops plus the stablehlo "
           "collective primitives they lower to (e.g. "
           "distributed.logging_num_collectives_before), for a later reader "
           "(StrategyScorer) to pick back up. Empty (the default) skips "
           "counting; only the 02-before-collectives and 03-final instances "
           "pass a name.">,
  ];
  let description = [{
    Reads distributed.logging_root (set by --set-distributed-logging) and
    distributed.logging_candidate_uid (set by cloneAndApplyDecisions) off its
    own module and, when both are present, writes the module to
    <logging_root>/Candidates/<uid>/ir/<checkpoint-name>.mlir. Does nothing
    (no directory, no file, no error) when either is absent. That is the
    normal case for a non-logging run, and for the winner-finalization run
    that applies a search's winning decisions directly onto the original
    module rather than through cloneAndApplyDecisions. The winner's
    checkpoints were already logged while it was a scoring clone, so
    re-finalizing it must not produce a second log entry.

    One instance exists per checkpoint. cloneAndApplyDecisions inserts
    00-after-decisions ahead of the lowering pipeline, and
    buildDistributedSearchLoweringPipeline inserts 01-before-act,
    02-before-collectives and 03-final. Each instance self-gates as above, so
    the only per-instance options are the checkpoint name and optional
    collective counting.
  }];
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
      [{Rewrites each axis.map pair using splitDivisibleMappings,
        replacing mapping groups with maximal one - to - one split groups.}];
}

def RefinePartitioningSlotsPass
    : Pass<"refine-partitioning-slots", "::mlir::ModuleOp"> {
  let summary = "Splits kernel partitioning-axis slots so each is purely "
                "sharded or purely local.";
  let dependentDialects = [
    "enzyme::axis::AxisDialect", "enzyme::distributed::DistributedDialect"
  ];
  let description = [{
    A partitioning-axis slot of a distributed.DistributedKernel is a space
    (an axis.product) that tensor dimensions select by index. A slot whose
    factors mix sharded and local kinds cannot be reordered relative to
    other slots without cutting through it, so this pass replaces each such
    slot by one slot per maximal run of same-kind factors and rewrites every
    reference (the kernel's own shardings and the sharding attributes on body
    ops) to the run of new slots. The replacement products are new values;
    the original axis.product is never modified, since other users may share
    it. Fully local and empty slots are kept.

    Later passes (canonicalize-sharded-factor-order) only reorder slot
    indices and never create slots, which requires this pass to have run
    after every factor's kind is decided.
  }];
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

def SinkKernelBoundaryValuesPass
    : Pass<"sink-kernel-boundary-values", "::mlir::ModuleOp"> {
  let summary = "Rebuilds a constant or arg-only pure computation inside "
                "every consuming kernel's own body, instead of threading it "
                "across a kernel boundary as an ordinary operand.";
  let dependentDialects = ["enzyme::distributed::DistributedDialect"];
  let description = [{
    ACT matches a kernel body against a closed subgraph: every value the
    body reads must be either a genuine data-dependent argument or a
    constant/computation the body materializes itself. Clustering instead
    assigns a function-level constant, or a pure computation reachable only
    from a kernel's own arguments (e.g. a transpose of a weight argument),
    to whichever DistributedKernelOp first needs it; a later kernel that
    also needs the same value receives it as an ordinary cross-kernel
    operand instead of recomputing it, which looks to ACT like an
    unrecognized external input.

    For each DistributedKernelOp operand whose defining op is another
    DistributedKernelOp's result, this pass walks the corresponding yielded
    value's producer chain within that kernel's body. If every op on the
    chain is a constant or free of memory effects, and every chain leaf that
    is a block argument corresponds to a value already present, by SSA
    identity, among the consuming kernel's own operands (so nothing sunk
    ever needs a new cross-kernel edge), the whole chain is cloned into the
    consuming kernel's body and its block argument's uses are redirected to
    the clone. The producer's result and the consumer's now-unused
    operand are left in place for the pruning/dedup canonicalizations on
    DistributedKernelOp to remove.

    Expected to run after distributed-merge-adjacent-trivial-kernels, so it
    only has to handle producer/consumer pairs merging did not already fuse
    into one body, and before the canonicalizer that precedes ACT dispatch,
    so the dead result/operand pairs it creates are cleaned up immediately.
  }];
}

def InlineKernelsPass : Pass<"distributed-inline-kernels", "::mlir::ModuleOp"> {
  let summary = "Dissolves every distributed.DistributedKernel wrapper by "
                "inlining its body into the surrounding block.";
  let dependentDialects = [
    "enzyme::distributed::DistributedDialect"
  ];
  let description = [{
    distributed-lower-kernels replaces a kernel's interior with locally-
    lowered ops but leaves the distributed.DistributedKernel wrapper op
    itself in place. This pass closes that gap: it splices each kernel
    body's operations into its parent block in the kernel's place, remaps
    block arguments to the kernel's own operands, replaces uses of the
    kernel's results with the values yielded by its terminator, and erases
    the now-empty kernel op.

    Runs after distributed-lower-kernels has fully localized every kernel
    body, for every candidate reaching schedule-graph-building (not only the
    search's eventual winner): ScheduleGraphAnalysis needs the dissolved,
    bare-op shape this pass produces to see real ops with real SSA/memref
    dependencies in place of an opaque distributed.DistributedKernel wrapper
    (see ScheduleGraph.h), whether it's building the winner's finalized
    schedule or scoring a still-disposable clone.

    distributed.DistributedCall / distributed.DistributedFunction are a
    function-call boundary and are left untouched here. A kernel inside a
    called function's body is still reached and inlined, since this pass
    walks the whole module rather than just the entry function.
  }];
}

def LowerCollectivesPass
    : Pass<"distributed-lower-collectives", "::mlir::ModuleOp"> {
  let summary = "Replaces every atomic distributed.Collective/Await pair "
                "with the real StableHLO async op chain its decomposition "
                "plan calls for.";
  let dependentDialects = [
    "enzyme::distributed::DistributedDialect", "stablehlo::StablehloDialect"
  ];
  let description = [{
    A distributed.Collective's PrimitiveStep chain (CollectiveDecomposer.h's
    planCollective) is a pure-data decomposition plan with no MLIR ops of its
    own. This pass turns a chosen chain into a real, async-wrapped StableHLO
    op chain, each async_start tagged with a "distributed.primitive_cost"
    attribute carrying that step's cost fields so a downstream scheduler can
    read a step's predicted cost straight off the IR instead of recomputing
    it via a second planCollective call. See CollectivePrimitiveBuilders.h
    for the exact op chain and attribute shape.

    Runs for every candidate reaching schedule-graph-building (not only the
    search's eventual winner), mirroring distributed-inline-kernels:
    ScheduleGraphAnalysis costs a materialized StableHLO async op chain, not
    an abstract distributed.Collective/Await pair, so every candidate needs
    this pass's output before it can be scored the same way the eventual
    winner is. Mesh/device cost parameters come from the unique physical
    mesh's optional metadata, else abstract defaults (MeshCostMetadata.h),
    matching every other consumer of planCollective. Replanning a collective
    independently per invocation, with no state shared across candidates,
    is what makes running this pass on every disposable scoring clone safe.
    An unplannable collective fails the pass with a diagnostic rather than
    being skipped, since a candidate that lowered this far is expected to
    have every collective plannable.
  }];
}

def ReorderSchedulePass
    : Pass<"distributed-reorder-schedule", "::mlir::ModuleOp"> {
  let summary = "Physically reorders each block's operations to match the "
                "admission order the event-DAG simulator predicts.";
  let dependentDialects = [
    "enzyme::distributed::DistributedDialect", "stablehlo::StablehloDialect"
  ];
  let description = [{
    Requires the module to already be fully materialized by
    distributed-inline-kernels and distributed-lower-collectives: it re-runs
    ScheduleGraph.h's event-DAG builder and ScheduleSimulator.h's timeline
    simulator over the real ops those passes produced, then reorders each
    block so its event-bearing ops appear in simulated admission order (the
    order events leave `ready` and start, not the order they finish).

    Sorting by admission order is topologically valid by construction: the
    simulator only admits an event once every predecessor has finished, so a
    predecessor is always admitted no later than its successors. Ops with no
    ScheduleEvent of their own (axis-algebra/mesh metadata, terminators,
    stablehlo.async_done) travel with the preceding event-bearing op instead
    of being reordered independently. A distributed.DistributedCall's callee
    body is reordered too, recursively, via its own fresh simulation.

    Meant to run once, winner-only, on the single search candidate the
    search ultimately picks.
  }];
}

def HoistKernelPartitionIdPass
    : Pass<"hoist-kernel-partition-id", "::mlir::ModuleOp"> {
  let summary = "Hoists stablehlo.partition_id out of DistributedKernelOp "
                "bodies into a new scalar operand/block argument.";
  let dependentDialects = [
    "enzyme::distributed::DistributedDialect", "stablehlo::StablehloDialect"
  ];
  let description = [{
    stablehlo.partition_id can end up inside a kernel body as a side effect
    of Shardy's own local lowering (distributed-lower-kernels), not as
    something this dialect introduces; see that pass for where it comes
    from. Left in place, it reaches distributed-lower-kernels-to-executable's
    ACT dispatch, whose cost model has no notion of the op at all.

    For every kernel with one or more occurrences directly in its body (there
    may be several, always deterministic and identical), this pass builds one
    stablehlo.partition_id in the kernel's parent scope, appends it as a new
    trailing operand with a matching block argument and (dimensionless, since
    the op is a rank-0 scalar) argument_shardings entry, rewrites every
    in-body occurrence to that block argument, and erases the now-dead
    op(s). A kernel with no such op is left untouched.

    Expected to run after distributed-lower-kernels (where the op is
    introduced) and before distributed-lower-kernels-to-executable (where it
    would otherwise reach ACT).
  }];
}

def LowerKernelsToExecutablePass
    : Pass<"distributed-lower-kernels-to-executable", "::mlir::ModuleOp"> {
  let summary = "Dispatches each local-size distributed.kernel body to the "
                "ACT external compiler for executable lowering.";
  let dependentDialects = [
    "enzyme::distributed::DistributedDialect", "mlir::func::FuncDialect",
    // act-patched-module-file parses an arbitrary external module, which may
    // reference any dialect the real ACT output does (e.g. memref, for the
    // alloc/subview/dealloc around each launch) -- loading a dialect lazily
    // mid-pass isn't allowed under the PassManager's multithreading, so it
    // must already be registered here.
    "mlir::memref::MemRefDialect"
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
           "kernel that produced it.">,
    Option<"actPatchedModuleFile", "act-patched-module-file", "std::string",
           "\"\"",
           "Test-only seam for the splice half of executable lowering, "
           "skipping real dispatch: if non-empty, parse this single literal "
           "patched-module .mlir file (the external compiler's own output "
           "shape; see spliceKernelBodyModule) and splice it into every "
           "kernel this pass walks, in place of dispatching. Only useful "
           "for a module with a single kernel, since there is only one "
           "file to feed for every kernel in the walk. Mutually exclusive "
           "with act-target/act-worker-dir-root/act-worker-script-dir "
           "below: setting both this and real dispatch is a configuration "
           "error, not an ambiguity to silently resolve.">,
    Option<"actTarget", "act-target", "std::string", "\"\"",
           "Accelerator target name (e.g. \"eva2\", \"eva2-fine\", "
           "\"eva2-coarse\") to dispatch each kernel body to via a "
           "persistent ACT worker process, forwarded verbatim as "
           "ActWorkerConfig::target. Real dispatch only activates once "
           "this, act-worker-dir-root, and act-worker-script-dir are all "
           "non-empty together (mirrors act-patched-module-file's "
           "empty-string-means-off convention); setting only some of the "
           "three is a configuration error. Real dispatch also requires the "
           "ACT_REPO_PATH environment variable (see ActWorkerClient.h).">,
    Option<"actWorkerDirRoot", "act-worker-dir-root", "std::string", "\"\"",
           "Root directory under which a per-(target, calling user) work "
           "directory (requests/, responses/, cache/, and the container "
           "lifecycle lockfile) is created, forwarded verbatim as "
           "ActWorkerConfig::workdirRoot. See act-target for when real "
           "dispatch actually activates.">,
    Option<"actWorkerScriptDir", "act-worker-script-dir", "std::string",
           "\"\"",
           "Directory containing act_worker.py, bind-mounted read-only "
           "into the worker container, forwarded verbatim as "
           "ActWorkerConfig::workerScriptDir. See act-target for when real "
           "dispatch actually activates.">
  ];
  let description = [{
    By the time this pass runs, distributed-lower-kernels has already
    divided every kernel's operands/results down to local (per-device)
    sizes, since that pass is about sharding, not code generation. What's
    left inside each kernel body is a plain, unsharded HLO subgraph, ready
    to hand to ACT for actual executable lowering and costing, per-kernel.

    With act-target/act-worker-dir-root/act-worker-script-dir all set, each
    kernel this pass walks is dispatched for real, via
    ensureActPatchedKernel's content-hash cache and persistent worker
    (Dialect/Distributed/ActWorkerClient.h): a hit or a successful dispatch
    splices ACT's patched body (with its act.cost attributes) into the
    kernel in place. dump-kernel-modules-to additionally emits the
    standalone module each kernel is dispatched as, and
    act-patched-module-file exercises the splice-back-in half against a
    literal file instead of a real dispatch, for tests that shouldn't
    depend on Docker.
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
    "sdy::SdyDialect", "stablehlo::StablehloDialect", "mlir::func::FuncDialect",
    // Needed transitively by LowerKernelsToExecutablePass, which every
    // candidate's nested pipeline runs (see the comment above).
    "mlir::memref::MemRefDialect"
  ];
  let options = [
    Option<"dumpCandidates", "dump-candidates", "bool", "false", "Print the cloned, partially-rewritten IR for each scored search candidate, for debugging. Default: false">,
    Option<"dumpKernelModulesTo", "dump-kernel-modules-to", "std::string", "\"\"", "If non-empty, write each scored candidate's kernel bodies as standalone modules under this directory (see LowerKernelsToExecutablePass's own option of the same name, which this threads down into every candidate's scoring pipeline, not just the final winner's). Useful for measuring how many distinct kernel shapes a real search actually visits, independent of any external-compiler dispatch. Default: \"\"">,
    Option<"actTarget", "act-target", "std::string", "\"\"", "Accelerator target name (e.g. \"eva2\", \"eva2-fine\", \"eva2-coarse\") forwarded to every scored candidate's own distributed-lower-kernels-to-executable invocation (see LowerKernelsToExecutablePass's option of the same name), so real ACT dispatch (backed by the same content-hash cache and persistent worker as every other use of these options) is what search scoring itself uses to pick a winner, not a roofline stand-in. Real dispatch only activates once this, act-worker-dir-root, and act-worker-script-dir are all non-empty together; setting only some of the three is a configuration error, checked once up front before the search loop starts (not per candidate), so a misconfiguration surfaces as one clear diagnostic instead of every candidate silently scoring -infinity. Real dispatch also requires the ACT_REPO_PATH environment variable (see ActWorkerClient.h), checked at the same point. Default: \"\"">,
    Option<"actWorkerDirRoot", "act-worker-dir-root", "std::string", "\"\"", "Root directory under which the per-(target, calling user) ACT worker's work directory (requests/, responses/, cache/, lockfile) is created, forwarded verbatim to every candidate's distributed-lower-kernels-to-executable invocation. See act-target for when real dispatch actually activates. Default: \"\"">,
    Option<"actWorkerScriptDir", "act-worker-script-dir", "std::string", "\"\"", "Directory containing act_worker.py, bind-mounted read-only into the worker container, forwarded verbatim to every candidate's distributed-lower-kernels-to-executable invocation. See act-target for when real dispatch actually activates. Default: \"\"">,
    Option<"actPatchedModuleFile", "act-patched-module-file", "std::string", "\"\"", "Forwarded to every scored candidate's own distributed-lower-kernels-to-executable invocation (see LowerKernelsToExecutablePass's option of the same name): a test-only seam that splices a literal ACT-patched module into every kernel instead of dispatching for real, so a test can exercise an ACT-dispatched candidate's scoring/logging without a real ACT/Docker round trip. Mutually exclusive with act-target/act-worker-dir-root/act-worker-script-dir, same as on that pass. Default: \"\"">,
    Option<"dumpFinalized", "dump-finalized", "bool", "false", "Print the cloned, partially-rewritten IR for each finalized search candidate, for debugging. Default: false">,
    Option<"dumpBest", "dump-best", "bool", "false", "Print the cloned, partially-rewritten IR for the best search candidate, for debugging. Default: false">,
    Option<"disableVerifier", "disable-verifier", "bool", "true", "Disable the MLIR verifier when running the internal lowering pipeline on each candidate clone during search, for runtime reasons. Default: true">,
    Option<"beamSize", "beam-size", "int", "100", "Maximum number of candidates retained per generation of the beam search. Default: 100">,
    Option<"logProgress", "log-progress", "bool", "false", "Log the deepest and shallowest axis-decision depth reached across the beam at each generation turnover, out of the total axis count. Useful for tracking how far a long-running search has gotten. Default: false">,
    Option<"baselineModule", "baseline-module", "std::string", "\"\"", "If non-empty, path to an MLIR file holding the Shardy expert plan, produced by shardy-to-distributed-pipeline with preserve-shardy-sharding=true (and the same physical mesh as this module). It is lowered and scored through the same pipeline as every search candidate, outside the beam search, and its result is logged under Baseline/ (never Candidates/) when logging is active. It does not affect which candidate wins. A baseline that fails to lower or cannot be scored fails this pass. Default: \"\"">,
    Option<"failOnBadCandidate", "fail-on-bad-candidate", "bool", "false", "Fail this pass outright the first time any scored candidate's lowering pipeline fails, instead of silently scoring it -infinity and letting the search route around it. Normal operation should leave this off, since a real search is expected to explore and discard plenty of infeasible candidates; turn it on for tests/debugging that want a lowering bug to surface immediately (with the offending candidate's IR printed) rather than being masked by the search finding a different, working candidate elsewhere. Default: false">,
    Option<"searchStrategy", "search-strategy", "std::string", "\"beam\"", "Which search to run: 'beam', 'random', 'mcts' or 'twostage'. search-samples is the evaluation budget for the last three. Default: beam">,
    Option<"searchSamples", "search-samples", "unsigned", "1000", "Evaluation budget for search-strategy=random, mcts and twostage. For mcts this is an iteration budget: an iteration that revisits an already-scored node reuses its score instead of evaluating again, so the number of real lowering-pipeline invocations (Main/info.json's num_candidates_scored) can come in under this value. Default: 1000">,
    Option<"searchSeed", "search-seed", "unsigned", "0", "Random seed for stochastic strategies, so a run is reproducible from its command line. Default: 0">,
    Option<"uctC", "uct-c", "double", "0.707", "MCTS exploration constant, applied to rewards normalised to [0,1]. Larger values explore more; 0 is pure exploitation. Default: 0.707">,
    Option<"uctBackup", "uct-backup", "std::string", "\"mean\"", "MCTS value backup: 'mean' for standard UCT, or 'max' for single-player MCTS. Default: mean">,
    Option<"mctsGreedyCompleteAfter", "mcts-greedy-complete-after", "int", "-1", "For search-strategy=mcts: number of uniformly random steps a rollout takes from the selected node before the remaining decisions are completed greedily (largest available chunk per axis, in the search's axis order). -1 keeps rollouts fully random until finalized; 0 makes every rollout fully greedy from the selected node. Values below -1 are rejected. Default: -1">,
    Option<"maxDecisionExtent", "max-decision-extent", "int", "8", "Upper bound on the chunk one decision takes from a free physical factor. The chunk is built from the prime factors of gcd(remaining extent, factor extent): start at the smallest prime (always taken, even if larger than this bound) and keep multiplying in the next prime while the product stays <= this bound. With 8: 8 -> 8, 16 -> 8, 12 -> 4, 9 -> 3, 11 -> 11, 30 -> 6. One child per free factor, so no branching on chunk size. Must be >= 2; 2 takes one factor of 2 (or the smallest prime) at a time, which equals the old primes-only behavior for power-of-2 extents. The greedy completer uses the same rule. Default: 8">,
    Option<"surrogateTrain", "surrogate-train", "unsigned", "256", "How much of the evaluation budget to spend fitting the surrogate, for search-strategy=twostage and for surrogate-prior. Default: 256">,
    Option<"surrogatePool", "surrogate-pool", "unsigned", "0", "For search-strategy=twostage: how many candidates to rank with the surrogate before exactly evaluating the best K. Ranking needs no IR lowering, so the pool should exceed K. 0 means 10x K. Default: 0">,
    Option<"surrogatePrior", "surrogate-prior", "double", "0.0", "Weight of a surrogate prior on MCTS's UCB score, PUCT-style. 0 leaves MCTS unchanged. Default: 0.0">,
    Option<"surrogateDump", "surrogate-dump", "std::string", "\"\"", "Write the fitted surrogate weights and their training data to this path, so the fit can be checked against another solver. Default: empty">,
    Option<"surrogateTarget", "surrogate-target", "std::string", "\"log\"", "What the surrogate regresses on: 'log', which keeps every prediction positive, or 'linear'. Default: log">
  ];
}

def MakeReplicationsExplicitPass
    : Pass<"distributed-make-replications-explicit", "::mlir::ModuleOp"> {
  let summary = "Makes every distributed.Collective's mesh bookkeeping "
                "explicitly cover every physical mesh axis.";
  let dependentDialects = [
    "enzyme::axis::AxisDialect", "enzyme::distributed::DistributedDialect"
  ];
  let description = [{
    A distributed.Collective's input_mesh/output_mesh/mapping are only
    required to account for whichever physical axes they actually mention --
    a physical axis absent from input_mesh entirely is well-defined (the
    value is uniformly replicated across it) but that meaning is only
    recoverable by reasoning about the whole module's PhysicalMesh from
    outside the collective op itself. This pass makes that meaning explicit
    once: for every physical axis of the module's PhysicalMesh missing from
    a collective's input_mesh or output_mesh, it inserts an explicit
    ReplicationAxis factor (and a matching identity Replicate->Replicate
    mapping pair) so every axis of the whole mesh is always directly
    represented on both sides, and downstream passes never need to infer
    replication from omission.
  }];
}

def AtomizeCollectivesPass
    : Pass<"distributed-atomize-collectives", "::mlir::ModuleOp"> {
  let summary = "Rewrites every distributed.Collective into atomic form: one "
                "factor per mesh atom, atom-to-atom mapping pairs.";
  let dependentDialects = [
    "enzyme::axis::AxisDialect", "enzyme::distributed::DistributedDialect"
  ];
  let description = [{
    Rewrites every distributed.Collective into atomic form. A collective is
    atomic when:

      * every mesh atom appears once in input_mesh and once in output_mesh,
      * every mapping pair relates one factor to one factor, and
      * on each axis, any two factors the collective mentions (mesh,
        reduction or mapping) are either equal or disjoint.

    An atom is a piece of an axis that no factor of the collective cuts
    further. The rewrite asserts its own result is atomic
    (isCollectiveAtomic in CollectiveAtoms.h); nothing in the IR marks a
    collective as atomic for other passes to check.

    Expects distributed-make-replications-explicit to have run. A collective
    with no common atom basis (a mapping pair that cannot be split into equal
    atoms, such as 3*2 against 2*3, or cuts on an axis that do not nest) is
    left unrewritten with a remark; the pass still succeeds, since such a
    collective is an infeasible sharding candidate rather than a bug.
  }];
}

def FuseCollectivesPass
    : Pass<"distributed-fuse-collectives", "::mlir::ModuleOp"> {
  let summary = "Fuses a distributed.Collective with the collective that "
                "consumes its result, e.g. reduce + slice into a "
                "reduce-scatter.";
  let dependentDialects = [
    "enzyme::axis::AxisDialect", "enzyme::distributed::DistributedDialect"
  ];
  let description = [{
    Fuses a pair of collectives (C1, C2) where C2's input_object is C1's
    result (through C1's single Await) and nothing else uses that result,
    repeating until no pair fuses so chains collapse. The fused collective
    reads C1's input, produces C2's output type, and is atomic (the pass
    asserts this, see isCollectiveAtomic in CollectiveAtoms.h). This makes
    role combinations reachable that two separate collectives never show,
    such as reduce-scatter (an all-reduce followed by a slice).

    The pair is resolved into one shared atom space in which C1's output tile
    and C2's input tile are the same axes, so the two are cut consistently
    even when they were written with different factorizations. Composition is
    then per atom:

      * a pair (L1 -> R1) of C1 and (L2 -> R2) of C2 with R1 the same digit as
        L2 compose to (L1 -> R2); a replicate digit composes like any other,
      * C1's reduced atoms stay reduced, and a C2 reduced atom is followed
        back through the C1 pair that produced it, whose source becomes
        reduced (C2 must not reduce an atom C1 produced from a replicate,
        which the pass asserts),
      * C1 and C2 reductions are merged into one group with one body.

    A pair is left alone, with a remark on C2 saying why, when C1's result has
    another user, the reduction bodies are not the same recognized kind, a C2
    reduction traces back to a device-local tile atom of C1's input (the model
    only reduces over mesh atoms), the two collectives touch different mesh
    axes, or the joint atom space cannot be built.

    Expects distributed-make-replications-explicit to have run, so that every
    mesh atom appears on both sides of both collectives. Idempotent.
  }];
}

def PrintComputeCostPass
    : Pass<"distributed-print-compute-cost", "::mlir::ModuleOp"> {
  let summary = "Debug pass: reports the roofline compute cost of each op in "
                "every distributed kernel body as remarks.";
  let description = [{
    Each costed op gets a remark with its flops, bytes and roofline time, and
    the module gets a summary remark. The device's peak rates come from the
    unique physical mesh's device_flops and device_mem_bandwidth, else abstract
    defaults; the options override both. See ComputeCost.h for the model.
  }];
  let dependentDialects = [
    "enzyme::axis::AxisDialect", "enzyme::distributed::DistributedDialect"
  ];
  let options = [
    Option<"flops", "flops", "double", "-1.0",
           "Peak FLOP per time unit of one device; negative uses the mesh's "
           "device_flops">,
    Option<"memBandwidth", "mem-bandwidth", "double", "-1.0",
           "Peak memory bytes per time unit of one device; negative uses the "
           "mesh's device_mem_bandwidth">
  ];
}

def PrintCollectivePlanPass
    : Pass<"distributed-print-collective-plan", "::mlir::ModuleOp"> {
  let summary = "Debug pass: reports each atomic collective's normal form "
                "(mesh atom roles and primitive) as remarks.";
  let dependentDialects = [
    "enzyme::axis::AxisDialect", "enzyme::distributed::DistributedDialect"
  ];
  let options = [
    Option<"chain", "chain", "bool", "false",
           "Also report the decomposition of each collective into primitive "
           "steps with their footprints">,
    ListOption<"bandwidths", "bandwidths", "double",
               "With chain: bandwidth per physical mesh axis; the mesh's "
               "axis_bandwidths (else 1) when omitted">,
    Option<"disableAllToAll", "disable-all-to-all", "bool", "false",
           "With chain: do not offer all-to-all, so tile-to-tile atoms use "
           "gather + local slice">,
    Option<"disableReduceScatter", "disable-reduce-scatter", "bool", "false",
           "With chain: do not offer reduce-scatter, so reduced atoms "
           "scattering onto a tile atom use all-reduce + local slice">,
    Option<"disableRelayVariants", "disable-relay-variants", "bool", "false",
           "With chain: decompose mesh-coupled components only by the "
           "half-split, without fused exchanges, path closures or "
           "conjugations">,
    Option<"disablePeelVariants", "disable-peel-variants", "bool", "false",
           "With chain: do not offer a peel for a (Reduced, Replicate) atom, "
           "so it is always a flat all-reduce">,
    Option<"kOrder", "k-order", "int64_t", "-1",
           "With chain: run the heuristic step orderer, expanding only the "
           "top N candidates of the lowest size tier at each state (N = 1 is "
           "greedy); negative leaves the orderer off (exact search)">,
    Option<"kVariant", "k-variant", "int64_t", "-1",
           "With chain: keep only the top N non-baseline variants of each "
           "component; negative leaves it unlimited. Enables the orderer's "
           "budget like k-order">,
    Option<"maxStates", "max-states", "int64_t", "-1",
           "With chain: give up after expanding more than N decomposer "
           "states; negative leaves it unlimited. Enables the orderer's "
           "budget like k-order">,
  ];
  let description = [{
    Emits one remark per distributed.Collective describing its
    NormalizedCollective: each mesh atom with its (in, out) roles and the
    communication primitive those roles call for, the tile atoms, the input
    payload size in bytes, and each reduction group's kind. Does not change
    the IR. A collective that is not atomic or cannot be resolved gets a remark
    saying so; run distributed-atomize-collectives first.

    With chain=true, a second remark per collective gives the chain the
    decomposer chose (each step's kind, atoms, payloads, latency, per-axis
    volume and demand, and duration), the outcome of the independent
    semantic check of that chain, and the SSA ports that attach the chain to
    the surrounding program. Costs use the uniform defaults of
    MeshCostParams over the mesh axes' extents, with the bandwidths option
    overriding per-axis bandwidth. disable-all-to-all and
    disable-reduce-scatter exercise the gather + slice and all-reduce + slice
    alternatives, which a cost-minimizing search never selects.
    disable-relay-variants restricts mesh-coupled components to the
    half-split so the baseline can be compared with the cheaper variants.
    disable-peel-variants restricts a (Reduced, Replicate) atom to the flat
    all-reduce so that baseline can be compared with the peel.
    k-order, k-variant and max-states supply a PlanBudget to the decomposer;
    giving any of them (a nonnegative value) switches the heuristic step
    orderer on, with the unset ones unlimited. Without any, the search is
    the exact one.
  }];
}

def PrintPrimitiveFootprintsPass
    : Pass<"distributed-print-primitive-footprints", "::mlir::ModuleOp"> {
  let summary = "Test pass: prints the closed-form footprint of one primitive "
                "collective as a remark.";
  let options = [
    Option<"kind", "kind", "std::string", "\"\"",
           "One of all-reduce, reduce-scatter, all-gather, all-to-all, "
           "collective-permute, local-slice">,
    Option<"payloadBytes", "payload-bytes", "int64_t", "0",
           "Per-device payload before the step, in bytes">,
    ListOption<"extents", "extents", "int64_t",
               "Extent of each mesh atom the step runs over">,
    ListOption<"axes", "axes", "int64_t",
               "Physical mesh axis of each atom, parallel to extents">,
    ListOption<"bandwidths", "bandwidths", "double",
               "Bandwidth per mesh axis, in bytes/ns; uniform "
               "MeshCostParams::kDefaultBandwidth when omitted">,
    Option<"latency", "latency", "double", "1000.0",
           "Latency of one round on every axis, in nanoseconds">,
    ListOption<"changedFractions", "changed-fractions", "double",
               "Collective-permute only: fraction of devices whose digit "
               "changes, one per atom">,
  ];
  let description = [{
    Builds one PrimitiveStep with the closed-form footprint builders of
    CollectiveCost.h and reports its payloads, latency, per-axis volume and
    demand, and isolated duration as a remark on the module. It exists to
    check the cost formulas against hand-computed numbers.
  }];
}

def PrintScheduleGraphPass
    : Pass<"distributed-print-schedule-graph", "::mlir::ModuleOp"> {
  let summary = "Debug pass: reports the event DAG of a lowered candidate "
                "module (ScheduleGraph.h) as remarks.";
  let dependentDialects = [
    "enzyme::axis::AxisDialect", "enzyme::distributed::DistributedDialect"
  ];
  let description = [{
    Builds the event DAG (ScheduleGraph.h): one node per op inside every
    distributed.DistributedKernel body, costed by ComputeCost.h's roofline,
    and one node per primitive step of every atomic distributed.Collective's
    decomposed chain (CollectiveDecomposer.h), connected by SSA def-use edges
    (kernel and collective-chain boundaries bridged by value aliasing, not
    rewriting). Each node gets a remark with its kind, cost/demand summary and
    predecessor node indices; the module gets a summary remark. Mesh/device
    cost parameters come from the unique physical mesh's optional metadata,
    else abstract defaults (MeshCostMetadata.h). An unplannable collective
    makes the whole graph infeasible, reported as a single remark instead of
    per-node ones.
  }];
}

def PrintScheduleCostPass
    : Pass<"distributed-print-schedule-cost", "::mlir::ModuleOp"> {
  let summary = "Debug pass: reports the simulated makespan and critical "
                "path of a lowered candidate module (ScheduleSimulator.h) "
                "as remarks.";
  let dependentDialects = [
    "enzyme::axis::AxisDialect", "enzyme::distributed::DistributedDialect"
  ];
  let description = [{
    Builds the event DAG (ScheduleGraph.h) and runs the discrete-event,
    max-min fair-share timeline simulator (ScheduleSimulator.h) over it: one
    remark per top-level node reports its index, kind and simulated
    start/finish time, and one summary remark on the module reports the
    predicted makespan and the DAG's own critical path (source to sink node
    indices, ignoring resource contention; see ScheduleSimulator.h). Mesh/
    device cost parameters come from the unique physical mesh's optional
    metadata, else abstract defaults (MeshCostMetadata.h), matching
    ScheduleGraph.h's own node costs. An unplannable collective (or any
    other cause of ScheduleGraph infeasibility) is reported as a single
    remark instead of a makespan.
  }];
}

def LowerForSanityCheckPass
    : Pass<"distributed-lower-for-sanity-check", "::mlir::ModuleOp"> {
  let summary = "Lowers the whole distributed program to a single-device "
                "serial StableHLO reference implementation.";
  let dependentDialects = [
    "enzyme::distributed::DistributedDialect", "mlir::stablehlo::StablehloDialect",
    "mlir::func::FuncDialect"
  ];
  let description = [{
    Produces a "raw serial StableHLO" oracle for the whole distributed
    program: every distributed.DistributedKernel becomes an explicit
    stablehlo.while looping once per relevant mesh coordinate, slicing its
    local operand shard out of (and update-slicing its local result shard
    into) an explicitly materialized <mesh axes> x <local dims> tensor built
    outside the loop; every distributed.Collective becomes a reduction plus
    a transpose/reshape over that same materialized representation. Not
    meant to run efficiently -- only to be compared against real distributed
    execution as a correctness reference. distributed.function is converted
    to func.func as the final step.

    Expects to run at/after the very end of the current lowering pipeline,
    once every DistributedKernel is trivial (see isTriviallyLocalKernel) and
    every axis-algebra factor bounding a kernel or collective is either a
    real physical communication axis or an explicit replication marker, with
    no unlowered LogicalMeshAxis/DeviceLocalAxis residue. A kernel or
    bounding cast/collective that doesn't meet this precondition is a
    pipeline-ordering bug, not something this pass handles partially: it
    emits a diagnostic and fails outright.
  }];
}

#endif // ENZYME_AD_JAX_DISTRIBUTED_PASSES
