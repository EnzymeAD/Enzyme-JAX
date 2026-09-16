#ifndef ENZYME_AD_JAX_DISTRIBUTED_PASSES
#define ENZYME_AD_JAX_DISTRIBUTED_PASSES

include "mlir/Pass/PassBase.td"


def ConvertMainToDistributedFunctionPass
    : Pass<"convert-main-to-distributed-function", "::mlir::ModuleOp"> {
  let summary = "Convert the main func.func into distributed.function.";
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
  let dependentDialects = ["enzyme::distributed::DistributedDialect"];
  let options = [
    Option<"dumpCandidates", "dump-candidates", "bool", "false", "Print the cloned, partially-rewritten IR for each scored search candidate, for debugging. Default: false">,
    Option<"dumpFinalized", "dump-finalized", "bool", "false", "Print the cloned, partially-rewritten IR for each finalized search candidate, for debugging. Default: false">,
    Option<"dumpBest", "dump-best", "bool", "false", "Print the cloned, partially-rewritten IR for the best search candidate, for debugging. Default: false">,
    Option<"disableVerifier", "disable-verifier", "bool", "true", "Disable the MLIR verifier when running the internal lowering pipeline on each candidate clone during search, for runtime reasons. Default: true">,
    Option<"beamSize", "beam-size", "int", "100", "Maximum number of candidates retained per generation of the beam search. Default: 100">
  ];
}

#endif // ENZYME_AD_JAX_DISTRIBUTED_PASSES
