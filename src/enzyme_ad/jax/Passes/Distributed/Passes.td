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
    Skeleton pass for eliminating distributed.DeviceLocalAxis markers left
    behind by LowerKernelsPass (axes that were never sharded across the mesh
    because they're serialized entirely within a device). This pass
    currently performs no rewrites.
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
