#ifndef ENZYME_AD_JAX_DISTRIBUTED_PASSES
#define ENZYME_AD_JAX_DISTRIBUTED_PASSES

include "mlir/Pass/PassBase.td"

def PrintShardyFunctionNamesPass
    : Pass<"print-shardy-function-names", "::mlir::ModuleOp"> {
  let summary = "Print names of functions that carry Shardy IR";
  let dependentDialects = [
    "enzyme::axis::AxisDialect", "sdy::SdyDialect",
    "enzyme::distributed::DistributedDialect"
  ];
  let description = [{Scans func.func operations and prints the symbol names of
                          functions that either carry sdy -
                      prefixed attributes on the function /
                          signature or contain sdy dialect operations /
                          attributes in their bodies.}];
}

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

def InsertPhysicalMeshPass : Pass<"insert-physical-mesh", "::mlir::ModuleOp"> {
  let summary = "Insert a mock distributed.PhysicalMesh derived from Shardy "
                "mesh metadata.";
  let dependentDialects =
      ["sdy::SdyDialect", "enzyme::distributed::DistributedDialect"];
  let description = [{
      Inserts a module -
      level distributed
          .PhysicalMesh when none exists by inferring axis extents from the
              common Shardy mesh seen in sharded functions
          .This currently uses a mock device_target and is intended to be
              replaced with loading an external physical mesh configuration.}];
}

def NaiveLogicalToPhysicalMeshPass
    : Pass<"naive-logical-to-physical-mesh", "::mlir::ModuleOp"> {
  let summary = "Naively replace module-level logical mesh axes with "
        "module-level physical mesh axes.";
  let dependentDialects = ["enzyme::distributed::DistributedDialect"];
  let description = [{
    Expects exactly one distributed.LogicalMeshAxes op and exactly one
    distributed.GetPhysicalMeshAxes op in module scope. Verifies that logical
    and physical axis extents match positionally, then rewrites logical-axis
    uses to the physical-axis values and erases the logical mesh op.}];
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

#endif // ENZYME_AD_JAX_DISTRIBUTED_PASSES
