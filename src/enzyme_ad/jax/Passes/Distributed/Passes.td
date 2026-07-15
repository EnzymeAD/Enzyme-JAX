#ifndef ENZYME_AD_JAX_DISTRIBUTED_PASSES
#define ENZYME_AD_JAX_DISTRIBUTED_PASSES

include "mlir/Pass/PassBase.td"

def PrintShardyFunctionNamesPass
		: Pass<"print-shardy-function-names", "::mlir::ModuleOp"> {
	let summary = "Print names of functions that carry Shardy IR";
    let dependentDialects = [
        "enzyme::axis::AxisDialect",
        "sdy::SdyDialect",
        "enzyme::distributed::DistributedDialect"
    ];
	let description = [{
		Scans func.func operations and prints the symbol names of functions that
		either carry sdy-prefixed attributes on the function/signature or contain
		sdy dialect operations/attributes in their bodies.
	}];
}

def ShardyToDistributedPass : Pass<"shardy-to-distributed", "::mlir::ModuleOp"> {
    let summary = "Convert Shardy functions to Distributed mesh computations.";
    let dependentDialects = [
        "enzyme::axis::AxisDialect",
        "sdy::SdyDialect",
        "enzyme::distributed::DistributedDialect"
    ];
    let description = [{
        Converts from Shardy to Distributed IR. This pass requires all Shardy functions
        to be available in the module: calling external functions is not supported.
        Also requires exactly one physical mesh to be declared in the module.
        Shardy functions on multiple meshes will be supported in the future, but
        for now this pass will fail if multiple meshes are declared.

        Note: this pass expects local tensor sizes as input and all 
        layout mismatches to have explicit layout conversion ops, either 
        reshard or reduce. This can be achieved with some collection of existing
        shardy passes:
         --sdy-propagation-pipeline 
         --sdy-export-pipeline='enable-insert-explicit-collectives=true'
         --sdy-convert-global-to-local='enable-rgv3=true'
    }];
}

def InsertPhysicalMeshPass : Pass<"insert-physical-mesh", "::mlir::ModuleOp"> {
    let summary = "Insert a mock distributed.PhysicalMesh derived from Shardy mesh metadata.";
    let dependentDialects = [
        "sdy::SdyDialect",
        "enzyme::distributed::DistributedDialect"
    ];
    let description = [{
        Inserts a module-level distributed.PhysicalMesh when none exists by
        inferring axis extents from the common Shardy mesh seen in sharded
        functions. This currently uses a mock device_target and is intended
        to be replaced with loading an external physical mesh configuration.
    }];
}

def DistributedToHloPass : Pass<"distributed-to-hlo", "::mlir::ModuleOp"> {
    let summary = "Lower distributed dialect constructs to StableHLO.";
    let dependentDialects = [
        "enzyme::distributed::DistributedDialect",
        "stablehlo::StablehloDialect"
    ];
    let description = [{
        Skeleton pass for lowering the distributed dialect to StableHLO.
        This pass currently performs no rewrites.
    }];
}

def CanonicalizeAxisMapsPass
    : Pass<"canonicalize-axis-maps", "::mlir::ModuleOp"> {
    let summary = "Rewrite axis.map mappings into maximal indivisible form.";
    let dependentDialects = [
        "enzyme::axis::AxisDialect"
    ];
    let description = [{
        Rewrites each axis.map pair using split_divisible, replacing mapping
        groups with maximal one-to-one split groups.
    }];
}

#endif // ENZYME_AD_JAX_DISTRIBUTED_PASSES
