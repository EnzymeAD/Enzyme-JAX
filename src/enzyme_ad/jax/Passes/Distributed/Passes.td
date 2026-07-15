#ifndef ENZYME_AD_JAX_DISTRIBUTED_PASSES
#define ENZYME_AD_JAX_DISTRIBUTED_PASSES

include "mlir/Pass/PassBase.td"

def PrintShardyFunctionNamesPass
		: Pass<"print-shardy-function-names", "::mlir::ModuleOp"> {
	let summary = "Print names of functions that carry Shardy IR";
	let description = [{
		Scans func.func operations and prints the symbol names of functions that
		either carry sdy-prefixed attributes on the function/signature or contain
		sdy dialect operations/attributes in their bodies.
	}];
}

#endif // ENZYME_AD_JAX_DISTRIBUTED_PASSES
