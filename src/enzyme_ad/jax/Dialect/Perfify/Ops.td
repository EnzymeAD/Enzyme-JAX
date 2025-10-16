#ifndef ENZYME_AD_JAX_DIALECT_PERFIFY_OPS_TD
#define ENZYME_AD_JAX_DIALECT_PERFIFY_OPS_TD

include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/OpBase.td"
include "Dialect.td"

// Perfify.cost op
def CostOp : PerfifyOp<"cost", []> {
  // summary
  // description
  // arguments
  let arguments = (ins StrAttr:$target_op, 
                  APIntAttr:$cycle_cost);
  let assemblyFormat = "$target_op $cycle_cost attr-dict";
  
}

#endif // ENZYME_AD_JAX_DIALECT_PERFIFY_OPS_TD