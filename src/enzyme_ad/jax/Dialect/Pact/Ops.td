#ifndef ENZYME_AD_JAX_DIALECT_PACT_OPS_TD
#define ENZYME_AD_JAX_DIALECT_PACT_OPS_TD

include "mlir/Interfaces/SideEffectInterfaces.td"
include "Dialect.td"

def SubgroupReduceOp : PactOp<"subgroup_reduce", [Pure]> {
    let summary = "Subgroup reduction with explicit contract";
    let arguments = (ins
        AnyType:$input,
        StrAttr:$reduction_kind
    );
    let results = (outs AnyType:$result);
    let assemblyFormat = "$input $reduction_kind attr-dict `:` type($input) `->` type($result)";
}

#endif // ENZYME_AD_JAX_DIALECT_PACT_OPS_TD
