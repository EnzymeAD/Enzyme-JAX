#ifndef ENZYME_AD_JAX_DIALECT_PACT_ATTRS_TD
#define ENZYME_AD_JAX_DIALECT_PACT_ATTRS_TD

include "Dialect.td"
include "mlir/IR/AttrTypeBase.td"

def PactTargetCapabilityAttr : PactAttr<"TargetCapability", "target_capability"> {
    let summary = "Flattened target capability projection";
    let description = [{
        Stores projected hardware capabilities as key-value pairs.
        Attached to module or gpu.module by InjectCapabilityPass.
        Read by AdaptiveMeetPass and PACTToROCDLPass.
    }];
    let parameters = (ins
        "StringAttr":$target_id,
        "DictionaryAttr":$capabilities
    );
    let assemblyFormat = "`<` $target_id `,` $capabilities `>`";
}

def PactPlanAttr : PactAttr<"Plan", "plan"> {
    let summary = "Adaptation plan attached to a pact op";
    let description = [{
        Represents the result of the adaptive meeting
    }];
    let parameters = (ins
        "StringAttr":$strategy,
        "StringAttr":$meet_level,
        "DictionaryAttr":$details
    );
    let assemblyFormat = "`<` $strategy `,` $meet_level `,` $details `>`";
}

#endif // ENZYME_AD_JAX_DIALECT_PACT_ATTRS_TD
