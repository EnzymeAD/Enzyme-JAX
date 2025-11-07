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

def ArgOp : PerfifyOp<"arg", []> {
  let arguments = (ins I64Attr:$val);
  let assemblyFormat = "$val attr-dict";
  let results = (outs I64);
}

def AssumeOp : PerfifyOp<"assume", []> {
  let arguments = (ins BoolAttr:$precondition);
  let assemblyFormat = "$precondition attr-dict";
}

def ConditionsOp : PerfifyOp<"conditions", []> {
  let arguments = (ins FlatSymbolRefAttr:$func_handle,
                  BoolAttr:$verify_huh);
  let regions = (region AnyRegion:$precondition, AnyRegion:$postcondition);

  let assemblyFormat = [{
    $func_handle $verify_huh attr-dict `pre`
    $precondition
    `post`
    $postcondition
  }];
}

def AssumptionsOp : PerfifyOp<"assumptions", []> {
  let regions = (region AnyRegion:$body);
  let assemblyFormat = [{$body attr-dict}];
}

#endif // ENZYME_AD_JAX_DIALECT_PERFIFY_OPS_TD