#ifndef ENZYME_AD_JAX_DIALECT_PERFIFY_OPS_TD
#define ENZYME_AD_JAX_DIALECT_PERFIFY_OPS_TD

include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "Dialect.td"
include "Types.td"

def DefSymbolOp : PerfifyOp<"define_symbol", [Symbol]> {
  let arguments = (ins SymbolNameAttr:$sym_name, TypeAttr:$type);
  let assemblyFormat = "$sym_name `:` $type attr-dict";
}

def SymCostOp : PerfifyOp<"sym_cost", [
    DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let arguments = (ins FlatSymbolRefAttr:$sym);
  let results  = (outs AnyType:$result);
  let assemblyFormat = "$sym attr-dict `:` type($result)";
}

def CostOp : PerfifyOp<"cost", []> {
  let arguments = (ins StrAttr:$target_op);
  let regions = (region AnyRegion:$body);
  let assemblyFormat = "$target_op $body attr-dict";
}

def FnCostOp : PerfifyOp<"fn_cost", []> { // this should be where we calculate the cost in the pass
  let results = (outs CostType:$out_cost);
  let assemblyFormat = "attr-dict `:` type($out_cost)"; 
}

def CompareOp : PerfifyOp<"cmp", []> {
  let arguments = (ins
    PerfifyCmpPredicateAttr:$predicate,
    CostType:$exp_cost,
    CostType:$func_cost
  );
  let assemblyFormat = "$predicate `,` $exp_cost `,` $func_cost attr-dict";
  let results = (outs I1:$result);
}

def ConstantCostOp : PerfifyOp<"constant_cost", []> {
  let regions = (region AnyRegion:$body);
  let assemblyFormat = "$body attr-dict `:` type($result)";
  let results = (outs CostType:$result);
}

def ArgOp : PerfifyOp<"arg", []> {
  let arguments = (ins I64Attr:$val);
  let assemblyFormat = "$val attr-dict";
  let results = (outs CostType:$argval);
}

def AssumeOp : PerfifyOp<"assume", [HasParent<"ConditionsOp">, Terminator]> {
  let arguments = (ins I1:$condition, OptionalAttr<BoolAttr>:$satres);
  let assemblyFormat = "$condition attr-dict";
}

def ConditionsOp : PerfifyOp<"conditions", [HasParent<"AssumptionsOp">, Terminator]> {
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

def AssumptionsOp : PerfifyOp<"assumptions", [Terminator]> {
  let regions = (region AnyRegion:$body);
  let assemblyFormat = [{$body attr-dict}];
}

def YieldOp : PerfifyOp<"yield", [Terminator]> {
  let arguments = (ins Optional<AnyType>:$value);
  let assemblyFormat = "attr-dict ($value^ `:` type($value))?";
}

#endif // ENZYME_AD_JAX_DIALECT_PERFIFY_OPS_TD
