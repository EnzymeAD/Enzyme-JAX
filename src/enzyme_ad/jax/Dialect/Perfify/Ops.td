#ifndef ENZYME_AD_JAX_DIALECT_PERFIFY_OPS_TD
#define ENZYME_AD_JAX_DIALECT_PERFIFY_OPS_TD

include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "Dialect.td"
include "Types.td"

def DefSymbolOp
    : PerfifyOp<"define_symbol", [Symbol, SymbolName, SymbolVisibility]> {
  let arguments = (ins SymbolNameAttr:$sym_name, TypeAttr:$type,
                   OptionalAttr<StrAttr>:$sym_visibility);
  let assemblyFormat = "($sym_visibility^)? $sym_name `:` $type attr-dict";
}

def SymCostOp : PerfifyOp<"sym_cost",
                          [DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let arguments = (ins FlatSymbolRefAttr : $sym);
  let results = (outs AnyType : $result);
  let assemblyFormat = "$sym attr-dict `:` type($result)";
}

def CostOp : PerfifyOp<"cost", [IsolatedFromAbove]> {
  let arguments = (ins StrAttr : $target_op);
  let regions = (region AnyRegion : $body);
  let assemblyFormat = "$target_op $body attr-dict";
}

def FnCostOp : PerfifyOp<"fn_cost", []> { // this should be where we calculate
                                          // the cost in the pass
  let results = (outs CostType : $out_cost);
  let assemblyFormat = "attr-dict `:` type($out_cost)";
}

def CompareOp : PerfifyOp<"cmp", []> {
  let arguments = (ins
    PerfifyCmpPredicateAttr:$predicate,
    CostType:$cost,
    FlatSymbolRefAttr:$bound
  );
  let assemblyFormat = "$predicate `,` $cost `,` $bound attr-dict";
  let results = (outs I1:$result);
  let extraClassDeclaration = [{ Operation *getBoundOp(); }];
  let hasVerifier = 1;
}

def EstimatedCostOp : PerfifyOp<"estimated_cost", [IsolatedFromAbove]> {
  let arguments = (ins SymbolNameAttr:$name);
  let regions = (region AnyRegion:$body);
  let assemblyFormat = "$name $body attr-dict";
  let hasVerifier = 1;
}

def ArgOp : PerfifyOp<"arg", []> {
  let arguments = (ins I64Attr:$val);
  let assemblyFormat = "$val attr-dict";
  let results = (outs CostType:$argval);
}

def AssumeOp : PerfifyOp<"assume", [HasParent<"ConditionsOp">, Terminator]> {
  let arguments = (ins I1:$condition, OptionalAttr<BoolAttr>:$satres);
  let assemblyFormat = "$condition (` ``{` `satres` `=` $satres^ `}`)? attr-dict";
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

def AssumptionsOp : PerfifyOp<"assumptions", []> {
  let regions = (region AnyRegion:$body);
  let assemblyFormat = [{$body attr-dict}];
}

def YieldOp : PerfifyOp<"yield", [Terminator]> {
  let arguments = (ins Optional<AnyType>:$value);
  let assemblyFormat = "attr-dict ($value^ `:` type($value))?";
}

def HypothesisOp : PerfifyOp<"hypothesis", []> {
  let arguments = (ins I1:$cmp, PerfifyHoareStates:$pre_post);
  let results = (outs I1:$result);
  let assemblyFormat = "$cmp $pre_post attr-dict";
  let hasVerifier = 1;
}

def IntervalOp : PerfifyOp<"interval", []> {
  let arguments = (ins SymbolNameAttr:$name,
                   OptionalAttr<FlatSymbolRefAttr>:$min_cost,
                   OptionalAttr<FlatSymbolRefAttr>:$max_cost);
  let assemblyFormat =
      "$name custom<IntervalBounds>($min_cost, $max_cost) attr-dict";
  let extraClassDeclaration = [{
    EstimatedCostOp getMinCostOp();
    EstimatedCostOp getMaxCostOp();
  }];
  let hasVerifier = 1;
}

#endif // ENZYME_AD_JAX_DIALECT_PERFIFY_OPS_TD
