#ifndef ENZYME_AD_JAX_DIALECT_TRITONEXT_OPS_TD
#define ENZYME_AD_JAX_DIALECT_TRITONEXT_OPS_TD

include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

include "Dialect.td"

def TensorI64
    : Type<CPred<"::llvm::isa<::mlir::TensorType>($_self) && "
                 "::llvm::cast<::mlir::TensorType>($_self).getShape().size() "
                 "== 0 && "
                 "::llvm::cast<::mlir::TensorType>($_self).getElementType()."
                 "isSignlessInteger(64)">,
           "tensor<i64>", "::mlir::TensorType">,
      BuildableType<"RankedTensorType::get({}, $_builder.getIntegerType(64))">;

def TritonModuleOp : TritonExtOp<"module", [
  IsolatedFromAbove, NoRegionArguments, SymbolTable, Symbol, NoTerminator
]> {
  let summary = "Triton Module operation";

  let arguments = (ins SymbolNameAttr : $sym_name);
  let regions = (region SizedRegion<1> : $bodyRegion);

  // clang-format off
  let assemblyFormat = [{
    $sym_name
    attr-dict-with-keyword $bodyRegion
  }];
  // clang-format on
}

def TritonCallOp : TritonExtOp<"call", [
  DeclareOpInterfaceMethods<SymbolUserOpInterface>,
  DeclareOpInterfaceMethods<CallOpInterface>,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Triton Call operation";

  let arguments = (ins
    SymbolRefAttr:$fn,
    TensorI64:$gridx,
    TensorI64:$gridy,
    TensorI64:$gridz,
    TensorI64:$clusterx,
    TensorI64:$clustery,
    TensorI64:$clusterz,
    Variadic<AnyType>:$inputs,
    DefaultValuedStrAttr<StrAttr, "">:$backend_config,
    OptionalAttr<AnyAttr>:$operand_layouts,
    OptionalAttr<AnyAttr>:$result_layouts,
    OptionalAttr<DictArrayAttr>:$arg_attrs,
    OptionalAttr<DictArrayAttr>:$res_attrs,
    DefaultValuedOptionalAttr<
        ArrayAttr, "{}">:$output_operand_aliases,
    OptionalAttr<UnitAttr>:$xla_side_effect_free
  );

  let results = (outs Variadic<AnyType>);

  // clang-format off
  let assemblyFormat = [{
    $fn
    `clusters` `in` `(` $clusterx `,` $clustery `,` $clusterz `)`
    `blocks` `in` `(` $gridx `,` $gridy `,` $gridz `)`
    `(` $inputs `)`
    attr-dict `:` functional-type($inputs, results)
  }];
  // clang-format on

  let hasCanonicalizer = 1;
}

#endif // ENZYME_AD_JAX_DIALECT_TRITONEXT_OPS_TD
