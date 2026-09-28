#ifndef ENZYME_AD_JAX_DIALECT_TESSERA_ATTRS_TD
#define ENZYME_AD_JAX_DIALECT_TESSERA_ATTRS_TD

include "Dialect.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"

def TesseraArgDirection : I64EnumAttr<"ArgDirection", 
        "direction of a lifted tessera argument",
        [I64EnumAttrCase<"in", 0>,
         I64EnumAttrCase<"out", 1>,
         I64EnumAttrCase<"inout", 2>]> {
    let genSpecializedAttr = 0;
    let cppNamespace = "::mlir::enzyme::tessera";
}

def TesseraArgDirectionAttr : EnumAttr<TesseraDialect, TesseraArgDirection, "dir"> {
    let assemblyFormat = "`<` $value `>`";
}

#endif // ENZYME_AD_JAX_DIALECT_TESSERA_ATTRS_TD
