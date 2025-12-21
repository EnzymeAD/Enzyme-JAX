#ifndef ENZYME_AD_JAX_DIALECT_TESSERA_ATTRS_TD
#define ENZYME_AD_JAX_DIALECT_TESSERA_ATTRS_TD

include "Dialect.td"

include "mlir/IR/AttrTypeBase.td"


def TesseraConvertAttr : TesseraAttr<"convert"> {
   let summary = "Marks a function for Tessera conversion";
   let parameters = (ins StringAttr:$value);
}

#endif // ENZYME_AD_JAX_DIALECT_TESSERA_ATTRS_TD
