#ifndef ENZYME_AD_JAX_DIALECT_TESSERA_ATTRS_TD
#define ENZYME_AD_JAX_DIALECT_TESSERA_ATTRS_TD

include "Dialect.td"

include "mlir/IR/AttrTypeBase.td"

def TesseraConvertAttr : TesseraAttr<"Convert", "convert"> {
    let summary = "Marks a function for Tessera conversion";
    let parameters = (ins 
	"std::string":$opName,
	"DenseBoolArrayAttr":$byRefArgs,
	"DenseI64ArrayAttr":$argSizes, 
	"bool":$pure);
    let assemblyFormat = "$opName `byref` `=` $byRefArgs `sizes` `=` $argSizes `pure` `=` $pure";
    let cppClassName = "ConvertAttr";
}

#endif // ENZYME_AD_JAX_DIALECT_TESSERA_ATTRS_TD
