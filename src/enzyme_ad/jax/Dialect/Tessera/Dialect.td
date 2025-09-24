#ifndef ENZYME_AD_JAX_DIALECT_TESSERA_DIALECT_TD
#define ENZYME_AD_JAX_DIALECT_TESSERA_DIALECT_TD

include "mlir/IR/DialectBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/Traits.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Tessera dialect definition.
//===----------------------------------------------------------------------===//

def TesseraDialect : Dialect {
    let name = "tessera";
    let summary = "A dialect for custom operations and scientific optimizations";
    let description = [{}];
    let cppNamespace = "::mlir::enzyme::tessera";
    let useDefaultTypePrinterParser = 1; // type or attribute?
}

//===----------------------------------------------------------------------===//
// Base Tessera operation definition.
//===----------------------------------------------------------------------===//

class TesseraOp<string mnemonic, list<Trait> traits = []>
    : Op<TesseraDialect, mnemonic, traits>;

class TesseraType<string name> : TypeDef<TesseraDialect, name>; // may need to be modified

#endif // ENZYME_AD_JAX_DIALECT_TESSERA_DIALECT_TD