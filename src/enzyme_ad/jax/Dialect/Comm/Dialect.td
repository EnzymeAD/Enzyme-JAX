#ifndef ENZYMEXLA_COMM_DIALECT_TD
#define ENZYMEXLA_COMM_DIALECT_TD

include "mlir/IR/DialectBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"

def Comm_Dialect : Dialect {
  let name = "comm";
  let summary = "A dialect for communication primitives and optimizations";
  let description = [{}];
  let cppNamespace = "::mlir::enzymexla::comm";
  // let useDefaultAttributePrinterParser = 1;
  let useDefaultTypePrinterParser = 1;
}

class Comm_Type<string name, string typeMnemonic, list<Trait> traits = []> : TypeDef<Comm_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

class Comm_Op<string mnemonic, list<Trait> traits = []> : Op<Comm_Dialect, mnemonic, traits>;

#endif // ENZYMEXLA_COMM_DIALECT_TD
