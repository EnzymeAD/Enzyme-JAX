#ifndef ENZYMEXLA_COMM_DIALECT_TD
#define ENZYMEXLA_COMM_DIALECT_TD

include "mlir/IR/DialectBase.td"
include "mlir/IR/OpBase.td"

def CommDialect : Dialect {
  let name = "comm";
  let summary = "A dialect for communication primitives and optimizations";
  let description = [{}];
  let cppNamespace = "::mlir::enzymexla::comm";
  let useDefaultAttributePrinterParser = 1;
  let useDefaultTypePrinterParser = 1;
}

class CommType<string name, string type_mnemonic, list<Trait> traits = []> : TypeDef<CommDialect, name, traits>;

class CommOp<string mnemonic, list<Trait> traits = []> : Op<CommDialect, mnemonic, traits>;

#endif // ENZYMEXLA_COMM_DIALECT_TD
