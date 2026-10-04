#ifndef ENZYME_AD_JAX_DIALECT_DISTRIBUTED_DIALECT_TD
#define ENZYME_AD_JAX_DIALECT_DISTRIBUTED_DIALECT_TD

include "mlir/IR/DialectBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/Traits.td"
include "mlir/IR/OpBase.td"

def DistributedDialect : Dialect {
    let name = "distributed";
    let description = [{}];
    let cppNamespace = "::mlir::enzyme::distributed";
    let useDefaultTypePrinterParser = 1;
}

class DistributedType<string name, string type_mnemonic, list<Trait> traits = []> : TypeDef<DistributedDialect, name, traits>{
    let mnemonic = type_mnemonic;
}

class DistributedOp <string mnemonic, list<Trait> traits = []>
    : Op<DistributedDialect, mnemonic, traits>{
    }


#endif // ENZYME_AD_JAX_DIALECT_DISTRIBUTED_DIALECT_TD