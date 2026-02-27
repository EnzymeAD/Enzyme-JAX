#ifndef ENZYME_AD_JAX_DIALECT_PACT_DIALECT_TD
#define ENZYME_AD_JAX_DIALECT_PACT_DIALECT_TD

include "mlir/IR/DialectBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/Traits.td"
include "mlir/IR/OpBase.td"

def PactDialect : Dialect {
    let name = "pact";
    let summary = "Implicit execution contract dialect for cross-target retargeting";
    let description = [{
        Pact (Program Assumption Contract Tracking) captures implicit execution
        assumptions in CUDA code and makes them explicit for cross-target compilation.
    }];
    let cppNamespace = "::mlir::enzyme::pact";
    let useDefaultAttributePrinterParser = 1;
}

class PactOp<string mnemonic, list<Trait> traits = []>
    : Op<PactDialect, mnemonic, traits>;

class PactAttr<string name, string attr_mnemonic> : AttrDef<PactDialect, name> {
    let mnemonic = attr_mnemonic;
}

#endif // ENZYME_AD_JAX_DIALECT_PACT_DIALECT_TD
