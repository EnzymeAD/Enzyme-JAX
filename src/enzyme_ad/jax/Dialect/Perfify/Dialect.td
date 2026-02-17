#ifndef ENZYME_AD_JAX_DIALECT_PERFIFY_DIALECT_TD
#define ENZYME_AD_JAX_DIALECT_PERFIFY_DIALECT_TD

include "mlir/IR/DialectBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/Traits.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Perfify dialect definition.
//===----------------------------------------------------------------------===//

def PerfifyDialect : Dialect {
    let name = "perfify";
    let summary = "A dialect for specifying and proving runtime bounds";
    let description = [{
        Lets users specify a bound on the number of steps/latency (per a predefined cost model) that a function or other operation should take.
        Leverages SAT solvers to automatically prove this, or interactive theorem provers to allow for complete proofs. 
    }];
    let cppNamespace = "::mlir::enzyme::perfify";
}

//===----------------------------------------------------------------------===//
// Base Perfify operation definition.
//===----------------------------------------------------------------------===//

class PerfifyOp<string mnemonic, list<Trait> traits = []>
    : Op<PerfifyDialect, mnemonic, traits>;

class PerfifyType<string name, string type_mnemonic, list<Trait> traits = []> : TypeDef<PerfifyDialect, name, traits>{
    let mnemonic = type_mnemonic;
}
#endif // ENZYME_AD_JAX_DIALECT_PERFIFY_DIALECT_TD