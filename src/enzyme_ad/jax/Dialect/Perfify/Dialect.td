#ifndef ENZYME_AD_JAX_DIALECT_PERFIFY_DIALECT_TD
#define ENZYME_AD_JAX_DIALECT_PERFIFY_DIALECT_TD

include "mlir/IR/BuiltinAttributes.td"
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
    let useDefaultAttributePrinterParser = 1;
    let useDefaultTypePrinterParser = 1;
}

//===----------------------------------------------------------------------===//
// Base Perfify operation definition.
//===----------------------------------------------------------------------===//

class PerfifyOp<string mnemonic, list<Trait> traits = []>
    : Op<PerfifyDialect, mnemonic, traits>;

class PerfifyAttr<string attrName, string attrMnemonic>
    : AttrDef<PerfifyDialect, attrName, []> {
  let mnemonic = attrMnemonic;
}

def PerfifyCostAttr : PerfifyAttr<"Cost", "cost"> {
    let cppClassName = "CostAttr";
    let assemblyFormat = "$value";
    let parameters = (ins APIntParameter<"">:$value);
}
#endif // ENZYME_AD_JAX_DIALECT_PERFIFY_DIALECT_TD