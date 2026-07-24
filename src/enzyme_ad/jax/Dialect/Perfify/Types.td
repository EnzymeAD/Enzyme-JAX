#ifndef ENZYME_PERFIFY_DIALECT_TYPES_H
#define ENZYME_PERFIFY_DIALECT_TYPES_H

include "Dialect.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"

def PolyConstantCostTypeInterface : TypeInterface<"PolyConstantCostTypeInterface"> {
  let cppNamespace = "::mlir::enzyme::perfify";
  let methods = [];
}

class PerfifyType<string name, string type_mnemonic, list<Trait> traits = []> : TypeDef<PerfifyDialect, name, traits>{
    let mnemonic = type_mnemonic;
}

def ConstantCostType : PerfifyType<"ConstantCost", "ccost", [DeclareTypeInterfaceMethods<PolyConstantCostTypeInterface>]>;
def PolyCostType : PerfifyType<"PolyCost", "pcost", [DeclareTypeInterfaceMethods<PolyConstantCostTypeInterface>]>;
#endif // ENZYME_PERFIFY_DIALECT_TYPES_H