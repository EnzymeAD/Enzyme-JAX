#ifndef ENZYME_PERFIFY_DIALECT_TYPES_H
#define ENZYME_PERFIFY_DIALECT_TYPES_H

include "Dialect.td"

class PerfifyType<string name, string type_mnemonic, list<Trait> traits = []> : TypeDef<PerfifyDialect, name, traits>{
    let mnemonic = type_mnemonic;
}


def CostType : PerfifyType<"Cost", "cost">;
#endif // ENZYME_PERFIFY_DIALECT_TYPES_H