#ifndef ENZYME_AD_JAX_DIALECT_AXIS_DIALECT_TD
#define ENZYME_AD_JAX_DIALECT_AXIS_DIALECT_TD

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/DialectBase.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/Traits.td"

def AxisDialect : Dialect {
  let name = "axis";
  let description = [{SSA algebra over canonical axes and axis factors.}];
  let cppNamespace = "::mlir::enzyme::axis";
  let useDefaultTypePrinterParser = 1;
}

class AxisDialectType<string name, string type_mnemonic,
                      list<Trait> traits = []>
    : TypeDef<AxisDialect, name, traits> {
  let mnemonic = type_mnemonic;
}

class AxisOp<string mnemonic, list<Trait> traits = []>
    : Op<AxisDialect, mnemonic, traits>;

#endif // ENZYME_AD_JAX_DIALECT_AXIS_DIALECT_TD
