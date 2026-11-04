#ifndef ENZYME_AD_JAX_DIALECT_JULIA_DIALECT_TD
#define ENZYME_AD_JAX_DIALECT_JULIA_DIALECT_TD

include "mlir/IR/DialectBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/Traits.td"
include "mlir/IR/OpBase.td"

def Julia_Dialect : Dialect {
  let name = "julia";
  let summary = "Backend dialect for lowering StableHLO to native Julia functions";
  let description = [{
    The `julia` dialect models both value-domain (functional, SSA array) and
    buffer-domain (destination-passing, column-major `Array{T, N}` and zero-copy
    `SubArray`/`ReshapedArray` view) operations for compiling StableHLO, CHLO,
    and EnzymeXLA programs into optimized native Julia functions.
  }];
  let cppNamespace = "::mlir::enzymexla::julia";
  let useDefaultAttributePrinterParser = 1;
  let useDefaultTypePrinterParser = 1;
  let hasConstantMaterializer = 1;
}

class Julia_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<Julia_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

class Julia_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<Julia_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
}

class Julia_Op<string mnemonic, list<Trait> traits = []>
    : Op<Julia_Dialect, mnemonic, traits>;

#endif // ENZYME_AD_JAX_DIALECT_JULIA_DIALECT_TD
