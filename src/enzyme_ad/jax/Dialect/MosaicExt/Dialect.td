#ifndef ENZYME_AD_JAX_DIALECT_MOSAICEXT_DIALECT_TD
#define ENZYME_AD_JAX_DIALECT_MOSAICEXT_DIALECT_TD

include "mlir/IR/DialectBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/Traits.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// MosaicExt dialect definition.
//===----------------------------------------------------------------------===//

def MosaicExtDialect : Dialect {
  let name = "enzymexla_mosaic_ext";
  let summary = "A dialect for extending Mosaic TPU";
  let cppNamespace = "::mlir::enzymexla::mosaic_ext";
  let dependentDialects = [
    "mlir::tpu::TPUDialect", "mlir::func::FuncDialect",
    "mlir::memref::MemRefDialect", "mlir::vector::VectorDialect",
    "mlir::LLVM::LLVMDialect", "mlir::stablehlo::StablehloDialect"
  ];
}

//===----------------------------------------------------------------------===//
// Base MosaicExt operation definition.
//===----------------------------------------------------------------------===//

class MosaicExtOp<string mnemonic, list<Trait> traits = []>
    : Op<MosaicExtDialect, mnemonic, traits>;

class MosaicExtType<string name> : TypeDef<MosaicExtDialect, name>;

#endif // ENZYME_AD_JAX_DIALECT_MOSAICEXT_DIALECT_TD
