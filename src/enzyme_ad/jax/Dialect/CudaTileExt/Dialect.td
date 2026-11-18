#ifndef ENZYME_AD_JAX_DIALECT_CUDATILEEXT_DIALECT_TD
#define ENZYME_AD_JAX_DIALECT_CUDATILEEXT_DIALECT_TD

include "mlir/IR/DialectBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/Traits.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// CudaTileExt dialect definition.
//===----------------------------------------------------------------------===//

def CudaTileExtDialect : Dialect {
  let name = "enzymexla_ct_ext";
  let summary = "A dialect for extending CUDA Tile";
  let cppNamespace = "::mlir::enzymexla::cuda_tile_ext";
  let dependentDialects = [
    "mlir::cuda_tile::CudaTileDialect", "mlir::LLVM::LLVMDialect",
    "mlir::stablehlo::StablehloDialect"
  ];
}

//===----------------------------------------------------------------------===//
// Base CudaTileExt operation definition.
//===----------------------------------------------------------------------===//

class CudaTileExtOp<string mnemonic, list<Trait> traits = []>
    : Op<CudaTileExtDialect, mnemonic, traits>;

class CudaTileExtType<string name> : TypeDef<CudaTileExtDialect, name>;

#endif // ENZYME_AD_JAX_DIALECT_CUDATILEEXT_DIALECT_TD
