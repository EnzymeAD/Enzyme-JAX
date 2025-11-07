#ifndef ENZYME_AD_JAX_DIALECT_TRITONEXT_DIALECT_TD
#define ENZYME_AD_JAX_DIALECT_TRITONEXT_DIALECT_TD

include "mlir/IR/DialectBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/Traits.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// TritonExt dialect definition.
//===----------------------------------------------------------------------===//

def TritonExtDialect : Dialect {
  let name = "enzymexla_triton_ext";
  let summary = "A dialect for extending Triton";
  let cppNamespace = "::mlir::enzymexla::triton_ext";
  let dependentDialects = [
    "mlir::triton::TritonDialect", "mlir::LLVM::LLVMDialect",
    "mlir::stablehlo::StablehloDialect"
  ];
}

//===----------------------------------------------------------------------===//
// Base TritonExt operation definition.
//===----------------------------------------------------------------------===//

class TritonExtOp<string mnemonic, list<Trait> traits = []>
    : Op<TritonExtDialect, mnemonic, traits>;

class TritonExtType<string name> : TypeDef<TritonExtDialect, name>;

#endif // ENZYME_AD_JAX_DIALECT_TRITONEXT_DIALECT_TD
