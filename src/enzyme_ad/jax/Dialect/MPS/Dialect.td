#ifndef ENZYMEXLA_MPS_DIALECT_TD
#define ENZYMEXLA_MPS_DIALECT_TD

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/IR/DialectBase.td"
include "mlir/IR/OpBase.td"

def MPS_Dialect : Dialect {
  let name = "mps";
  let summary = "Metal Performance Shaders Graph (MPSGraph) operations";
  let description = [{
    Operations of Apple's MPSGraph framework. Each op corresponds to one
    `MPSGraph` method; overloads of a method are folded into a single op.

    An `MPSGraphTensor` is represented by a builtin `tensor` value; its
    `MPSDataType` and shape are the tensor element type and shape. Parameters
    that only describe the result shape or data type (`shape`, `outputShape`,
    `dataType`, ...) are taken from the result type. Op descriptors
    (`MPSGraph*OpDescriptor`) are flattened into op attributes.
  }];
  let cppNamespace = "::mlir::mps";
  let useDefaultAttributePrinterParser = 1;
}

class MPS_Op<string mnemonic, list<Trait> traits = []>
    : Op<MPS_Dialect, mnemonic, traits> {
  let assemblyFormat =
      "operands prop-dict attr-dict `:` functional-type(operands, results)";
}

// MPSDataType. Signed integers use signless MLIR integer types.
def MPS_ElementType : AnyTypeOf<[
  F16, F32, BF16, I1, I<4>, I8, I16, I32, I64, UI<4>, UI8, UI16, UI32, UI64,
  Complex<F16>, Complex<F32>
], "MPSDataType">;

def MPS_Tensor : TensorOf<[MPS_ElementType]>;
def MPS_BoolTensor : TensorOf<[I1]>;

#endif // ENZYMEXLA_MPS_DIALECT_TD
