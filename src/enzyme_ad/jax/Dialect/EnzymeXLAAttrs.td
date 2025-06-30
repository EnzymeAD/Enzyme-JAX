#ifndef ENZYMEXLA_ATTRS
#define ENZYMEXLA_ATTRS

include "Dialect.td"

include "mlir/IR/EnumAttr.td"
include "mlir/IR/AttrTypeBase.td"

// Machine Learning

def EnzymeXLA_GeluApproximation : I32EnumAttr<"GeluApproximation",
    "Approximation to use for the GELU activation function",
    [
      I32EnumAttrCase<"NONE", 0>,
      I32EnumAttrCase<"TANH", 1>,
      I32EnumAttrCase<"SIGMOID", 2>
    ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::enzymexla";
}

def EnzymeXLA_GeluApproximationAttr : EnumAttr<EnzymeXLA_Dialect,
    EnzymeXLA_GeluApproximation, "gelu_approximation">;

#endif // ENZYMEXLA_ATTRS
