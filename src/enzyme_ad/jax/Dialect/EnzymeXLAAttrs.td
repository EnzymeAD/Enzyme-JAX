#ifndef ENZYMEXLA_ATTRS
#define ENZYMEXLA_ATTRS

include "Dialect.td"

include "mlir/IR/EnumAttr.td"
include "mlir/IR/AttrTypeBase.td"

// Linear Algebra

def EnzymeXLA_LapackLayout : I32EnumAttr<"LapackLayout",
    "Ordering of elements in matrix",
    [
      I32EnumAttrCase<"col_major", 0>,
      I32EnumAttrCase<"row_major", 1>
    ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::enzymexla";
}

def EnzymeXLA_LapackLayoutAttr : EnumAttr<EnzymeXLA_Dialect,
    EnzymeXLA_LapackLayout, "layout">;

def EnzymeXLA_LapackSide : I32EnumAttr<"LapackSide",
    "Ordering of elements in matrix",
    [
      I32EnumAttrCase<"left", 0>,
      I32EnumAttrCase<"right", 1>
    ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::enzymexla";
}

def EnzymeXLA_LapackSideAttr : EnumAttr<EnzymeXLA_Dialect,
    EnzymeXLA_LapackSide, "side">;

def EnzymeXLA_QrAlgorithm : I32EnumAttr<"QrAlgorithm",
    "Algorithm to use for the QR factorization",
    [
      I32EnumAttrCase<"geqrf", 0>,
      I32EnumAttrCase<"geqrt", 1>
    ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::enzymexla";
}

def EnzymeXLA_QrAlgorithmAttr : EnumAttr<EnzymeXLA_Dialect,
  EnzymeXLA_QrAlgorithm, "qr_algorithm">;

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
