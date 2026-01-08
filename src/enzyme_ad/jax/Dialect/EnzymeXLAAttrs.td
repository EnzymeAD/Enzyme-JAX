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

def EnzymeXLA_LapackTranspose : I32EnumAttr<"LapackTranspose",
    "Whether to transpose the matrix",
    [
      I32EnumAttrCase<"none", 0>,
      I32EnumAttrCase<"transpose", 1>,
      I32EnumAttrCase<"adjoint", 2>
    ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::enzymexla";
}

def EnzymeXLA_LapackTransposeAttr : EnumAttr<EnzymeXLA_Dialect,
    EnzymeXLA_LapackTranspose, "transpose"> {
  let assemblyFormat = "`<` $value `>`";
}

def EnzymeXLA_LapackSide : I32EnumAttr<"LapackSide",
    "Side of the matrix to apply the operation",
    [
      I32EnumAttrCase<"left", 0>,
      I32EnumAttrCase<"right", 1>
    ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::enzymexla";
}

def EnzymeXLA_LapackSideAttr : EnumAttr<EnzymeXLA_Dialect,
    EnzymeXLA_LapackSide, "side"> {
  let assemblyFormat = "`<` $value `>`";
}

def EnzymeXLA_LapackUplo : I32EnumAttr<"LapackUplo",
    "Uplo of the matrix to apply the operation. If `F` is specified, either the upper or lower triangle of the matrix (which must be symmetric) can be used for the operation.",
    [
      I32EnumAttrCase<"U", 0>,
      I32EnumAttrCase<"L", 1>,
      I32EnumAttrCase<"F", 2>
    ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::enzymexla";
}

def EnzymeXLA_LapackUploAttr : EnumAttr<EnzymeXLA_Dialect,
    EnzymeXLA_LapackUplo, "uplo"> {
  let assemblyFormat = "`<` $value `>`";
}

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
  EnzymeXLA_QrAlgorithm, "qr_algorithm"> {
  let assemblyFormat = "`<` $value `>`";
}

def EnzymeXLA_SVDAlgorithm : I32EnumAttr<"SVDAlgorithm",
    "Algorithm to use for the SVD factorization",
    [
      I32EnumAttrCase<"DEFAULT", 0>,
      I32EnumAttrCase<"QRIteration", 1>,
      I32EnumAttrCase<"DivideAndConquer", 2>,
      I32EnumAttrCase<"Jacobi", 3>
    ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::enzymexla";
}

def EnzymeXLA_SVDAlgorithmAttr : EnumAttr<EnzymeXLA_Dialect,
    EnzymeXLA_SVDAlgorithm, "svd_algorithm"> {
  let assemblyFormat = "`<` $value `>`";
  let cppType = "::mlir::enzymexla::SVDAlgorithmAttr";
}

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

// Analysis Passes

def EnzymeXLA_GuaranteedAnalysisResult : I32EnumAttr<"GuaranteedAnalysisResult",
    "Result of an analysis pass",
    [
      I32EnumAttrCase<"GUARANTEED", 0>,
      I32EnumAttrCase<"NOTGUARANTEED", 1>,
      I32EnumAttrCase<"UNKNOWN", 2>
    ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::enzymexla";
}

def EnzymeXLA_GuaranteedAnalysisResultAttr : EnumAttr<EnzymeXLA_Dialect,
    EnzymeXLA_GuaranteedAnalysisResult, "guaranteed">;

// MPI

def EnzymeXLA_MPIDatatype : I32EnumAttr<"MPIDatatype",
    "MPI Datatype",
    [
      I32EnumAttrCase<"U", 0>,
      I32EnumAttrCase<"L", 1>,
      I32EnumAttrCase<"F", 2>
    ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::enzymexla";
}

def EnzymeXLA_MPIDatatypeAttr : EnumAttr<EnzymeXLA_Dialect,
    EnzymeXLA_MPIDatatype, "datatype"> {
  let assemblyFormat = "`<` $value `>`";
}

#endif // ENZYMEXLA_ATTRS
