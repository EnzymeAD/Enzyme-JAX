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
      I32EnumAttrCase<"MPI_DATATYPE_NULL", 0>,
      I32EnumAttrCase<"MPI_INT8_T", 1>,
      I32EnumAttrCase<"MPI_UINT8_T", 2>,
      I32EnumAttrCase<"MPI_INT16_T", 3>,
      I32EnumAttrCase<"MPI_UINT16_T", 4>,
      I32EnumAttrCase<"MPI_INT32_T", 5>,
      I32EnumAttrCase<"MPI_UINT32_T", 6>,
      I32EnumAttrCase<"MPI_INT64_T", 7>,
      I32EnumAttrCase<"MPI_UINT64_T", 8>,
      I32EnumAttrCase<"MPI_BYTE", 9>,
      I32EnumAttrCase<"MPI_SHORT", 10>,
      I32EnumAttrCase<"MPI_UNSIGNED_SHORT", 11>,
      I32EnumAttrCase<"MPI_INT", 12>,
      I32EnumAttrCase<"MPI_UNSIGNED", 13>,
      I32EnumAttrCase<"MPI_LONG", 14>,
      I32EnumAttrCase<"MPI_UNSIGNED_LONG", 15>,
      I32EnumAttrCase<"MPI_LONG_LONG_INT", 16>,
      I32EnumAttrCase<"MPI_UNSIGNED_LONG_LONG", 17>,
      I32EnumAttrCase<"MPI_CHAR", 18>,
      I32EnumAttrCase<"MPI_SIGNED_CHAR", 19>,
      I32EnumAttrCase<"MPI_UNSIGNED_CHAR", 20>,
      I32EnumAttrCase<"MPI_WCHAR", 21>,
      I32EnumAttrCase<"MPI_FLOAT", 22>,
      I32EnumAttrCase<"MPI_DOUBLE", 23>,
      I32EnumAttrCase<"MPI_C_FLOAT_COMPLEX", 24>,
      I32EnumAttrCase<"MPI_C_DOUBLE_COMPLEX", 25>,
      I32EnumAttrCase<"MPI_C_BOOL", 26>
    ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::enzymexla";
}

def EnzymeXLA_MPIDatatypeAttr : EnumAttr<EnzymeXLA_Dialect,
    EnzymeXLA_MPIDatatype, "datatype"> {
  let assemblyFormat = "`<` $value `>`";
}

def EnzymeXLA_MPIOp : I32EnumAttr<"MPIOp",
    "MPI Operator",
    [
      I32EnumAttrCase<"MPI_OP_NULL", 0>,
      I32EnumAttrCase<"MPI_BAND", 1>,
      I32EnumAttrCase<"MPI_BOR", 2>,
      I32EnumAttrCase<"MPI_BXOR", 3>,
      I32EnumAttrCase<"MPI_LAND", 4>,
      I32EnumAttrCase<"MPI_LOR", 5>,
      I32EnumAttrCase<"MPI_LXOR", 6>,
      I32EnumAttrCase<"MPI_MAX", 7>,
      I32EnumAttrCase<"MPI_MIN", 8>,
      I32EnumAttrCase<"MPI_PROD", 9>,
      I32EnumAttrCase<"MPI_REPLACE", 10>,
      I32EnumAttrCase<"MPI_SUM", 11>,
      I32EnumAttrCase<"MPI_NO_OP", 12>
    ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::enzymexla";
}

def EnzymeXLA_MPIOpAttr : EnumAttr<EnzymeXLA_Dialect,
    EnzymeXLA_MPIOp, "op"> {
  let assemblyFormat = "`<` $value `>`";
}

#endif // ENZYMEXLA_ATTRS
