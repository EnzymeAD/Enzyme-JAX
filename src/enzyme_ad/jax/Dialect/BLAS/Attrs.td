#ifndef ENZYME_AD_JAX_DIALECT_BLAS_ATTRS_TD
#define ENZYME_AD_JAX_DIALECT_BLAS_ATTRS_TD

include "src/enzyme_ad/jax/Dialect/BLAS/Dialect.td"

include "mlir/IR/EnumAttr.td"

def BLAS_Layout : I32EnumAttr<"BlasLayout",
    "Ordering of elements in matrix",
    [
      I32EnumAttrCase<"col_major", 0x43>, // 'C'
      I32EnumAttrCase<"row_major", 0x52>  // 'R'
    ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::enzymexla::blas";
}

def BLAS_LayoutAttr : EnumAttr<BLAS_Dialect, BLAS_Layout, "layout"> {
  let assemblyFormat = "`<` $value `>`";
  let extraClassDeclaration = [{
    char getBlasChar() const {
      switch (getValue()) {
        case BlasLayout::col_major: return 'C';
        case BlasLayout::row_major: return 'R';
        default: llvm_unreachable("invalid BlasLayout");
      }
    }
  }];
}

def BLAS_Transpose : I32EnumAttr<"BlasTranspose",
    "Whether to transpose the matrix",
    [
      I32EnumAttrCase<"none", 0x4e>, // 'N'
      I32EnumAttrCase<"transpose", 0x54>, // 'T'
      I32EnumAttrCase<"adjoint", 0x43>  // 'C'
    ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::enzymexla::blas";
}

def BLAS_TransposeAttr : EnumAttr<BLAS_Dialect, BLAS_Transpose, "transpose"> {
  let assemblyFormat = "`<` $value `>`";
  let extraClassDeclaration = [{
    char getBlasChar() const {
      switch (getValue()) {
        case BlasTranspose::none: return 'N';
        case BlasTranspose::transpose: return 'T';
        case BlasTranspose::adjoint: return 'C';
        default: llvm_unreachable("invalid BlasTranspose");
      }
    }
  }];
}

def BLAS_Side : I32EnumAttr<"BlasSide",
    "Side of the matrix to apply the operation",
    [
      I32EnumAttrCase<"left", 0x4c>, // 'L'
      I32EnumAttrCase<"right", 0x52>  // 'R'
    ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::enzymexla::blas";
}

def BLAS_SideAttr : EnumAttr<BLAS_Dialect, BLAS_Side, "side"> {
  let assemblyFormat = "`<` $value `>`";
  let extraClassDeclaration = [{
    char getBlasChar() const {
      switch (getValue()) {
        case BlasSide::left: return 'L';
        case BlasSide::right: return 'R';
        default: llvm_unreachable("invalid BlasSide");
      }
    }
  }];
}

def BLAS_Uplo : I32EnumAttr<"BlasUplo",
    "Uplo of the matrix to apply the operation. If `F` is specified, either the upper or lower triangle of the matrix (which must be symmetric) can be used for the operation.",
    [
      I32EnumAttrCase<"upper", 0x55>, // 'U'
      I32EnumAttrCase<"lower", 0x4c>, // 'L'
      I32EnumAttrCase<"any", 0x46>    // 'F'
    ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::enzymexla::blas";
}

def BLAS_UploAttr : EnumAttr<BLAS_Dialect, BLAS_Uplo, "uplo"> {
  let assemblyFormat = "`<` $value `>`";
  let extraClassDeclaration = [{
    char getBlasChar() const {
      switch (getValue()) {
        case BlasUplo::upper: return 'U';
        case BlasUplo::lower: return 'L';
        case BlasUplo::any: return 'F';
        default: llvm_unreachable("invalid BlasUplo");
      }
    }
  }];
}

def BLAS_Diag : I32EnumAttr<"BlasDiag",
    "Whether the matrix is unit triangular",
    [
      I32EnumAttrCase<"non_unit", 0x4e>, // 'N'
      I32EnumAttrCase<"unit", 0x55>      // 'U'
    ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::enzymexla::blas";
}

def BLAS_DiagAttr : EnumAttr<BLAS_Dialect, BLAS_Diag, "diag"> {
  let assemblyFormat = "`<` $value `>`";
  let extraClassDeclaration = [{
    char getBlasChar() const {
      switch (getValue()) {
        case BlasDiag::non_unit: return 'N';
        case BlasDiag::unit: return 'U';
        default: llvm_unreachable("invalid BlasDiag");
      }
    }
  }];
}

#endif // ENZYME_AD_JAX_DIALECT_BLAS_ATTRS_TD
