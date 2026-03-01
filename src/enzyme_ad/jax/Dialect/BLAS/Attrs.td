#ifndef ENZYMEXLA_DIALECT_BLAS_ATTRS_TD
#define ENZYMEXLA_DIALECT_BLAS_ATTRS_TD

include "Dialect.td"

include "mlir/IR/AttrTypeBase.td"

def BLAS_Layout : I8EnumAttr<"BlasLayout",
    "Ordering of elements in matrix",
    [
      I8EnumAttrCase<"col_major", 0x43>, // 'C'
      I8EnumAttrCase<"row_major", 0x52>  // 'R'
    ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::enzymexla::blas";
}

def BLAS_LayoutAttr : EnumAttr<BLAS_Dialect, BLAS_Layout, "layout"> {
  let assemblyFormat = "`<` $value `>`";
  let extraClassDeclaration = [{
    char getBlasChar() const {
      switch (getValue()) {
        case BLAS_Layout::col_major: return 'C';
        case BLAS_Layout::row_major: return 'R';
        default: llvm_unreachable("invalid BLAS_Layout");
      }
    }
  }];
}

def BLAS_Transpose : I8EnumAttr<"BlasTranspose",
    "Whether to transpose the matrix",
    [
      I8EnumAttrCase<"none", 0x4e>, // 'N'
      I8EnumAttrCase<"transpose", 0x54>, // 'T'
      I8EnumAttrCase<"adjoint", 0x43>  // 'C'
    ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::enzymexla::blas";
}

def BLAS_TransposeAttr : EnumAttr<BLAS_Dialect, BLAS_Transpose, "transpose"> {
  let assemblyFormat = "`<` $value `>`";
  let extraClassDeclaration = [{
    char getBlasChar() const {
      switch (getValue()) {
        case BLAS_Transpose::none: return 'N';
        case BLAS_Transpose::transpose: return 'T';
        case BLAS_Transpose::adjoint: return 'C';
        default: llvm_unreachable("invalid BLAS_Transpose");
      }
    }
  }];
}

def BLAS_Side : I8EnumAttr<"BlasSide",
    "Side of the matrix to apply the operation",
    [
      I8EnumAttrCase<"left", 0x4c>, // 'L'
      I8EnumAttrCase<"right", 0x52>  // 'R'
    ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::enzymexla::blas";
}

def BLAS_SideAttr : EnumAttr<BLAS_Dialect, BLAS_Side, "side"> {
  let assemblyFormat = "`<` $value `>`";
  let extraClassDeclaration = [{
    char getBlasChar() const {
      switch (getValue()) {
        case BLAS_Side::left: return 'L';
        case BLAS_Side::right: return 'R';
        default: llvm_unreachable("invalid BLAS_Side");
      }
    }
  }];
}

def BLAS_Uplo : I8EnumAttr<"BlasUplo",
    "Uplo of the matrix to apply the operation. If `F` is specified, either the upper or lower triangle of the matrix (which must be symmetric) can be used for the operation.",
    [
      I8EnumAttrCase<"upper", 0x55>, // 'U'
      I8EnumAttrCase<"lower", 0x4c>, // 'L'
      I8EnumAttrCase<"any", 0x46>    // 'F'
    ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::enzymexla::blas";
}

def BLAS_UploAttr : EnumAttr<BLAS_Dialect, BLAS_Uplo, "uplo"> {
  let assemblyFormat = "`<` $value `>`";
  let extraClassDeclaration = [{
    char getBlasChar() const {
      switch (getValue()) {
        case BLAS_Uplo::upper: return 'U';
        case BLAS_Uplo::lower: return 'L';
        case BLAS_Uplo::any: return 'F';
        default: llvm_unreachable("invalid BLAS_Uplo");
      }
    }
  }];
}

def BLAS_Diag : I8EnumAttr<"BlasDiag",
    "Whether the matrix is unit triangular",
    [
      I8EnumAttrCase<"non_unit", 0x4e>, // 'N'
      I8EnumAttrCase<"unit", 0x55>      // 'U'
    ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::enzymexla::blas";
}

def BLAS_DiagAttr : EnumAttr<BLAS_Dialect, BLAS_Diag, "diag"> {
  let assemblyFormat = "`<` $value `>`";
  let extraClassDeclaration = [{
    char getBlasChar() const {
      switch (getValue()) {
        case BLAS_Diag::non_unit: return 'N';
        case BLAS_Diag::unit: return 'U';
        default: llvm_unreachable("invalid BLAS_Diag");
      }
    }
  }];
}

#endif // ENZYMEXLA_DIALECT_BLAS_ATTRS_TD
