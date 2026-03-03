#ifndef ENZYMEXLA_DIALECT_LAPACK_OPS_TD
#define ENZYMEXLA_DIALECT_LAPACK_OPS_TD

include "src/enzyme_ad/jax/Dialect/LAPACK/Dialect.td"
include "src/enzyme_ad/jax/Dialect/BLAS/Attrs.td"

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "stablehlo/dialect/Base.td"

class LAPACK_Op<string mnemonic, list<Trait> traits = []>
    : Op<LAPACK_Dialect, mnemonic, traits>;

def GeqrfOp : LAPACK_Op<"geqrf", [Pure]> {
  let summary = "QR factorization operation (low-level op)";

  let description = [{
    This operation computes the QR factorization of a matrix using Householder
    reflections. Mathematically, it decomposes A into the product of an
    orthogonal matrix Q and an upper triangular matrix R, such that A = QR.

    This operation is modeled after LAPACK's *GEQRF routines, which returns the
    result in the QR packed format. 
  }];

  let arguments = (ins HLO_Tensor : $input);

  let results = (outs
    HLO_Tensor:$output,
    HLO_Tensor:$tau,
    HLO_Tensor:$info
  );

  let assemblyFormat = [{
    $input attr-dict `:` functional-type($input, results)
  }];
}

def GeqrtOp : LAPACK_Op<"geqrt", [Pure]> {
  let summary = "QR factorization operation (low-level op)";

  let description = [{
    This operation computes the QR factorization of a matrix using Householder
    reflections. Mathematically, it decomposes A into the product of an
    orthogonal matrix Q and an upper triangular matrix R, such that A = QR.

    This operation is modeled after LAPACK's *GEQRT routines, which returns the 
    result in the QR CompactWY format.
  }];

  let arguments = (ins
      HLO_Tensor:$input,
      OptionalAttr<I64Attr>:$blocksize // nb argument, min(m,n) >= nb >= 1
    );

  let results = (outs
    HLO_Tensor:$output,
    HLO_Tensor:$T,
    HLO_Tensor:$info
  );

  let assemblyFormat = [{
    $input attr-dict `:` functional-type($input, results)
  }];
}

def OrgqrOp : LAPACK_Op<"orgqr", [Pure]> {
  let summary = [{
    Generate orthonormal/unitary matrix from a product of
    elementary Householder reflectors
  }];

  let description = [{
    This operation is modeled after LAPACK's *ORGQR/*UNGQR routines.
  }];

  let arguments = (ins
    HLO_Tensor:$input,
    HLO_Tensor:$tau
  );

  let results = (outs
    HLO_Tensor:$output
  );

  let assemblyFormat = [{
    $input `,` $tau attr-dict `:` functional-type(operands, results)
  }];
}

def OrmqrOp : LAPACK_Op<"ormqr", [Pure]> {
  let summary = [{
    Perform a matrix multiplication with the Q matrix from a QR
    decomposition, given only the householder reflections.
  }];

  let description = [{
    This operation is modeled after LAPACK's *ORMQR routines.
  }];

  let arguments = (ins
    HLO_Tensor:$A,
    HLO_Tensor:$tau,
    HLO_Tensor:$C,
    BLAS_SideAttr:$side,
    DefaultValuedAttr<BLAS_TransposeAttr, "::mlir::enzymexla::blas::BlasTranspose::none">:$transpose // true if apply Q^T
  );

  let results = (outs
    HLO_Tensor:$output
  );

  let assemblyFormat = [{
    $A `,` $tau `,` $C attr-dict `:` functional-type(operands, results)
  }];
}

def GemqrtOp : LAPACK_Op<"gemqrt", [Pure]> {
  let summary = [{
    Perform a matrix multiplication with the Q matrix from a QR
    decomposition, given the WY decomposition (i.e. geqrt).
  }];

  let description = [{
    This operation is modeled after LAPACK's *GEMQR routines.
  }];

  let arguments = (ins
    HLO_Tensor:$V,
    HLO_Tensor:$T,
    HLO_Tensor:$C,
    BLAS_SideAttr:$side,
    DefaultValuedAttr<BLAS_TransposeAttr, "::mlir::enzymexla::blas::BlasTranspose::none">:$transpose // true if apply Q^T
  );

  let results = (outs
    HLO_Tensor:$output
  );

  let assemblyFormat = [{
    $V `,` $T `,` $C attr-dict `:` functional-type(operands, results)
}];
}

def GetrfOp : LAPACK_Op<"getrf", [Pure]> {
  let summary = "LU factorization operation with row-major pivoting.";

  let arguments = (ins HLO_Tensor : $input);

  let results = (outs
    HLO_Tensor:$output,
    HLO_Tensor:$pivots,
    HLO_Tensor:$permutation,
    HLO_Tensor:$info
  );

  let assemblyFormat = [{
    $input attr-dict `:` functional-type($input, results)
  }];
}

def GetriOp : LAPACK_Op<"getri", [Pure]> {
  let summary = "Computes the inverse of a matrix using the LU factorization.";

  let arguments = (ins
    HLO_Tensor:$input,
    HLO_Tensor:$ipiv
  );

  let results = (outs
    HLO_Tensor:$output
  );

  let assemblyFormat = [{
    $input `,` $ipiv attr-dict `:` functional-type(operands, results)
  }];
}

def GesddOp : LAPACK_Op<"gesdd", [Pure]> {
  let summary = "Singular Value Decomposition (SVD) factorization operation.";

  let arguments = (ins
    HLO_Tensor:$input,
    DefaultValuedAttr<BoolAttr, "false">:$full,
    DefaultValuedAttr<BoolAttr, "true">:$compute_uv
  );

  let results = (outs
    HLO_Tensor:$U,
    HLO_Tensor:$S,
    HLO_Tensor:$Vt,
    HLO_Tensor:$info
  );

  let assemblyFormat = [{
    $input attr-dict `:` functional-type($input, results)
  }];
}

def GesvdOp : LAPACK_Op<"gesvd", [Pure]> {
  let summary = "Singular Value Decomposition (SVD) factorization operation.";

  let arguments = (ins
    HLO_Tensor:$input,
    DefaultValuedAttr<BoolAttr, "false">:$full,
    DefaultValuedAttr<BoolAttr, "true">:$compute_uv
  );

  let results = (outs
    HLO_Tensor:$U,
    HLO_Tensor:$S,
    HLO_Tensor:$Vt,
    HLO_Tensor:$info
  );

  let assemblyFormat = [{
    $input attr-dict `:` functional-type($input, results)
  }];
}

def GesvjOp : LAPACK_Op<"gesvj", [Pure]> {
  let summary = "Singular Value Decomposition (SVD) factorization operation.";

  let arguments = (ins
    HLO_Tensor:$input,
    DefaultValuedAttr<BoolAttr, "false">:$full,
    DefaultValuedAttr<BoolAttr, "true">:$compute_uv
  );

  let results = (outs
    HLO_Tensor:$U,
    HLO_Tensor:$S,
    HLO_Tensor:$Vt,
    HLO_Tensor:$info
  );

  let assemblyFormat = [{
    $input attr-dict `:` functional-type($input, results)
  }];
}

#endif // ENZYMEXLA_DIALECT_LAPACK_OPS_TD
