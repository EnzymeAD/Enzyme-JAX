#ifndef ENZYME_AD_JAX_DIALECT_BLAS_OPS_TD
#define ENZYME_AD_JAX_DIALECT_BLAS_OPS_TD

include "src/enzyme_ad/jax/Dialect/BLAS/Dialect.td"
include "src/enzyme_ad/jax/Dialect/BLAS/Attrs.td"

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "stablehlo/dialect/Base.td"

class BLAS_Op<string mnemonic, list<Trait> traits = []>
    : Op<BLAS_Dialect, mnemonic, traits>;

def SymmOp : BLAS_Op<"symm", [Pure, SameOperandsAndResultElementType]> {
  let summary = "Multiplication involving a symmetric matrix";

  let description = [{
    C := alpha*A*B + beta*C, or C := alpha*B*A + beta*C, where alpha and beta are scalars,  A is a symmetric matrix"
  }];

  let arguments = (ins
    HLO_ScalarTensor:$alpha,
    HLO_Tensor:$A,
    HLO_Tensor:$B,
    HLO_ScalarTensor:$beta,
    HLO_Tensor:$C,
    BLAS_SideAttr:$side,
    BLAS_UploAttr:$uplo
  );

  let results = (outs
    HLO_Tensor: $output
  );

  let assemblyFormat = [{
    $alpha `,` $A `,` $B `,` $beta `,` $C attr-dict `:` functional-type(operands, results)
  }];

  let hasVerifier = 1;
}

def SyrkOp : BLAS_Op<"syrk", [Pure, SameOperandsAndResultElementType]> {
  let summary = "Multiplication involving a symmetric matrix";

  let description = [{
    C_out := alpha*A*A^T + beta*C, or C_out := alpha*A^T*A + beta*C, where alpha and beta
    are scalars. C must be a n x n symmetric matrix.

    `output_uplo` determines which part of `C_out` is populated. Accessing the values in
    the non-`output_uplo` part of the matrix is undefined behavior.

    LAPACK/BLAS routines typically require a single `uplo` attribute and it is implicitly
    assumed that the output `uplo` corresponds to the input `uplo`. This means the burden
    lies on the user to manually copy data if they need to access the other half of the
    matrix. By specifying the `output_uplo` we can perform transformations that analyze the
    entire dataflow, and avoid computing/copying half of the tensor all together. Generally,
    it is recommended to set this attribute to `blas::BlasUplo::any`, and our passes
    will automatically refine this to minimize data copies.
  }];

  let arguments = (ins
    HLO_ScalarTensor:$alpha,
    HLO_Tensor:$A,
    HLO_ScalarTensor:$beta,
    HLO_Tensor:$C,
    BLAS_UploAttr:$uplo,
    BLAS_UploAttr:$output_uplo,
    DefaultValuedAttr<BLAS_TransposeAttr, "::mlir::enzymexla::blas::BlasTranspose::none">:$transpose
  );

  let results = (outs
    HLO_Tensor: $output
  );

  let assemblyFormat = [{
    $alpha `,` $A `,` $beta `,` $C attr-dict `:` functional-type(operands, results)
  }];

  let hasVerifier = 1;
}

def TrmmOp : BLAS_Op<"trmm", [Pure, SameOperandsAndResultElementType]> {
  let summary = "Multiplication involving a triangular matrix";

  let description = [{
    B := alpha * op(A) x B, or B := alpha * B x op(A), where alpha is a scalar,
    B is a m x n matrix, A is a unit, or non-unit, upper or lower triangular
    matrix, and op(A) is one of op(A) = A, or op(A) = A^T or A^H.
  }];

  let arguments = (ins
    HLO_ScalarTensor:$alpha,
    HLO_Tensor:$A,
    HLO_ScalarTensor:$B,
    BLAS_SideAttr:$side,
    BLAS_UploAttr:$uplo,
    DefaultValuedAttr<BLAS_TransposeAttr, "::mlir::enzymexla::blas::BlasTranspose::none">:$transpose
  );

  let results = (outs
    HLO_Tensor: $output
  );

  let assemblyFormat = [{
    $alpha `,` $A `,` $B `,` attr-dict `:` functional-type(operands, results)
  }];

  let hasVerifier = 1;
}

def TrsmOp : BLAS_Op<"trsm", [Pure]> {
  let summary = "Triangular matrix-matrix solve operation";
  let description = [{
    Solves op(A) * X = alpha * B, or X * op(A) = alpha * B, where alpha is a scalar,
    X and B are m x n matrices, A is a unit, or non-unit, upper or lower triangular
    matrix, and op(A) is one of op(A) = A, or op(A) = A^T or A^H.
  }];

  let arguments = (ins
    HLO_Tensor:$alpha,
    HLO_Tensor:$A,
    HLO_Tensor:$B,
    BLAS_SideAttr:$side,
    BLAS_UploAttr:$uplo,
    BLAS_TransposeAttr:$transa,
    BLAS_DiagAttr:$diag
  );

  let results = (outs
    HLO_Tensor:$output
  );

  let assemblyFormat = [{
    $alpha `,` $A `,` $B attr-dict `:` functional-type(operands, results)
  }];

  // TODO let hasVerifier = 1;
}

#endif // ENZYME_AD_JAX_DIALECT_BLAS_OPS_TD
