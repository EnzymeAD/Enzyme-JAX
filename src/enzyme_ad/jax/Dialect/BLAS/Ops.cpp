#include "Dialect.h"

using namespace mlir;
using namespace mlir::blas;

#define GET_OP_CLASSES
#include "src/enzyme_ad/jax/Dialect/BLAS/BlasOps.cpp.inc"

LogicalResult SymmOp::verify() {
  auto alpha = getAlpha();
  auto A = getA();
  auto B = getB();
  auto beta = getBeta();
  auto C = getC();
  // auto side = getSide();
  // auto uplo = getUplo();

  auto type_alpha = cast<RankedTensorType>(alpha.getType());
  auto type_A = cast<RankedTensorType>(A.getType());
  auto type_B = cast<RankedTensorType>(B.getType());
  auto type_beta = cast<RankedTensorType>(beta.getType());
  auto type_C = cast<RankedTensorType>(C.getType());

  auto eltype = type_alpha.getElementType();

  if (eltype != type_A.getElementType() || eltype != type_B.getElementType() ||
      eltype != type_beta.getElementType() || eltype != type_C.getElementType())
    return emitOpError(
        "alpha, beta, A, B, and C must all have the same element type");

  if (!llvm::isa<FloatType>(eltype) && !llvm::isa<ComplexType>(eltype) &&
      !llvm::isa<FloatType>(cast<ComplexType>(eltype).getElementType()))
    return emitOpError("element type must be floating-point or complex type of "
                       "floating-point element type");

  if (type_alpha.getRank() != 0)
    return emitOpError("alpha must be a scalar");

  if (type_beta.getRank() != 0)
    return emitOpError("beta must be a scalar");

  // TODO check shapes of A, B, and C

  return success();
}

LogicalResult SyrkOp::verify() {
  auto alpha = getAlpha();
  auto A = getA();
  auto beta = getBeta();
  auto C = getC();
  // auto uplo = getUplo();
  // auto output_uplo = getOutputUplo();
  // auto transpose = getTranspose();

  auto type_alpha = cast<RankedTensorType>(alpha.getType());
  auto type_A = cast<RankedTensorType>(A.getType());
  auto type_beta = cast<RankedTensorType>(beta.getType());
  auto type_C = cast<RankedTensorType>(C.getType());

  auto eltype = type_alpha.getElementType();

  if (eltype != type_A.getElementType() ||
      eltype != type_beta.getElementType() || eltype != type_C.getElementType())
    return emitOpError(
        "alpha, beta, A, and C must all have the same element type");

  if (!llvm::isa<FloatType>(eltype) && !llvm::isa<ComplexType>(eltype) &&
      !llvm::isa<FloatType>(cast<ComplexType>(eltype).getElementType()))
    return emitOpError("element type must be floating-point or complex type of "
                       "floating-point element type");

  if (type_alpha.getRank() != 0)
    return emitOpError("alpha must be a scalar");

  if (type_beta.getRank() != 0)
    return emitOpError("beta must be a scalar");

  // TODO check shapes of A, B, and C

  bool is_complex = isa<ComplexType>(eltype);
  auto trans = getTranspose();
  if (!is_complex && trans == BlasTranspose::adjoint) {
    return emitOpError("cannot do the adjoint of a complex tensor");
  }

  return success();
}

LogicalResult TrmmOp::verify() {
  auto alpha = getAlpha();
  auto A = getA();
  auto B = getB();
  // auto side = getSide();
  // auto uplo = getUplo();
  // auto transpose = getTranspose();

  auto type_alpha = cast<RankedTensorType>(alpha.getType());
  auto type_A = cast<RankedTensorType>(A.getType());
  auto type_B = cast<RankedTensorType>(B.getType());

  auto eltype = type_alpha.getElementType();

  if (eltype != type_A.getElementType() || eltype != type_B.getElementType())
    return emitOpError("alpha, A, and B must all have the same element type");

  if (!llvm::isa<FloatType>(eltype) && !llvm::isa<ComplexType>(eltype) &&
      !llvm::isa<FloatType>(cast<ComplexType>(eltype).getElementType()))
    return emitOpError("element type must be floating-point or complex type of "
                       "floating-point element type");

  if (type_alpha.getRank() != 0)
    return emitOpError("alpha must be a scalar");

  // TODO check shapes of A and B

  return success();
}
