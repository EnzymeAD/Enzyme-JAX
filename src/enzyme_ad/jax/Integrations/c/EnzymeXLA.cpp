#include "EnzymeXLA.h"

#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"

#include "src/enzyme_ad/jax/Dialect/Dialect.h"
#include "src/enzyme_ad/jax/Dialect/Ops.h"

MlirAttribute enzymexlaLapackLayoutAttrGet(MlirContext ctx, uint8_t col_major) {
  mlir::enzymexla::LapackLayout layout;
  if (col_major) {
    layout = mlir::enzymexla::LapackLayout::col_major;
  } else {
    layout = mlir::enzymexla::LapackLayout::row_major;
  }
  return wrap(mlir::enzymexla::LapackLayoutAttr::get(unwrap(ctx), layout));
}

MlirAttribute enzymexlaLapackTransposeAttrGet(MlirContext ctx, int32_t mode) {
  mlir::enzymexla::LapackTranspose transpose;
  switch (mode) {
  case 0:
    transpose = mlir::enzymexla::LapackTranspose::none;
    break;
  case 1:
    transpose = mlir::enzymexla::LapackTranspose::transpose;
    break;
  case 2:
    transpose = mlir::enzymexla::LapackTranspose::adjoint;
  }
  return wrap(
      mlir::enzymexla::LapackTransposeAttr::get(unwrap(ctx), transpose));
}

MlirAttribute enzymexlaLapackSideAttrGet(MlirContext ctx, uint8_t left_side) {
  mlir::enzymexla::LapackSide side;
  if (left_side) {
    side = mlir::enzymexla::LapackSide::left;
  } else {
    side = mlir::enzymexla::LapackSide::right;
  }
  return wrap(mlir::enzymexla::LapackSideAttr::get(unwrap(ctx), side));
}

MlirAttribute enzymexlaLapackUploAttrGet(MlirContext ctx, uint8_t up) {
  mlir::enzymexla::LapackUplo uplo;
  if (up) {
    uplo = mlir::enzymexla::LapackUplo::U;
  } else {
    uplo = mlir::enzymexla::LapackUplo::L;
  }
  return wrap(mlir::enzymexla::LapackUploAttr::get(unwrap(ctx), uplo));
}

MlirAttribute enzymexlaQRAlgorithmAttrGet(MlirContext ctx, int32_t mode) {
  mlir::enzymexla::QrAlgorithm algorithm;
  switch (mode) {
  case 0:
    algorithm = mlir::enzymexla::QrAlgorithm::geqrf;
    break;
  case 1:
    algorithm = mlir::enzymexla::QrAlgorithm::geqrt;
  }
  return wrap(mlir::enzymexla::QrAlgorithmAttr::get(unwrap(ctx), algorithm));
}

MlirAttribute enzymexlaSVDAlgorithmAttrGet(MlirContext ctx, int32_t mode) {
  mlir::enzymexla::SVDAlgorithm algorithm;
  switch (mode) {
  case 0:
    algorithm = mlir::enzymexla::SVDAlgorithm::DEFAULT;
    break;
  case 1:
    algorithm = mlir::enzymexla::SVDAlgorithm::QRIteration;
    break;
  case 2:
    algorithm = mlir::enzymexla::SVDAlgorithm::DivideAndConquer;
    break;
  case 3:
    algorithm = mlir::enzymexla::SVDAlgorithm::Jacobi;
    break;
  }
  return wrap(mlir::enzymexla::SVDAlgorithmAttr::get(unwrap(ctx), algorithm));
}

MlirAttribute enzymexlaGeluApproximationAttrGet(MlirContext ctx, int32_t mode) {
  mlir::enzymexla::GeluApproximation approximation;
  switch (mode) {
  case 0:
    approximation = mlir::enzymexla::GeluApproximation::NONE;
    break;
  case 1:
    approximation = mlir::enzymexla::GeluApproximation::TANH;
    break;
  case 2:
    approximation = mlir::enzymexla::GeluApproximation::SIGMOID;
  }
  return wrap(
      mlir::enzymexla::GeluApproximationAttr::get(unwrap(ctx), approximation));
}

MlirAttribute enzymexlaGuaranteedAnalysisAttrGet(MlirContext ctx, int32_t mode) {
  mlir::enzymexla::GuaranteedAnalysisAnalysis analysis;
  switch (mode) {
  case 0:
    analysis = mlir::enzymexla::GuaranteedAnalysisAnalysis::GUARANTEED;
    break;
  case 1:
    analysis = mlir::enzymexla::GuaranteedAnalysisAnalysis::NOTGUARANTEED;
    break;
  case 2:
    analysis = mlir::enzymexla::GuaranteedAnalysisAnalysis::UNKNOWN;
    break;
  }
  return wrap(
      mlir::enzymexla::GuaranteedAnalysisAttr::get(unwrap(ctx), analysis));
}
