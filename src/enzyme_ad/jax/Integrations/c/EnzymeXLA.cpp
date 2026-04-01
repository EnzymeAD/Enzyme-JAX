#include "EnzymeXLA.h"

#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"

#include "src/enzyme_ad/jax/Dialect/Dialect.h"
#include "src/enzyme_ad/jax/Dialect/Ops.h"

MlirAttribute blasLayoutAttrGet(MlirContext ctx, uint8_t col_major) {
  mlir::blas::BlasLayout layout;
  if (col_major) {
    layout = mlir::blas::BlasLayout::col_major;
  } else {
    layout = mlir::blas::BlasLayout::row_major;
  }
  return wrap(mlir::blas::BlasLayoutAttr::get(unwrap(ctx), layout));
}

MlirAttribute blasTransposeAttrGet(MlirContext ctx, int32_t mode) {
  mlir::blas::BlasTranspose transpose;
  switch (mode) {
  case 0:
    transpose = mlir::blas::BlasTranspose::none;
    break;
  case 1:
    transpose = mlir::blas::BlasTranspose::transpose;
    break;
  case 2:
    transpose = mlir::blas::BlasTranspose::adjoint;
  }
  return wrap(mlir::blas::BlasTransposeAttr::get(unwrap(ctx), transpose));
}

MlirAttribute blasSideAttrGet(MlirContext ctx, uint8_t left_side) {
  mlir::blas::BlasSide side;
  if (left_side) {
    side = mlir::blas::BlasSide::left;
  } else {
    side = mlir::blas::BlasSide::right;
  }
  return wrap(mlir::blas::BlasSideAttr::get(unwrap(ctx), side));
}

MlirAttribute blasUploAttrGet(MlirContext ctx, int32_t mode) {
  mlir::blas::BlasUplo uplo;
  switch (mode) {
  case 0:
    uplo = mlir::blas::BlasUplo::upper;
    break;
  case 1:
    uplo = mlir::blas::BlasUplo::lower;
    break;
  case 2:
    uplo = mlir::blas::BlasUplo::any;
    break;
  }
  return wrap(mlir::blas::BlasUploAttr::get(unwrap(ctx), uplo));
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

MlirAttribute enzymexlaGuaranteedAnalysisResultAttrGet(MlirContext ctx,
                                                       int32_t mode) {
  mlir::enzymexla::GuaranteedAnalysisResult analysis;
  switch (mode) {
  case 0:
    analysis = mlir::enzymexla::GuaranteedAnalysisResult::GUARANTEED;
    break;
  case 1:
    analysis = mlir::enzymexla::GuaranteedAnalysisResult::NOTGUARANTEED;
    break;
  case 2:
    analysis = mlir::enzymexla::GuaranteedAnalysisResult::UNKNOWN;
    break;
  }
  return wrap(mlir::enzymexla::GuaranteedAnalysisResultAttr::get(unwrap(ctx),
                                                                 analysis));
}

MlirAttribute enzymexlaMPIDatatypeAttrGet(MlirContext ctx, int32_t mode) {
  mlir::enzymexla::MPIDatatype datatype;
  switch (mode) {
  case 0:
    datatype = mlir::enzymexla::MPIDatatype::MPI_DATATYPE_NULL;
    break;
  case 1:
    datatype = mlir::enzymexla::MPIDatatype::MPI_INT8_T;
    break;
  case 2:
    datatype = mlir::enzymexla::MPIDatatype::MPI_UINT8_T;
    break;
  case 3:
    datatype = mlir::enzymexla::MPIDatatype::MPI_INT16_T;
    break;
  case 4:
    datatype = mlir::enzymexla::MPIDatatype::MPI_UINT16_T;
    break;
  case 5:
    datatype = mlir::enzymexla::MPIDatatype::MPI_INT32_T;
    break;
  case 6:
    datatype = mlir::enzymexla::MPIDatatype::MPI_UINT32_T;
    break;
  case 7:
    datatype = mlir::enzymexla::MPIDatatype::MPI_INT64_T;
    break;
  case 8:
    datatype = mlir::enzymexla::MPIDatatype::MPI_UINT64_T;
    break;
  case 9:
    datatype = mlir::enzymexla::MPIDatatype::MPI_BYTE;
    break;
  case 10:
    datatype = mlir::enzymexla::MPIDatatype::MPI_SHORT;
    break;
  case 11:
    datatype = mlir::enzymexla::MPIDatatype::MPI_UNSIGNED_SHORT;
    break;
  case 12:
    datatype = mlir::enzymexla::MPIDatatype::MPI_INT;
    break;
  case 13:
    datatype = mlir::enzymexla::MPIDatatype::MPI_UNSIGNED;
    break;
  case 14:
    datatype = mlir::enzymexla::MPIDatatype::MPI_LONG;
    break;
  case 15:
    datatype = mlir::enzymexla::MPIDatatype::MPI_UNSIGNED_LONG;
    break;
  case 16:
    datatype = mlir::enzymexla::MPIDatatype::MPI_LONG_LONG_INT;
    break;
  case 17:
    datatype = mlir::enzymexla::MPIDatatype::MPI_UNSIGNED_LONG_LONG;
    break;
  case 18:
    datatype = mlir::enzymexla::MPIDatatype::MPI_CHAR;
    break;
  case 19:
    datatype = mlir::enzymexla::MPIDatatype::MPI_SIGNED_CHAR;
    break;
  case 20:
    datatype = mlir::enzymexla::MPIDatatype::MPI_UNSIGNED_CHAR;
    break;
  case 21:
    datatype = mlir::enzymexla::MPIDatatype::MPI_WCHAR;
    break;
  case 22:
    datatype = mlir::enzymexla::MPIDatatype::MPI_FLOAT;
    break;
  case 23:
    datatype = mlir::enzymexla::MPIDatatype::MPI_DOUBLE;
    break;
  case 24:
    datatype = mlir::enzymexla::MPIDatatype::MPI_C_FLOAT_COMPLEX;
    break;
  case 25:
    datatype = mlir::enzymexla::MPIDatatype::MPI_C_DOUBLE_COMPLEX;
    break;
  case 26:
    datatype = mlir::enzymexla::MPIDatatype::MPI_C_BOOL;
    break;
  default:
    llvm_unreachable("Invalid MPI datatype mode");
  }
  return wrap(mlir::enzymexla::MPIDatatypeAttr::get(unwrap(ctx), datatype));
}

MlirAttribute enzymexlaMPIOpAttrGet(MlirContext ctx, int32_t mode) {
  mlir::enzymexla::MPIOp op;
  switch (mode) {
  case 0:
    op = mlir::enzymexla::MPIOp::MPI_OP_NULL;
    break;
  case 1:
    op = mlir::enzymexla::MPIOp::MPI_BAND;
    break;
  case 2:
    op = mlir::enzymexla::MPIOp::MPI_BOR;
    break;
  case 3:
    op = mlir::enzymexla::MPIOp::MPI_BXOR;
    break;
  case 4:
    op = mlir::enzymexla::MPIOp::MPI_LAND;
    break;
  case 5:
    op = mlir::enzymexla::MPIOp::MPI_LOR;
    break;
  case 6:
    op = mlir::enzymexla::MPIOp::MPI_LXOR;
    break;
  case 7:
    op = mlir::enzymexla::MPIOp::MPI_MAX;
    break;
  case 8:
    op = mlir::enzymexla::MPIOp::MPI_MIN;
    break;
  case 9:
    op = mlir::enzymexla::MPIOp::MPI_PROD;
    break;
  case 10:
    op = mlir::enzymexla::MPIOp::MPI_REPLACE;
    break;
  case 11:
    op = mlir::enzymexla::MPIOp::MPI_SUM;
    break;
  case 12:
    op = mlir::enzymexla::MPIOp::MPI_NO_OP;
    break;
  default:
    llvm_unreachable("Invalid MPI op mode");
  }
  return wrap(mlir::enzymexla::MPIOpAttr::get(unwrap(ctx), op));
}
