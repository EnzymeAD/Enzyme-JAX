#ifndef ENZYMEXLA_INTEGRATIONS_C_ENZYMEXLA_H_
#define ENZYMEXLA_INTEGRATIONS_C_ENZYMEXLA_H_

#include <stdbool.h>
#include <stdint.h>
#include <sys/types.h>

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#include "src/enzyme_ad/jax/Dialect/Ops.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// Linear Algebra Ops
//===----------------------------------------------------------------------===//

typedef enum {
  ENZYMEXLA_LAPACK_LAYOUT_COLUMN_MAJOR = 0,
  ENZYMEXLA_LAPACK_LAYOUT_ROW_MAJOR = 1,
} EnzymeXlaLapackLayout;

MLIR_CAPI_EXPORTED MlirAttribute
enzymexlaLapackLayoutAttrGet(MlirContext ctx, EnzymeXlaLapackLayout layout);

typedef enum {
  ENZYMEXLA_LAPACK_TRANSPOSE_NONE = 0,
  ENZYMEXLA_LAPACK_TRANSPOSE_TRANSPOSE = 1,
  ENZYMEXLA_LAPACK_TRANSPOSE_CONJUGATE_TRANSPOSE = 2,
} EnzymeXlaLapackTranspose;

MLIR_CAPI_EXPORTED MlirAttribute enzymexlaLapackTransposeAttrGet(
    MlirContext ctx, EnzymeXlaLapackTranspose transpose);

typedef enum {
  ENZYMEXLA_LAPACK_SIDE_LEFT = 0,
  ENZYMEXLA_LAPACK_SIDE_RIGHT = 1,
} EnzymeXlaLapackSide;

MLIR_CAPI_EXPORTED MlirAttribute
enzymexlaLapackSideAttrGet(MlirContext ctx, EnzymeXlaLapackSide side);

typedef enum {
  ENZYMEXLA_LAPACK_UPLO_LOWER = 0,
  ENZYMEXLA_LAPACK_UPLO_UPPER = 1,
  ENZYMEXLA_LAPACK_UPLO_FULL = 2,
} EnzymeXlaLapackUplo;

MLIR_CAPI_EXPORTED MlirAttribute
enzymexlaLapackUploAttrGet(MlirContext ctx, EnzymeXlaLapackUplo uplo);

typedef enum {
  ENZYMEXLA_QR_ALGORITHM_NONE = 0,
  ENZYMEXLA_QR_ALGORITHM_HOUSEHOLDER = 1,
} EnzymeXlaQRAlgorithm;

MLIR_CAPI_EXPORTED MlirAttribute
enzymexlaQRAlgorithmAttrGet(MlirContext ctx, EnzymeXlaQRAlgorithm algorithm);

typedef enum {
  ENZYMEXLA_SVD_ALGORITHM_NONE = 0,
  ENZYMEXLA_SVD_ALGORITHM_QRITERATION = 1,
  ENZYMEXLA_SVD_ALGORITHM_DIVIDEANDCONQUER = 2,
  ENZYMEXLA_SVD_ALGORITHM_JACOBI = 3,
} EnzymeXlaSVDAlgorithm;

MLIR_CAPI_EXPORTED MlirAttribute
enzymexlaSVDAlgorithmAttrGet(MlirContext ctx, EnzymeXlaSVDAlgorithm algorithm);

//===----------------------------------------------------------------------===//
// Machine Learning Ops
//===----------------------------------------------------------------------===//

typedef enum {
  ENZYMEXLA_GELU_APPROXIMATION_NONE = 0,
  ENZYMEXLA_GELU_APPROXIMATION_TANH = 1,
  ENZYMEXLA_GELU_APPROXIMATION_SIGMOID = 2,
} EnzymeXlaGeluApproximation;

MLIR_CAPI_EXPORTED MlirAttribute enzymexlaGeluApproximationAttrGet(
    MlirContext ctx, EnzymeXlaGeluApproximation approximation);

//===----------------------------------------------------------------------===//
// MPI Ops
//===----------------------------------------------------------------------===//

typedef enum {
  ENZYMEXLA_MPI_DATATYPE_NULL = 0,
  ENZYMEXLA_MPI_INT8_T = 1,
  ENZYMEXLA_MPI_UINT8_T = 2,
  ENZYMEXLA_MPI_INT16_T = 3,
  ENZYMEXLA_MPI_UINT16_T = 4,
  ENZYMEXLA_MPI_INT32_T = 5,
  ENZYMEXLA_MPI_UINT32_T = 6,
  ENZYMEXLA_MPI_INT64_T = 7,
  ENZYMEXLA_MPI_UINT64_T = 8,
  ENZYMEXLA_MPI_BYTE = 9,
  ENZYMEXLA_MPI_SHORT = 10,
  ENZYMEXLA_MPI_UNSIGNED_SHORT = 11,
  ENZYMEXLA_MPI_INT = 12,
  ENZYMEXLA_MPI_UNSIGNED = 13,
  ENZYMEXLA_MPI_LONG = 14,
  ENZYMEXLA_MPI_UNSIGNED_LONG = 15,
  ENZYMEXLA_MPI_LONG_LONG_INT = 16,
  ENZYMEXLA_MPI_UNSIGNED_LONG_LONG = 17,
  ENZYMEXLA_MPI_CHAR = 18,
  ENZYMEXLA_MPI_SIGNED_CHAR = 19,
  ENZYMEXLA_MPI_UNSIGNED_CHAR = 20,
  ENZYMEXLA_MPI_WCHAR = 21,
  ENZYMEXLA_MPI_FLOAT = 22,
  ENZYMEXLA_MPI_DOUBLE = 23,
  ENZYMEXLA_MPI_C_FLOAT_COMPLEX = 24,
  ENZYMEXLA_MPI_C_DOUBLE_COMPLEX = 25,
  ENZYMEXLA_MPI_C_BOOL = 26,
} EnzymeXlaMPIDatatype;

MLIR_CAPI_EXPORTED MlirAttribute
enzymexlaMPIDatatypeAttrGet(MlirContext ctx, EnzymeXlaMPIDatatype datatype);

typedef enum {
  ENZYMEXLA_MPI_OP_NULL = 0,
  ENZYMEXLA_MPI_BAND = 1,
  ENZYMEXLA_MPI_BOR = 2,
  ENZYMEXLA_MPI_BXOR = 3,
  ENZYMEXLA_MPI_LAND = 4,
  ENZYMEXLA_MPI_LOR = 5,
  ENZYMEXLA_MPI_LXOR = 6,
  ENZYMEXLA_MPI_MAX = 7,
  ENZYMEXLA_MPI_MIN = 8,
  ENZYMEXLA_MPI_PROD = 9,
  ENZYMEXLA_MPI_REPLACE = 10,
  ENZYMEXLA_MPI_SUM = 11,
  ENZYMEXLA_MPI_NO_OP = 12,
} EnzymeXlaMPIOp;

MLIR_CAPI_EXPORTED MlirAttribute enzymexlaMPIOpAttrGet(MlirContext ctx,
                                                       EnzymeXlaMPIOp op);

//===----------------------------------------------------------------------===//
// Other Ops / Attributes
//===----------------------------------------------------------------------===//

typedef enum {
  ENZYMEXLA_GUARANTEED_ANALYSIS_RESULT_GUARANTEED = 0,
  ENZYMEXLA_GUARANTEED_ANALYSIS_RESULT_NOTGUARANTEED = 1,
  ENZYMEXLA_GUARANTEED_ANALYSIS_RESULT_UNKNOWN = 2,
} EnzymeXlaGuaranteedAnalysisResult;

MLIR_CAPI_EXPORTED MlirAttribute enzymexlaGuaranteedAnalysisResultAttrGet(
    MlirContext ctx, EnzymeXlaGuaranteedAnalysisResult result);

//===----------------------------------------------------------------------===//
// Compile Options for optimization pass generation
//===----------------------------------------------------------------------===//

/// Enum for propagation direction (reshape/transpose).
typedef enum {
  ENZYMEXLA_PROPAGATE_NONE = 0,
  ENZYMEXLA_PROPAGATE_UP = 1,
  ENZYMEXLA_PROPAGATE_DOWN = 2,
} EnzymeXLAPropagateDirection;

/// Options that control which transform passes are generated.
typedef struct {
  int64_t max_constant_threshold;
  int64_t while_unroll_threshold;

  // Propagation directions
  EnzymeXLAPropagateDirection reshape_propagate;
  EnzymeXLAPropagateDirection transpose_propagate;

  // Feature flags
  bool no_nan;
  bool all_finite;
  bool dus_to_concat;
  bool dus_slice_simplify;
  bool sum_to_reducewindow;
  bool sum_to_conv;
  bool aggressive_sum_to_conv;
  bool while_concat;
  bool aggressive_propagation;

  // Sharding / BLAS-LAPACK flags
  bool is_sharded;
  bool raise_shlo_to_blas_lapack;

  // Communication flags
  bool recognize_comms;
  bool lower_comms;

  // Enable flags for optional pass groups
  bool enable_self_to_convolution_like_passes;
  bool enable_structured_tensors_detection_passes;
  bool enable_structured_tensors_passes;
  bool enable_scatter_gather_optimization_passes;
  bool enable_slice_to_batch_passes;
  bool enable_reduce_slice_fusion_passes;
  bool enable_concat_to_batch_passes;
  bool enable_loop_raising_passes;
  bool enable_licm_optimization_passes;
  bool enable_pad_optimization_passes;
} EnzymeXLATransformPassesOptions;

//===----------------------------------------------------------------------===//
// Transform pass list generation
//===----------------------------------------------------------------------===//

/// Returns the transform passes list as a semicolon-separated string.
/// The caller must free the returned string using
/// enzymexlaFreeTransformPassesList.
///
/// Two separate lists are produced:
///   - `mainPasses`: the primary transform pass list
///   - `lowerPasses`: the lowering transform pass list (for lower_comms)
///
/// Each is returned as a semicolon-separated string of pass patterns.
MLIR_CAPI_EXPORTED void
enzymexlaGetTransformPassesList(const EnzymeXLATransformPassesOptions *options,
                                char **mainPasses, char **lowerPasses);

/// Free a string returned by enzymexlaGetTransformPassesList.
MLIR_CAPI_EXPORTED void enzymexlaFreeTransformPassesList(char *passes);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // ENZYMEXLA_INTEGRATIONS_C_ENZYMEXLA_H_
