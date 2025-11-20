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

MLIR_CAPI_EXPORTED MlirAttribute
enzymexlaLapackLayoutAttrGet(MlirContext ctx, uint8_t col_major);

MLIR_CAPI_EXPORTED MlirAttribute
enzymexlaLapackTransposeAttrGet(MlirContext ctx, int32_t mode);

MLIR_CAPI_EXPORTED MlirAttribute enzymexlaLapackSideAttrGet(MlirContext ctx,
                                                            uint8_t left_side);

MLIR_CAPI_EXPORTED MlirAttribute enzymexlaLapackUploAttrGet(MlirContext ctx,
                                                            uint8_t up);

MLIR_CAPI_EXPORTED MlirAttribute enzymexlaQRAlgorithmAttrGet(MlirContext ctx,
                                                             int32_t mode);

MLIR_CAPI_EXPORTED MlirAttribute enzymexlaSVDAlgorithmAttrGet(MlirContext ctx,
                                                              int32_t mode);

//===----------------------------------------------------------------------===//
// Machine Learning Ops
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED MlirAttribute
enzymexlaGeluApproximationAttrGet(MlirContext ctx, int32_t mode);

//===----------------------------------------------------------------------===//
// Other Ops / Attributes
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED MlirAttribute enzymexlaGuaranteedAnalysisAttrGet(
    MlirContext ctx, int32_t mode);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // ENZYMEXLA_INTEGRATIONS_C_ENZYMEXLA_H_
