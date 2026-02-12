#ifndef ENZYMEXLA_INTEGRATIONS_C_ENZYMEXLAOPTPASSES_H_
#define ENZYMEXLA_INTEGRATIONS_C_ENZYMEXLAOPTPASSES_H_

#include <stdbool.h>
#include <stdint.h>

#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

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
MLIR_CAPI_EXPORTED void enzymexlaGetTransformPassesList(
    const EnzymeXLATransformPassesOptions *options, char **mainPasses,
    char **lowerPasses);

/// Free a string returned by enzymexlaGetTransformPassesList.
MLIR_CAPI_EXPORTED void enzymexlaFreeTransformPassesList(char *passes);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // ENZYMEXLA_INTEGRATIONS_C_ENZYMEXLAOPTPASSES_H_
