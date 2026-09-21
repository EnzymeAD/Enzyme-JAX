#ifndef ENZYME_AD_JAX_DIALECT_DISTRIBUTED_COMPUTE_COST_H
#define ENZYME_AD_JAX_DIALECT_DISTRIBUTED_COMPUTE_COST_H

#include "MeshCostMetadata.h"

#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/STLFunctionalExtras.h"

#include <cstddef>
#include <optional>

namespace mlir::enzyme::distributed {

// Per-device compute cost of a lowered module: a roofline time for each
// stablehlo op inside a distributed.DistributedKernel body, summed.
//
// Compute model
// -------------
// The model can be altered assumption by assumption.
//   R1  Per-op roofline: an op takes max(flops / F, bytes / M), where F and M
//       are the device's peak FLOP and memory bytes per time unit
//       (DeviceCostParams). Its flops depend on the op kind (table below).
//   R2  Kernel bodies are at per-device local shapes, which the lowering
//       pipeline establishes (kernels have no partitioning axes). The program
//       is SPMD, so the time of one device is the time of all, and each
//       kernel runs once.
//   R3  bytes = bytes of all operands + bytes of all results of the op. Each
//       value is read once per use and written once; intermediate values are
//       never assumed to stay on chip.
//   R4  No fusion: every op is costed alone, so a chain of elementwise ops
//       pays for each intermediate's memory traffic.
//   R5  Compute and communication do not overlap: the compute total is added
//       to the collective total.
//   R6  Shapes are static. An op with a non-static-shaped or non-tensor
//       operand or result is skipped and counted.
//   R7  An element occupies ceil(bit width / 8) bytes (complex: twice its
//       component). Types without a bit width (opaque, tokens) are skipped
//       and counted like dynamic shapes.
//   R8  Only the top-level ops of a kernel body, other than its terminator,
//       are costed. An op's regions (reduction bodies, control flow) are not
//       entered.
//
// Flops per op (result and operand sizes are element counts; T is
// kTranscendentalWeight):
//   dot_general               2 * numel(result) * prod(contracting dims)
//   convolution               2 * numel(result) * numel(kernel) / C_out,
//                             i.e. prod(kernel spatial) * C_in / groups per
//                             output element (batch_group_count is ignored)
//   reduce, reduce_window     sum of operand numels (the combiner is one flop
//                             per input element; init values are scalars)
//   elementwise arithmetic    numel(result): add, subtract, multiply, divide,
//     and comparison          remainder, negate, abs, sign, floor, ceil,
//                             round_*, maximum, minimum, and, or, xor, not,
//                             shifts, popcnt, compare, select, clamp,
//                             is_finite, real, imag, complex
//   transcendental            T * numel(result): exponential,
//                             exponential_minus_one, log, log_plus_one, tanh,
//                             logistic, sqrt, rsqrt, cbrt, power, sine, cosine,
//                             tan, atan2
//   data movement, 0 flops    transpose, broadcast_in_dim, slice,
//                             dynamic_slice, dynamic_update_slice,
//                             concatenate, pad, gather, scatter, convert,
//                             reverse, copy
//   free (0 flops, 0 bytes)   reshape, bitcast_convert, constant, iota, tuple,
//                             get_tuple_element, optimization_barrier
//   collectives               all_reduce, all_gather, reduce_scatter,
//                             all_to_all, collective_permute,
//                             collective_broadcast: communication that the
//                             collective model does not see. Costed like an
//                             unknown op and counted separately.
//   anything else             unknown: bytes only, and counted.
//
// The weights are meant to be coarse. Only the relative size of compute,
// memory traffic and communication is meaningful.

// Flops of one transcendental element relative to one arithmetic flop.
constexpr double kTranscendentalWeight = 1.0;

// Cost of one op.
struct OpCost {
  double flops = 0;
  double bytes = 0;
  // The op's kind was not in the table: only its bytes are costed.
  bool unknown = false;
  // A stablehlo collective inside a kernel body. Also `unknown`.
  bool collective = false;

  double time(const DeviceCostParams &device) const;
};

// Costs `op`, or returns nullopt if it has a dynamic shape or a type without a
// static size (R6, R7).
std::optional<OpCost> costOp(Operation *op);

struct ComputeCostSummary {
  // Sum of the per-op roofline times.
  double total = 0;
  size_t numKernels = 0;
  // Ops costed, including unknown ones.
  size_t numOps = 0;
  // Ops costed by bytes alone.
  size_t numUnknownOps = 0;
  // Stablehlo collectives inside kernel bodies (a subset of the unknown ops).
  size_t numCollectiveOps = 0;
  // Ops skipped for a dynamic shape or unsized type.
  size_t numSkippedOps = 0;
};

// Sums the roofline time of every op in every distributed.DistributedKernel
// body of `module`. If `visitor` is given it is called for each costed op with
// its cost, in walk order.
ComputeCostSummary summarizeKernelCompute(
    ModuleOp module, const DeviceCostParams &device,
    llvm::function_ref<void(Operation *, const OpCost &)> visitor = nullptr);

} // namespace mlir::enzyme::distributed

#endif // ENZYME_AD_JAX_DIALECT_DISTRIBUTED_COMPUTE_COST_H
