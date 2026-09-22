#ifndef ENZYME_AD_JAX_PASSES_DISTRIBUTED_COLLECTIVE_PRIMITIVE_BUILDERS_H
#define ENZYME_AD_JAX_PASSES_DISTRIBUTED_COLLECTIVE_PRIMITIVE_BUILDERS_H

#include "src/enzyme_ad/jax/Dialect/Axis/Utilities.h"
#include "src/enzyme_ad/jax/Dialect/Distributed/CollectiveCost.h"
#include "src/enzyme_ad/jax/Dialect/Distributed/Dialect.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir::enzyme::distributed {

// Per-PrimitiveKind StableHLO op builders for LowerCollectivesPass
// (LowerCollectives.cpp), which replaces an already-decomposed
// distributed.Collective's CollectivePlan (CollectiveDecomposer.h) with the
// real op chain it calls for: one call below per PrimitiveStep, threading
// each step's result into the next step's operand.
//
// PrimitiveStep/StepAtom (CollectiveCost.h) are plain data with no SSA axis
// values and no tensor shape, only a byte payload count per N7 (a
// deliberate cost-model choice: payload bytes are the only size input it
// needs). The builders below therefore share a fixed convention instead of
// trying to recover a per-step tensor shape that was never part of that
// data: every builder consumes and produces a flat 1-D tensor sized by the
// step's own payloadIn/payloadOut byte count divided by the element's byte
// width, and LowerCollectivesPass reshapes to/from the real input/result
// tensor type only once, at the very start and end of a chain.
//
// Every builder wraps its op in StableHLO's async-start/async-done pattern
// (stablehlo.async_start containing the real op, immediately awaited by a
// stablehlo.async_done), matching how this dialect already represents a
// collective's asynchrony, and tags the async_start op with a
// "distributed.primitive_cost" dictionary attribute (see
// buildPrimitiveCostAttr) carrying the same per-step cost fields
// CollectiveCost.h's decomposition DP already computed, so a downstream
// scheduler can read a step's predicted cost straight off the IR instead of
// recomputing it via a second planCollective call.

// The "distributed.primitive_cost" dictionary attribute for `step`: "kind"
// (the PrimitiveKind's name, see CollectiveCost.h's toString), "payload_in"
// / "payload_out" (bytes, I64), "latency" / "transfer_time" /
// "isolated_duration" (F64), and "volume" / "rho" (F64 arrays, one entry per
// physical mesh axis, zero on axes the step does not touch; see
// PrimitiveStep's own field comments for exactly what each number means).
DictionaryAttr buildPrimitiveCostAttr(OpBuilder &builder,
                                      const PrimitiveStep &step);

// Builds a stablehlo replica-groups attribute directly from a decomposed
// PrimitiveStep's plain StepAtoms and the physical mesh's own axis types.
//
// Each atom's own stride (StepAtom's field comment, CollectiveCost.h) gives
// its place within its physical axis's mixed-radix coordinate, so an atom
// that is a genuine sub-portion of its axis (several atoms sharing one
// physical axis) is positioned correctly, not just a whole-axis atom.
Attribute
createReplicaGroupsFromStepAtoms(ArrayRef<StepAtom> atoms,
                                 ArrayRef<PhysicalCommAxisType> meshAxisTypes,
                                 OpBuilder &builder);

// Reshapes `tensor` to a flat 1-D tensor of its own element count (a no-op
// if it already is one).
Value flattenTo1D(OpBuilder &builder, Location loc, Value tensor);

// Reshapes a flat 1-D tensor to `targetType` (whose element count must
// already match; a no-op if the types already match).
Value reshapeFrom1D(OpBuilder &builder, Location loc, Value flat,
                    RankedTensorType targetType);

// One chain-materialization builder per PrimitiveKind that moves data.
// `operand` and every builder's result are flat 1-D tensors, per the
// file-level comment above; the result is the stablehlo.async_done value
// (i.e. already awaited), ready to feed the next step or the chain's final
// reshape.

Value buildAllReduceStep(OpBuilder &builder, Location loc, Value operand,
                         const PrimitiveStep &step,
                         ArrayRef<PhysicalCommAxisType> meshAxisTypes,
                         Region &reductionBody);

Value buildReduceScatterStep(OpBuilder &builder, Location loc, Value operand,
                             const PrimitiveStep &step,
                             ArrayRef<PhysicalCommAxisType> meshAxisTypes,
                             Region &reductionBody);

Value buildAllGatherStep(OpBuilder &builder, Location loc, Value operand,
                         const PrimitiveStep &step,
                         ArrayRef<PhysicalCommAxisType> meshAxisTypes);

Value buildAllToAllStep(OpBuilder &builder, Location loc, Value operand,
                        const PrimitiveStep &step,
                        ArrayRef<PhysicalCommAxisType> meshAxisTypes);

// Builds any permute shape planCollective can produce: a two-atom swap
// (D16's conjugation), or a longer cycle / path closure (D5, D15). A
// PrimitiveStep's `atoms` alone don't say which atom's digit flows to which;
// that wiring lives in the collective's normal form (MeshAtom::in/out),
// discarded by the time a flat StepAtom exists. So a step with more than
// two atoms re-derives it via a second normalizeCollective call on
// `collective` (cheap and expected to always succeed: planCollective already
// normalized the same collective once). Every atom must have equal extent
// (asserted).
Value buildPermuteStep(OpBuilder &builder, Location loc, Value operand,
                       const PrimitiveStep &step,
                       ArrayRef<PhysicalCommAxisType> meshAxisTypes,
                       DistributedCollectiveOp collective);

// Device-local slice: no StableHLO collective inside the async region, just
// a per-device dynamic offset computed by decoding the atom's own digit out
// of stablehlo.partition_id (StepAtom's stride locates it within its
// physical axis, same as createReplicaGroupsFromStepAtoms). Still
// async-wrapped like every other step, for a uniform chain shape; it costs
// nothing (CollectiveCost.h's localSliceFootprint), so the wrapping is free
// of any real latency in practice.
Value buildLocalSliceStep(OpBuilder &builder, Location loc, Value operand,
                          const PrimitiveStep &step,
                          ArrayRef<PhysicalCommAxisType> meshAxisTypes);

} // namespace mlir::enzyme::distributed

#endif // ENZYME_AD_JAX_PASSES_DISTRIBUTED_COLLECTIVE_PRIMITIVE_BUILDERS_H
