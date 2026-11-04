#ifndef ENZYME_AD_JAX_PASSES_DISTRIBUTED_COLLECTIVE_PRIMITIVE_BUILDERS_H
#define ENZYME_AD_JAX_PASSES_DISTRIBUTED_COLLECTIVE_PRIMITIVE_BUILDERS_H

#include "src/enzyme_ad/jax/Dialect/Axis/Utilities.h"
#include "src/enzyme_ad/jax/Dialect/Distributed/CollectiveCost.h"
#include "src/enzyme_ad/jax/Dialect/Distributed/Dialect.h"
#include "src/enzyme_ad/jax/Dialect/Distributed/NormalizedCollective.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir::enzyme::distributed {

// Per-PrimitiveKind StableHLO op builders for LowerCollectivesPass
// (LowerCollectives.cpp), which replaces an already-decomposed
// distributed.Collective's CollectivePlan (CollectiveDecomposer.h) with the
// real op chain it calls for: one call below per PrimitiveStep, threading
// each step's result into the next step's operand.
//
// The chain's tensor is in digit shape: one dimension per local digit of the
// chain's ChainDigitModel (CollectiveDecomposer.h), in its order, with
// extent-1 dimensions inserted or dropped by free reshapes (reshapeToDigits).
// Each builder takes that operand and the dimension(s) its step acts on, so a
// step slices, scatters, gathers or exchanges exactly the digit the
// collective's map names:
//   - a local slice, a reduce-scatter and the split half of an all-to-all
//     shrink the dimension of the digit they place by the atom's extent;
//   - an all-gather and the concat half of an all-to-all grow an extent-1
//     dimension the caller inserted where the gathered digit belongs;
//   - an all-reduce and a permute keep the operand's shape;
//   - a peel (CollectiveDecomposer.h's D17) scatters and later gathers the
//     dimension of a digit that stays local in between, shrinking it to the
//     device's shard and growing it back.
// Group members are ordered by the atom's coordinate
// (createReplicaGroupsFromStepAtoms), so a grown dimension's index is the
// coordinate of the device the data came from, and of a shrunk dimension each
// device keeps the piece at its own coordinate.
//
// Every builder emits its step as one plain StableHLO op and tags it with a
// "distributed.primitive_cost" dictionary attribute (see
// buildPrimitiveCostAttr) carrying the same per-step cost fields
// CollectiveCost.h's decomposition DP already computed, so a downstream
// scheduler can read a step's predicted cost straight off the IR instead of
// recomputing it via a second planCollective call. The ops are synchronous in
// the IR: the schedule simulator models a step's overlap from its
// dependencies, and a backend schedules the asynchrony itself.

// The "distributed.primitive_cost" dictionary attribute for `step`: "kind"
// (the PrimitiveKind's name, see CollectiveCost.h's toString), "payload_in"
// / "payload_out" (bytes, I64), "latency" / "transfer_time" /
// "isolated_duration" (F64), and "volume" / "rho" (F64 arrays, one entry per
// physical mesh axis, zero on axes the step does not touch; see
// PrimitiveStep's own field comments for exactly what each number means).
DictionaryAttr buildPrimitiveCostAttr(OpBuilder &builder,
                                      const PrimitiveStep &step);

// Builds explicit stablehlo replica groups of global device ids for a
// decomposed PrimitiveStep: one group per assignment of the device-id digits
// outside `atoms`, its members enumerating the atoms with the first most
// significant. For example, on one physical axis <4, 1>, the atom of extent 2
// and stride 2 gives [[0, 2], [1, 3]].
//
// Each atom's stride (StepAtom's field comment, CollectiveCost.h) places it
// within its physical axis's coordinate, so an atom that is a sub-portion of
// its axis is a device-id digit like a whole-axis atom.
DenseIntElementsAttr
createReplicaGroupsFromStepAtoms(ArrayRef<StepAtom> atoms,
                                 ArrayRef<PhysicalCommAxisType> meshAxisTypes,
                                 OpBuilder &builder);

// Reshapes `tensor` to `shape`, which regroups the same row-major elements
// (a no-op if the type already has that shape).
Value reshapeToDigits(OpBuilder &builder, Location loc, Value tensor,
                      ArrayRef<int64_t> shape);

// One chain-materialization builder per PrimitiveKind that moves data, on a
// digit-shaped operand (see the file comment). The result is the cost-tagged
// op's own result, ready to feed the next step.

Value buildAllReduceStep(OpBuilder &builder, Location loc, Value operand,
                         const PrimitiveStep &step,
                         ArrayRef<PhysicalCommAxisType> meshAxisTypes,
                         Region &reductionBody);

// Divides `dimension` by the atom's extent.
Value buildReduceScatterStep(OpBuilder &builder, Location loc, Value operand,
                             const PrimitiveStep &step, int64_t dimension,
                             ArrayRef<PhysicalCommAxisType> meshAxisTypes,
                             Region &reductionBody);

// Multiplies `dimension` by the atom's extent.
Value buildAllGatherStep(OpBuilder &builder, Location loc, Value operand,
                         const PrimitiveStep &step, int64_t dimension,
                         ArrayRef<PhysicalCommAxisType> meshAxisTypes);

// Divides `splitDimension` and multiplies `concatDimension` by the atom's
// extent.
Value buildAllToAllStep(OpBuilder &builder, Location loc, Value operand,
                        const PrimitiveStep &step, int64_t splitDimension,
                        int64_t concatDimension,
                        ArrayRef<PhysicalCommAxisType> meshAxisTypes);

// Builds any permute shape planCollective can produce: a two-atom swap
// (D16's conjugation), or a longer cycle / path closure (D5, D15). A
// PrimitiveStep's `atoms` alone don't say which atom's digit flows to which;
// for a step with more than two atoms that wiring is read from the
// collective's normal form (MeshAtom::in/out). Every atom must have equal
// extent (asserted).
Value buildPermuteStep(OpBuilder &builder, Location loc, Value operand,
                       const PrimitiveStep &step,
                       ArrayRef<PhysicalCommAxisType> meshAxisTypes,
                       const NormalizedCollective &normalized);

// Device-local slice: no StableHLO collective, just a stablehlo.dynamic_slice
// of `dimension` at a per-device offset computed by decoding the atom's own
// digit out of stablehlo.partition_id (StepAtom's stride locates it within
// its physical axis, same as createReplicaGroupsFromStepAtoms). Divides
// `dimension` by the atom's extent. It is cost-tagged like every other step,
// for a uniform chain shape; it costs nothing (CollectiveCost.h's
// localSliceFootprint).
Value buildLocalSliceStep(OpBuilder &builder, Location loc, Value operand,
                          const PrimitiveStep &step, int64_t dimension,
                          ArrayRef<PhysicalCommAxisType> meshAxisTypes);

} // namespace mlir::enzyme::distributed

#endif // ENZYME_AD_JAX_PASSES_DISTRIBUTED_COLLECTIVE_PRIMITIVE_BUILDERS_H
