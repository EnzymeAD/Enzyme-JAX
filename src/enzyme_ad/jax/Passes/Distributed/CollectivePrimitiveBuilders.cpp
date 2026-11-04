#include "src/enzyme_ad/jax/Passes/Distributed/CollectivePrimitiveBuilders.h"

#include "src/enzyme_ad/jax/Dialect/Distributed/NormalizedCollective.h"
#include "src/enzyme_ad/jax/Dialect/Distributed/Utilities.h"

#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ErrorHandling.h"

#include <atomic>
#include <optional>
#include <string>

namespace mlir::enzyme::distributed {

namespace {

int64_t nextStablehloChannelId() {
  // StableHLO channel ids are global within this process and independent of
  // which communication op kind requests the next id.
  static std::atomic<int64_t> nextChannelId{1};
  return nextChannelId.fetch_add(1, std::memory_order_relaxed);
}

stablehlo::ChannelHandleAttr
createUniqueStablehloChannelHandle(MLIRContext *ctx) {
  // StableHLO collectives use channel type 0.
  constexpr int64_t kCollectiveChannelType = 0;
  return stablehlo::ChannelHandleAttr::get(ctx, nextStablehloChannelId(),
                                           kCollectiveChannelType);
}

// Tags `stepOp`, the one op a lowered PrimitiveStep becomes, with the step's
// cost (buildPrimitiveCostAttr) and returns its result.
Value tagStepCost(OpBuilder &builder, Operation *stepOp,
                  const PrimitiveStep &step) {
  stepOp->setAttr("distributed.primitive_cost",
                  buildPrimitiveCostAttr(builder, step));
  return stepOp->getResult(0);
}

// See CollectivePrimitiveBuilders.h's file comment: element counts are
// derived from a step's own payload byte counts, not re-derived from group
// size, since the decomposer already accounts for group size when computing
// payloadIn/payloadOut (CollectiveDecomposer.h's D3).
int64_t elementBytesOf(Type elementType) {
  return (elementType.getIntOrFloatBitWidth() + 7) / 8;
}

} // namespace

DictionaryAttr buildPrimitiveCostAttr(OpBuilder &builder,
                                      const PrimitiveStep &step) {
  SmallVector<NamedAttribute> fields;
  fields.push_back(
      builder.getNamedAttr("kind", builder.getStringAttr(toString(step.kind))));
  fields.push_back(builder.getNamedAttr(
      "payload_in", builder.getI64IntegerAttr(step.payloadIn)));
  fields.push_back(builder.getNamedAttr(
      "payload_out", builder.getI64IntegerAttr(step.payloadOut)));
  fields.push_back(
      builder.getNamedAttr("latency", builder.getF64FloatAttr(step.latency)));
  fields.push_back(builder.getNamedAttr(
      "transfer_time", builder.getF64FloatAttr(step.transferTime)));
  fields.push_back(builder.getNamedAttr(
      "isolated_duration", builder.getF64FloatAttr(step.isolatedDuration())));
  fields.push_back(builder.getNamedAttr(
      "volume", builder.getArrayAttr(llvm::to_vector<4>(
                    llvm::map_range(step.volume, [&](double v) -> Attribute {
                      return builder.getF64FloatAttr(v);
                    })))));
  fields.push_back(builder.getNamedAttr(
      "rho", builder.getArrayAttr(llvm::to_vector<4>(
                 llvm::map_range(step.rho, [&](double r) -> Attribute {
                   return builder.getF64FloatAttr(r);
                 })))));
  return builder.getDictionaryAttr(fields);
}

DenseIntElementsAttr
createReplicaGroupsFromStepAtoms(ArrayRef<StepAtom> atoms,
                                 ArrayRef<PhysicalCommAxisType> meshAxisTypes,
                                 OpBuilder &builder) {
  assert(!atoms.empty() &&
         "expected at least one atom for replica group creation");

  // A device id is the sum of each physical axis's coordinate times its id
  // stride, and an atom's stride locates its digit within its axis's
  // coordinate (StepAtom's field comment, CollectiveCost.h), so the atom is
  // the device-id digit of stride `atom.stride * idStride`.
  SmallVector<axis::AxisAtom> deviceIdDigits;
  deviceIdDigits.reserve(atoms.size());
  for (const StepAtom &atom : atoms) {
    PhysicalCommAxisType axisType = meshAxisTypes[atom.axis];
    deviceIdDigits.push_back(
        {atom.extent,
         atom.stride * static_cast<uint64_t>(axisType.getIdStride())});
  }
  SmallVector<SmallVector<uint64_t>> groups = axis::groupIndicesByDigits(
      getDenseDeviceCount(meshAxisTypes), deviceIdDigits);

  SmallVector<int64_t> deviceIds;
  for (ArrayRef<uint64_t> group : groups)
    llvm::append_range(deviceIds, group);
  auto groupsType =
      RankedTensorType::get({static_cast<int64_t>(groups.size()),
                             static_cast<int64_t>(groups.front().size())},
                            builder.getI64Type());
  return DenseIntElementsAttr::get(groupsType, deviceIds);
}

Value flattenTo1D(OpBuilder &builder, Location loc, Value tensor) {
  auto type = cast<RankedTensorType>(tensor.getType());
  auto flatType =
      RankedTensorType::get({type.getNumElements()}, type.getElementType());
  if (type == flatType)
    return tensor;
  return builder.create<stablehlo::ReshapeOp>(loc, flatType, tensor);
}

Value reshapeFrom1D(OpBuilder &builder, Location loc, Value flat,
                    RankedTensorType targetType) {
  if (flat.getType() == targetType)
    return flat;
  return builder.create<stablehlo::ReshapeOp>(loc, targetType, flat);
}

Value buildAllReduceStep(OpBuilder &builder, Location loc, Value operand,
                         const PrimitiveStep &step,
                         ArrayRef<PhysicalCommAxisType> meshAxisTypes,
                         Region &reductionBody) {
  auto operandType = cast<RankedTensorType>(operand.getType());
  Type elementType = operandType.getElementType();
  // All-reduce keeps the payload the same size.
  auto resultType =
      RankedTensorType::get({operandType.getNumElements()}, elementType);
  Attribute replicaGroups =
      createReplicaGroupsFromStepAtoms(step.atoms, meshAxisTypes, builder);

  auto allReduceOp = builder.create<stablehlo::AllReduceOp>(
      loc, TypeRange{resultType}, ValueRange{operand}, replicaGroups,
      createUniqueStablehloChannelHandle(builder.getContext()),
      /*use_global_device_ids=*/true);
  IRMapping mapper;
  reductionBody.cloneInto(&allReduceOp.getComputation(), mapper);
  return tagStepCost(builder, allReduceOp, step);
}

Value buildReduceScatterStep(OpBuilder &builder, Location loc, Value operand,
                             const PrimitiveStep &step,
                             ArrayRef<PhysicalCommAxisType> meshAxisTypes,
                             Region &reductionBody) {
  auto operandType = cast<RankedTensorType>(operand.getType());
  Type elementType = operandType.getElementType();
  int64_t outElems = step.payloadOut / elementBytesOf(elementType);
  auto resultType = RankedTensorType::get({outElems}, elementType);
  Attribute replicaGroups =
      createReplicaGroupsFromStepAtoms(step.atoms, meshAxisTypes, builder);

  auto reduceScatterOp = builder.create<stablehlo::ReduceScatterOp>(
      loc, resultType, operand, /*scatter_dimension=*/uint64_t{0},
      replicaGroups, createUniqueStablehloChannelHandle(builder.getContext()),
      /*use_global_device_ids=*/true);
  IRMapping mapper;
  reductionBody.cloneInto(&reduceScatterOp.getComputation(), mapper);
  return tagStepCost(builder, reduceScatterOp, step);
}

Value buildAllGatherStep(OpBuilder &builder, Location loc, Value operand,
                         const PrimitiveStep &step,
                         ArrayRef<PhysicalCommAxisType> meshAxisTypes) {
  auto operandType = cast<RankedTensorType>(operand.getType());
  Type elementType = operandType.getElementType();
  int64_t outElems = step.payloadOut / elementBytesOf(elementType);
  auto resultType = RankedTensorType::get({outElems}, elementType);
  Attribute replicaGroups =
      createReplicaGroupsFromStepAtoms(step.atoms, meshAxisTypes, builder);

  auto allGatherOp = builder.create<stablehlo::AllGatherOp>(
      loc, TypeRange{resultType}, ValueRange{operand},
      /*all_gather_dim=*/uint64_t{0}, replicaGroups,
      createUniqueStablehloChannelHandle(builder.getContext()),
      /*use_global_device_ids=*/true);
  return tagStepCost(builder, allGatherOp, step);
}

Value buildAllToAllStep(OpBuilder &builder, Location loc, Value operand,
                        const PrimitiveStep &step,
                        ArrayRef<PhysicalCommAxisType> meshAxisTypes) {
  assert(step.atoms.size() == 1 &&
         "buildAllToAllStep expects one mesh atom per step (D1)");
  int64_t n = static_cast<int64_t>(step.atoms[0].extent);

  auto operandType = cast<RankedTensorType>(operand.getType());
  assert(operandType.getNumElements() % n == 0 &&
         "all-to-all's flat payload must divide evenly by its group size");

  Attribute replicaGroups =
      createReplicaGroupsFromStepAtoms(step.atoms, meshAxisTypes, builder);

  // Splitting the flat payload into n chunks, sending chunk j to group
  // member j and concatenating what arrives in member order is exactly an
  // all-to-all on the 1-D tensor, so it needs no reshape. All-to-all keeps
  // the payload size.
  auto allToAllOp = builder.create<stablehlo::AllToAllOp>(
      loc, TypeRange{operandType}, ValueRange{operand},
      /*split_dimension=*/uint64_t{0}, /*concat_dimension=*/uint64_t{0},
      /*split_count=*/static_cast<uint64_t>(n), replicaGroups,
      createUniqueStablehloChannelHandle(builder.getContext()));
  return tagStepCost(builder, allToAllOp, step);
}

namespace {

// The mesh atom `label` names, as an index into collective.meshAtoms.
size_t meshIndexOfLabel(const NormalizedCollective &collective,
                        const AtomLabel &label) {
  for (size_t i = 0; i < collective.meshAtoms.size(); ++i)
    if (collective.meshAtoms[i].axis == label.axis &&
        collective.meshAtoms[i].atom == label.atom)
      return i;
  llvm_unreachable("a mesh partner names a mesh atom of the collective");
}

// Which atom's digit each atom in `atoms` takes when this permute step runs.
// A StepAtom is deliberately flat data (CollectiveCost.h) with no MLIR values
// or role information, so for a genuine multi-atom cycle (more than a single
// swapping pair) that wiring (which mesh atom feeds which) has to be
// re-derived from the collective's normal form rather than read off the step
// itself. This mirrors the same static closed/linked-vs-swap classification
// and source derivation CollectiveDecomposer.cpp's chain verifier
// (ChainChecker::applyPermute/applyComponentPermute/applySwap) already uses
// to check a chain realizes its collective, minus that verifier's occupant
// bookkeeping: the wiring below is a static property of MeshAtom::in/out, not
// something that depends on execution state.
SmallVector<size_t>
computePermuteSources(const NormalizedCollective &collective,
                      ArrayRef<StepAtom> atoms) {
  SmallVector<size_t> meshIndices;
  for (const StepAtom &atom : atoms)
    meshIndices.push_back(meshIndexOfLabel(
        collective, AtomLabel{AtomSpace::Mesh, atom.axis, atom.atom}));

  auto positionOf = [&](size_t meshIdx) -> std::optional<size_t> {
    for (size_t i = 0; i < meshIndices.size(); ++i)
      if (meshIndices[i] == meshIdx)
        return i;
    return std::nullopt;
  };

  // A component permute: every mesh partner of every atom in the step is
  // also in the step (a complete permutation cycle or path closure, D5/D15).
  bool closed = true, linked = true;
  for (size_t meshIdx : meshIndices) {
    const MeshAtom &atom = collective.meshAtoms[meshIdx];
    bool hasLink = false;
    for (const MeshAtomRole *role : {&atom.in, &atom.out})
      if (role->kind == AtomRole::Mesh) {
        hasLink = true;
        closed &=
            positionOf(meshIndexOfLabel(collective, role->partner)).has_value();
      }
    linked &= hasLink;
  }

  SmallVector<size_t> source(atoms.size());
  if (closed && linked) {
    for (size_t i = 0; i < meshIndices.size(); ++i) {
      const MeshAtom &atom = collective.meshAtoms[meshIndices[i]];
      size_t sourceMeshIdx;
      if (atom.out.kind == AtomRole::Mesh) {
        sourceMeshIdx = meshIndexOfLabel(collective, atom.out.partner);
      } else {
        // This atom's out role isn't a mesh link, so it's a path's start
        // (D15): it takes the content of the path's end, found by walking
        // the chain of `in` links.
        sourceMeshIdx = meshIndices[i];
        while (collective.meshAtoms[sourceMeshIdx].in.kind == AtomRole::Mesh)
          sourceMeshIdx = meshIndexOfLabel(
              collective, collective.meshAtoms[sourceMeshIdx].in.partner);
      }
      std::optional<size_t> pos = positionOf(sourceMeshIdx);
      assert(pos && "a component permute's source atom must be in the step");
      source[i] = *pos;
    }
  } else {
    // D16's conjugation swap: two atoms, at least one with no mesh link of
    // its own, that simply trade contents.
    assert(atoms.size() == 2 &&
           "a permute step that is not a closed, mesh-linked component must "
           "be a two-atom swap");
    source = {1, 0};
  }
  return source;
}

} // namespace

Value buildPermuteStep(OpBuilder &builder, Location loc, Value operand,
                       const PrimitiveStep &step,
                       ArrayRef<PhysicalCommAxisType> meshAxisTypes,
                       DistributedCollectiveOp collective) {
  assert(step.atoms.size() >= 2 && "a permute step needs at least two atoms");
#ifndef NDEBUG
  for (const StepAtom &atom : step.atoms)
    assert(atom.extent == step.atoms.front().extent &&
           "every atom of a permute step must have equal extent (see the "
           "header comment)");
#endif

  // Which atom's digit each atom in step.atoms takes, computed by
  // computePermuteSources below. A flat StepAtom list alone does not carry
  // this (see that function's comment); the plain 2-atom case is common
  // enough (it's what every whole-axis swap and D16 conjugation produces)
  // to skip the extra work of re-deriving it.
  SmallVector<size_t> source;
  if (step.atoms.size() == 2) {
    source = {1, 0};
  } else {
    std::string failureReason;
    std::optional<NormalizedCollective> normalized =
        normalizeCollective(collective, meshAxisTypes, failureReason);
    assert(normalized &&
           "planCollective already normalized this collective once; it "
           "cannot fail to normalize the same way again");
    source = computePermuteSources(*normalized, step.atoms);
  }

  // The source/target pairing is static (every device's paired peer is
  // known at compile time, even though the value that moves is not), so the
  // whole table is built directly from mesh metadata rather than any
  // runtime device id.
  int64_t deviceCount =
      static_cast<int64_t>(getDenseDeviceCount(meshAxisTypes));

  auto axisCoordinate = [&](int64_t device, size_t axis) -> int64_t {
    PhysicalCommAxisType axisType = meshAxisTypes[axis];
    return (device / static_cast<int64_t>(axisType.getIdStride())) %
           static_cast<int64_t>(axisType.getExtent());
  };
  // An atom's own digit within its axis's mixed-radix coordinate (StepAtom's
  // field comment, CollectiveCost.h): stride picks out the atom's place,
  // extent bounds its digit.
  auto atomDigit = [&](ArrayRef<int64_t> coords,
                       const StepAtom &atom) -> int64_t {
    return (coords[atom.axis] / static_cast<int64_t>(atom.stride)) %
           static_cast<int64_t>(atom.extent);
  };

  SmallVector<int64_t> pairs;
  pairs.reserve(deviceCount * 2);
  SmallVector<int64_t> coords(meshAxisTypes.size());
  SmallVector<int64_t> oldDigits(step.atoms.size());
  for (int64_t device = 0; device < deviceCount; ++device) {
    for (size_t axis = 0; axis < meshAxisTypes.size(); ++axis)
      coords[axis] = axisCoordinate(device, axis);
    // Every atom's digit is read before any is written, so a cycle's
    // simultaneous reassignment doesn't see another atom's already-updated
    // value (the direct generalization of the 2-atom case's std::swap).
    for (size_t i = 0; i < step.atoms.size(); ++i)
      oldDigits[i] = atomDigit(coords, step.atoms[i]);
    for (size_t i = 0; i < step.atoms.size(); ++i) {
      const StepAtom &atom = step.atoms[i];
      int64_t delta = oldDigits[source[i]] - oldDigits[i];
      coords[atom.axis] += delta * static_cast<int64_t>(atom.stride);
    }
    int64_t target = 0;
    for (size_t axis = 0; axis < meshAxisTypes.size(); ++axis)
      target += coords[axis] *
                static_cast<int64_t>(meshAxisTypes[axis].getIdStride());
    pairs.push_back(device);
    pairs.push_back(target);
  }

  auto pairsType =
      RankedTensorType::get({deviceCount, 2}, builder.getI64Type());
  auto sourceTargetPairs = DenseIntElementsAttr::get(pairsType, pairs);
  // A permute preserves the payload, so the operand's own type is the
  // result type.
  auto permuteOp = builder.create<stablehlo::CollectivePermuteOp>(
      loc, operand.getType(), operand, sourceTargetPairs,
      createUniqueStablehloChannelHandle(builder.getContext()));
  return tagStepCost(builder, permuteOp, step);
}

Value buildLocalSliceStep(OpBuilder &builder, Location loc, Value operand,
                          const PrimitiveStep &step,
                          ArrayRef<PhysicalCommAxisType> meshAxisTypes) {
  assert(!step.atoms.empty() && "LocalSlice needs at least one atom");
  const StepAtom &atom = step.atoms[0];
  PhysicalCommAxisType axisType = meshAxisTypes[atom.axis];

  auto operandType = cast<RankedTensorType>(operand.getType());
  Type elementType = operandType.getElementType();
  int64_t outElems = step.payloadOut / elementBytesOf(elementType);
  auto resultType = RankedTensorType::get({outElems}, elementType);

  // Decode the atom's own digit from stablehlo.partition_id the same way
  // LowerForSanityCheck.cpp's buildDeviceId encodes it (id = sum of
  // coordinate * idStride over every axis, and within one axis, coordinate =
  // sum of digit * the digit's own atom stride, see StepAtom's field
  // comment, CollectiveCost.h): invert both terms at once,
  // digit = (id / (idStride * atom.stride)) % atom.extent.
  Type indexElementType = builder.getI64Type();
  auto scalarIndexType = RankedTensorType::get({}, indexElementType);
  auto constIndex = [&](int64_t value) -> Value {
    return builder.create<stablehlo::ConstantOp>(
        loc, DenseIntElementsAttr::get(scalarIndexType, value));
  };
  Value partitionId = builder.create<stablehlo::PartitionIdOp>(loc);
  Value partitionIdSigned =
      builder.create<stablehlo::ConvertOp>(loc, scalarIndexType, partitionId);
  Value deviceStride = constIndex(static_cast<int64_t>(atom.stride) *
                                  static_cast<int64_t>(axisType.getIdStride()));
  Value extent = constIndex(static_cast<int64_t>(atom.extent));
  Value axisCoordinate = builder.create<stablehlo::RemOp>(
      loc,
      builder.create<stablehlo::DivOp>(loc, partitionIdSigned, deviceStride),
      extent);
  Value chunkSize = constIndex(outElems);
  Value start =
      builder.create<stablehlo::MulOp>(loc, axisCoordinate, chunkSize);

  auto sliceOp = builder.create<stablehlo::DynamicSliceOp>(
      loc, resultType, operand, ValueRange{start},
      builder.getDenseI64ArrayAttr({outElems}));
  return tagStepCost(builder, sliceOp, step);
}

} // namespace mlir::enzyme::distributed
