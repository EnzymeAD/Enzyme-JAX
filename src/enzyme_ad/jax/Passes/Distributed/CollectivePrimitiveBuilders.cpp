#include "src/enzyme_ad/jax/Passes/Distributed/CollectivePrimitiveBuilders.h"

#include "src/enzyme_ad/jax/Dialect/Distributed/NormalizedCollective.h"

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

// Wraps one op built by `buildInner` in stablehlo.async_start/async_done,
// tags the async_start with `costAttr`, and returns the awaited
// (stablehlo.async_done) result.
//
// stablehlo.async_start's body must contain exactly one operation plus its
// terminator (verifier-enforced), and the region is IsolatedFromAbove, so
// every value the wrapped op needs (the data operand and, for LocalSlice, a
// dynamically computed start index) must be passed in through
// `operands` and referenced inside `buildInner` only via the block
// arguments it is handed (in the same order), never via the outer
// `operands` themselves. Any op that isn't the one real op being wrapped
// (a reshape reinterpreting a step's canonical shape, or the arithmetic
// that computes a dynamic-slice start index) has to be built outside this
// wrapper, before/after the async_start/async_done pair.
Value wrapInAsyncStartDone(
    OpBuilder &builder, Location loc, ValueRange operands, Type resultType,
    DictionaryAttr costAttr,
    llvm::function_ref<Value(OpBuilder &, ValueRange)> buildInner) {
  auto futureType = stablehlo::FutureType::get(builder.getContext(),
                                               SmallVector<Type>{resultType});
  OperationState state(loc, stablehlo::AsyncStartOp::getOperationName());
  state.addOperands(operands);
  state.addTypes(futureType);
  state.addRegion();
  auto asyncStartOp = cast<stablehlo::AsyncStartOp>(builder.create(state));
  asyncStartOp->setAttr("distributed.primitive_cost", costAttr);

  {
    OpBuilder::InsertionGuard guard(builder);
    Block *block = builder.createBlock(&asyncStartOp->getRegion(0));
    SmallVector<Value> blockOperands;
    blockOperands.reserve(operands.size());
    for (Value operand : operands)
      blockOperands.push_back(block->addArgument(operand.getType(), loc));
    builder.setInsertionPointToStart(block);
    Value innerResult = buildInner(builder, blockOperands);
    builder.create<stablehlo::ReturnOp>(loc, ValueRange{innerResult});
  }
  return builder.create<stablehlo::AsyncDoneOp>(loc, asyncStartOp.getResult());
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

Attribute
createReplicaGroupsFromStepAtoms(ArrayRef<StepAtom> atoms,
                                 ArrayRef<PhysicalCommAxisType> meshAxisTypes,
                                 OpBuilder &builder) {
  assert(!atoms.empty() &&
         "expected at least one atom for replica group creation");

  int64_t flatMeshExtent = 1;
  for (PhysicalCommAxisType axisType : meshAxisTypes)
    flatMeshExtent *= static_cast<int64_t>(axisType.getExtent());

  constexpr StringLiteral kFlatAxisName = "axis_0";
  SmallVector<stablehlo::MeshAxisAttr> meshAxes;
  meshAxes.push_back(stablehlo::MeshAxisAttr::get(
      builder.getContext(), kFlatAxisName, flatMeshExtent));

  SmallVector<Attribute> axisRefs;
  axisRefs.reserve(atoms.size());
  for (const StepAtom &atom : atoms) {
    PhysicalCommAxisType axisType = meshAxisTypes[atom.axis];
    // The flat StableHLO axis built above is the whole mesh's device id, in
    // the same row-major convention buildDeviceId (LowerForSanityCheck.cpp)
    // encodes it: id = sum of coordinate * idStride over every physical
    // axis. Within one physical axis, `atom.stride` is this atom's own
    // digit multiplier in that axis's mixed-radix coordinate (see StepAtom's
    // field comment, CollectiveCost.h), so `atom.stride * idStride` is this
    // atom's device-id stride, the same formula
    // createReplicaGroupsFromPhysicalFactors (DistributedToHlo.cpp) uses for
    // a factor's own stride, generalized to a sub-portion atom instead of a
    // whole-axis one.
    int64_t deviceStride = static_cast<int64_t>(atom.stride) *
                           static_cast<int64_t>(axisType.getIdStride());
    auto subAxis = stablehlo::SubAxisInfoAttr::get(
        builder.getContext(), deviceStride, static_cast<int64_t>(atom.extent));
    axisRefs.push_back(stablehlo::AxisRefAttr::get(builder.getContext(),
                                                   kFlatAxisName, subAxis));
  }

  auto meshAttr = stablehlo::MeshAttr::get(builder.getContext(), meshAxes,
                                           DenseIntElementsAttr());
  return stablehlo::ReplicaGroupMeshAxesAttr::get(
      builder.getContext(), meshAttr, builder.getArrayAttr(axisRefs));
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
  DictionaryAttr costAttr = buildPrimitiveCostAttr(builder, step);

  return wrapInAsyncStartDone(
      builder, loc, ValueRange{operand}, resultType, costAttr,
      [&](OpBuilder &innerBuilder, ValueRange blockOperands) -> Value {
        Value blockOperand = blockOperands.front();
        stablehlo::ChannelHandleAttr channelHandle =
            createUniqueStablehloChannelHandle(innerBuilder.getContext());
        auto allReduceOp = innerBuilder.create<stablehlo::AllReduceOp>(
            loc, TypeRange{resultType}, ValueRange{blockOperand}, replicaGroups,
            channelHandle, /*use_global_device_ids=*/true);
        RankedTensorType scalarType = RankedTensorType::get({}, elementType);
        {
          OpBuilder::InsertionGuard guard(innerBuilder);
          Block *block =
              innerBuilder.createBlock(&allReduceOp.getComputation());
          block->addArgument(scalarType, loc);
          block->addArgument(scalarType, loc);
          innerBuilder.setInsertionPointToStart(block);
          Value lhs = block->getArgument(0);
          Value rhs = block->getArgument(1);
          IRMapping mapper;
          Block &sourceBlock = reductionBody.front();
          mapper.map(sourceBlock.getArgument(0), lhs);
          mapper.map(sourceBlock.getArgument(1), rhs);
          for (Operation &nestedOp : sourceBlock.without_terminator())
            innerBuilder.clone(nestedOp, mapper);
          auto returnOp =
              cast<stablehlo::ReturnOp>(sourceBlock.getTerminator());
          SmallVector<Value> returnedValues;
          for (Value returnOperand : returnOp.getOperands())
            returnedValues.push_back(mapper.lookupOrDefault(returnOperand));
          innerBuilder.create<stablehlo::ReturnOp>(loc, returnedValues);
        }
        return allReduceOp.getResults().front();
      });
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
  DictionaryAttr costAttr = buildPrimitiveCostAttr(builder, step);

  return wrapInAsyncStartDone(
      builder, loc, ValueRange{operand}, resultType, costAttr,
      [&](OpBuilder &innerBuilder, ValueRange blockOperands) -> Value {
        Value blockOperand = blockOperands.front();
        stablehlo::ChannelHandleAttr channelHandle =
            createUniqueStablehloChannelHandle(innerBuilder.getContext());
        auto reduceScatterOp = innerBuilder.create<stablehlo::ReduceScatterOp>(
            loc, resultType, blockOperand, /*scatter_dimension=*/uint64_t{0},
            replicaGroups, channelHandle, /*use_global_device_ids=*/true);
        RankedTensorType scalarType = RankedTensorType::get({}, elementType);
        {
          OpBuilder::InsertionGuard guard(innerBuilder);
          Block *block =
              innerBuilder.createBlock(&reduceScatterOp.getComputation());
          block->addArgument(scalarType, loc);
          block->addArgument(scalarType, loc);
          innerBuilder.setInsertionPointToStart(block);
          Value lhs = block->getArgument(0);
          Value rhs = block->getArgument(1);
          IRMapping mapper;
          Block &sourceBlock = reductionBody.front();
          mapper.map(sourceBlock.getArgument(0), lhs);
          mapper.map(sourceBlock.getArgument(1), rhs);
          for (Operation &nestedOp : sourceBlock.without_terminator())
            innerBuilder.clone(nestedOp, mapper);
          auto returnOp =
              cast<stablehlo::ReturnOp>(sourceBlock.getTerminator());
          SmallVector<Value> returnedValues;
          for (Value returnOperand : returnOp.getOperands())
            returnedValues.push_back(mapper.lookupOrDefault(returnOperand));
          innerBuilder.create<stablehlo::ReturnOp>(loc, returnedValues);
        }
        return reduceScatterOp.getResult();
      });
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
  DictionaryAttr costAttr = buildPrimitiveCostAttr(builder, step);

  return wrapInAsyncStartDone(
      builder, loc, ValueRange{operand}, resultType, costAttr,
      [&](OpBuilder &innerBuilder, ValueRange blockOperands) -> Value {
        Value blockOperand = blockOperands.front();
        stablehlo::ChannelHandleAttr channelHandle =
            createUniqueStablehloChannelHandle(innerBuilder.getContext());
        auto allGatherOp = innerBuilder.create<stablehlo::AllGatherOp>(
            loc, TypeRange{resultType}, ValueRange{blockOperand},
            /*all_gather_dim=*/uint64_t{0}, replicaGroups, channelHandle,
            /*use_global_device_ids=*/true);
        return allGatherOp.getResults().front();
      });
}

Value buildAllToAllStep(OpBuilder &builder, Location loc, Value operand,
                        const PrimitiveStep &step,
                        ArrayRef<PhysicalCommAxisType> meshAxisTypes) {
  assert(step.atoms.size() == 1 &&
         "buildAllToAllStep expects one mesh atom per step (D1)");
  int64_t n = static_cast<int64_t>(step.atoms[0].extent);

  auto operandType = cast<RankedTensorType>(operand.getType());
  Type elementType = operandType.getElementType();
  int64_t inElems = operandType.getNumElements();
  assert(inElems % n == 0 &&
         "all-to-all's flat payload must divide evenly by its group size");

  Attribute replicaGroups =
      createReplicaGroupsFromStepAtoms(step.atoms, meshAxisTypes, builder);
  DictionaryAttr costAttr = buildPrimitiveCostAttr(builder, step);

  // stablehlo.all_to_all needs distinct split/concat dimensions, which a
  // flat 1-D tensor does not have (see the header comment on the flat-1-D
  // convention), but stablehlo.async_start's body may only ever contain the
  // one op it wraps plus its terminator (verifier-enforced), so the
  // reshapes bracketing it have to sit outside the async region rather than
  // inside it: reshape to a placeholder [1, inElems] before the wrapped
  // op splits dim 1 into n parts and concats the received parts along dim
  // 0 (giving [n, inElems / n]), then flatten the awaited result back to
  // 1-D (all-to-all keeps the payload size, so this is
  // elementwise-identity).
  auto splitShapeType = RankedTensorType::get({1, inElems}, elementType);
  Value reshapedOperand =
      builder.create<stablehlo::ReshapeOp>(loc, splitShapeType, operand);
  auto allToAllResultType =
      RankedTensorType::get({n, inElems / n}, elementType);

  Value awaited = wrapInAsyncStartDone(
      builder, loc, ValueRange{reshapedOperand}, allToAllResultType, costAttr,
      [&](OpBuilder &innerBuilder, ValueRange blockOperands) -> Value {
        stablehlo::ChannelHandleAttr channelHandle =
            createUniqueStablehloChannelHandle(innerBuilder.getContext());
        auto allToAllOp = innerBuilder.create<stablehlo::AllToAllOp>(
            loc, TypeRange{allToAllResultType}, blockOperands,
            /*split_dimension=*/uint64_t{1}, /*concat_dimension=*/uint64_t{0},
            /*split_count=*/static_cast<uint64_t>(n), replicaGroups,
            channelHandle);
        return allToAllOp.getResults().front();
      });
  return flattenTo1D(builder, loc, awaited);
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
  int64_t deviceCount = 1;
  for (PhysicalCommAxisType axisType : meshAxisTypes)
    deviceCount *= static_cast<int64_t>(axisType.getExtent());

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
  DictionaryAttr costAttr = buildPrimitiveCostAttr(builder, step);
  // A permute preserves the payload, so the operand's own type is the
  // result type.
  Type resultType = operand.getType();

  return wrapInAsyncStartDone(
      builder, loc, ValueRange{operand}, resultType, costAttr,
      [&](OpBuilder &innerBuilder, ValueRange blockOperands) -> Value {
        stablehlo::ChannelHandleAttr channelHandle =
            createUniqueStablehloChannelHandle(innerBuilder.getContext());
        auto permuteOp = innerBuilder.create<stablehlo::CollectivePermuteOp>(
            loc, resultType, blockOperands.front(), sourceTargetPairs,
            channelHandle);
        return permuteOp.getResult();
      });
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
  DictionaryAttr costAttr = buildPrimitiveCostAttr(builder, step);

  // stablehlo.async_start's body may only ever contain the one op it wraps
  // plus its terminator (verifier-enforced), so the arithmetic that decodes
  // this device's coordinate has to be built outside the async region and
  // handed in as a second operand, rather than computed inline next to the
  // dynamic_slice itself.
  //
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

  return wrapInAsyncStartDone(
      builder, loc, ValueRange{operand, start}, resultType, costAttr,
      [&](OpBuilder &innerBuilder, ValueRange blockOperands) -> Value {
        return innerBuilder.create<stablehlo::DynamicSliceOp>(
            loc, resultType, blockOperands[0], ValueRange{blockOperands[1]},
            innerBuilder.getDenseI64ArrayAttr({outElems}));
      });
}

} // namespace mlir::enzyme::distributed
