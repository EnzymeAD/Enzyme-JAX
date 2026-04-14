#include "Dialect.h"

namespace mlir::enzyme::distributed {

llvm::SmallVector<mlir::Region *> MeshComputationOp::getLanes() {
  llvm::SmallVector<mlir::Region *> lanes;
  lanes.reserve((*this)->getNumRegions());
  for (Region &region : (*this)->getRegions()) {
    lanes.push_back(&region);
  }
  return lanes;
}

ValueRange MeshComputationOp::getSpmdAxes() const {
  auto segmentSizesAttr =
      (*this)->getAttrOfType<DenseI32ArrayAttr>("operandSegmentSizes");
  if (!segmentSizesAttr || segmentSizesAttr.size() != 3) {
    return {};
  }
  return ValueRange(
      (*this)->getOperands().slice(/*start=*/0, segmentSizesAttr[0]));
}

ValueRange MeshComputationOp::getMpmdAxes() const {
  auto segmentSizesAttr =
      (*this)->getAttrOfType<DenseI32ArrayAttr>("operandSegmentSizes");
  if (!segmentSizesAttr || segmentSizesAttr.size() != 3) {
    return {};
  }
  return ValueRange(
      (*this)->getOperands().slice(/*start=*/segmentSizesAttr[0],
                                   segmentSizesAttr[1]));
}

uint32_t MeshComputationOp::getNumDeviceBodies() const {
  auto attr = (*this)->getAttrOfType<IntegerAttr>("num_device_bodies");
  return attr ? static_cast<uint32_t>(attr.getInt()) : 0;
}

uint32_t MeshComputationOp::getNumCommunicationBodies() const {
  auto attr =
      (*this)->getAttrOfType<IntegerAttr>("num_communication_bodies");
  return attr ? static_cast<uint32_t>(attr.getInt()) : 0;
}

Region &MeshComputationOp::getDeviceBody(unsigned idx) {
  return getOperation()->getRegion(idx);
}

const Region &MeshComputationOp::getDeviceBody(unsigned idx) const {
  return (*this)->getRegion(idx);
}

// Maps a multi-axis MPMD device coordinate to a flat index in the device-body
// partition. Higher-index MPMD axes are treated as higher-significance.
FailureOr<unsigned> MeshComputationOp::findComputationBodyIndexByDeviceIndex(
    const DeviceIndex &deviceIndex) const {
  auto mpmdAxes = getMpmdAxes();
  if (deviceIndex.size() != mpmdAxes.size()) {
    return failure();
  }

  unsigned flatIndex = 0;
  unsigned stride = 1;
  for (size_t axisPos = 0; axisPos < mpmdAxes.size(); ++axisPos) {
    Value axis = mpmdAxes[axisPos];
    int64_t axisSize =
        static_cast<int64_t>(getAxisSize(TypedOpResult<LogicalCommAxisType>(
            axis)));
    if (axisSize <= 0) {
      return failure();
    }

    unsigned coordinate = deviceIndex[axisPos];
    if (coordinate >= static_cast<unsigned>(axisSize)) {
      return failure();
    }

    flatIndex += coordinate * stride;
    stride *= static_cast<unsigned>(axisSize);
  }

  uint32_t numDeviceBodies = getNumDeviceBodies();
  if (flatIndex >= static_cast<unsigned>(numDeviceBodies)) {
    return failure();
  }
  return flatIndex;
}

Region &MeshComputationOp::getDeviceBodyByDeviceIndex(
    const DeviceIndex &deviceIndex) {
  auto bodyIndex = findComputationBodyIndexByDeviceIndex(deviceIndex);
  assert(succeeded(bodyIndex) &&
         "expected a valid DeviceIndex in the MPMD submesh");
  return getDeviceBody(*bodyIndex);
}

const Region &MeshComputationOp::getDeviceBodyByDeviceIndex(
    const DeviceIndex &deviceIndex) const {
  auto bodyIndex = findComputationBodyIndexByDeviceIndex(deviceIndex);
  assert(succeeded(bodyIndex) &&
         "expected a valid DeviceIndex in the MPMD submesh");
  return getDeviceBody(*bodyIndex);
}

Region &MeshComputationOp::getCommunicationBody(unsigned idx) {
  unsigned communicationBodyStart = static_cast<unsigned>(getNumDeviceBodies());
  return getOperation()->getRegion(communicationBodyStart + idx);
}

const Region &MeshComputationOp::getCommunicationBody(unsigned idx) const {
  unsigned communicationBodyStart = getNumDeviceBodies();
  return (*this)->getRegion(communicationBodyStart + idx);
}

// Returns the communication-body index for a communication axis in the
// combined communication partition. The ordering is SPMD axes first, then MPMD
// axes, and the returned index is relative to the start of that partition.
FailureOr<unsigned>
MeshComputationOp::findCommunicationBodyIndexForAxis(Value axis) const {
  unsigned communicationBodyIndex = 0;
  auto spmdAxes = getSpmdAxes();
  auto mpmdAxes = getMpmdAxes();
  for (bool spmdCase : {true, false}) {
    auto axisRange = spmdCase ? spmdAxes : mpmdAxes;
    for (Value candidate : axisRange) {
      if (candidate == axis) {
        return communicationBodyIndex;
      }
      ++communicationBodyIndex;
    }
  }
  return failure();
}

Region &MeshComputationOp::getCommunicationBodyForAxis(Value axis) {
  auto bodyIndex = findCommunicationBodyIndexForAxis(axis);
  assert(succeeded(bodyIndex) &&
         "communication body lookup expects a logical axis from either the "
         "SPMD or MPMD partitions");
  return getCommunicationBody(*bodyIndex);
}

const Region &MeshComputationOp::getCommunicationBodyForAxis(Value axis) const {
  auto bodyIndex = findCommunicationBodyIndexForAxis(axis);
  assert(succeeded(bodyIndex) &&
         "communication body lookup expects a logical axis from either the "
         "SPMD or MPMD partitions");
  return getCommunicationBody(*bodyIndex);
}

// Verifies region partition sizes and axis/body cardinality invariants.
LogicalResult MeshComputationOp::verify() {
  auto segmentSizesAttr =
      (*this)->getAttrOfType<DenseI32ArrayAttr>("operandSegmentSizes");
  if (!segmentSizesAttr) {
    return emitOpError() << "requires operandSegmentSizes attribute";
  }
  if (segmentSizesAttr.size() != 3) {
    return emitOpError()
           << "requires operandSegmentSizes to have 3 entries "
              "(spmd_axes, mpmd_axes, input_tensors)";
  }

  auto numDeviceBodiesAttr =
      (*this)->getAttrOfType<IntegerAttr>("num_device_bodies");
  if (!numDeviceBodiesAttr) {
    return emitOpError() << "requires num_device_bodies attribute";
  }
  auto numCommunicationBodiesAttr =
      (*this)->getAttrOfType<IntegerAttr>("num_communication_bodies");
  if (!numCommunicationBodiesAttr) {
    return emitOpError() << "requires num_communication_bodies attribute";
  }

  if (numDeviceBodiesAttr.getInt() < 0) {
    return emitOpError() << "requires num_device_bodies to be non-negative";
  }
  if (numCommunicationBodiesAttr.getInt() < 0) {
    return emitOpError()
           << "requires num_communication_bodies to be non-negative";
  }

  // Check overall region count
  unsigned expectedBodyCount =
      static_cast<unsigned>(getNumDeviceBodies()) +
      static_cast<unsigned>(getNumCommunicationBodies());
  if (getOperation()->getNumRegions() != expectedBodyCount) {
    return emitOpError()
           << "requires the number of regions to equal device bodies + "
              "communication bodies";
  }

  // Check communication region count matches attributes
  size_t expectedCommunicationBodyCount =
      getSpmdAxes().size() + getMpmdAxes().size();
  if (expectedCommunicationBodyCount !=
      static_cast<size_t>(getNumCommunicationBodies())) {
    return emitOpError()
           << "requires the number of communication bodies to equal the "
              "number of SPMD + MPMD axes";
  }

  // Check device region count matches the product of MPMD axis sizes
  int64_t expectedDeviceBodyCount = 1;
  for (Value axis : getMpmdAxes()) {
    expectedDeviceBodyCount *= static_cast<int64_t>(
        getAxisSize(TypedOpResult<LogicalCommAxisType>(axis)));
  }
  if (expectedDeviceBodyCount != getNumDeviceBodies()) {
    return emitOpError()
           << "requires the number of device bodies to equal the product of "
              "the MPMD axis sizes";
  }

  // Check all axis are disjoint
  llvm::SmallVector<Value> logicalAxes;
  logicalAxes.append(getSpmdAxes().begin(), getSpmdAxes().end());
  logicalAxes.append(getMpmdAxes().begin(), getMpmdAxes().end());
  if (!areLogicalAxesDisjoint(logicalAxes)) {
    return emitOpError()
           << "requires SPMD and MPMD axes to be disjoint, and all factors "
              "for the same physical axis to come from one factorization op";
  }

  return success();
}

} // namespace mlir::enzyme::distributed

#define GET_OP_CLASSES
#include "src/enzyme_ad/jax/Dialect/Distributed/DistributedOps.cpp.inc"
