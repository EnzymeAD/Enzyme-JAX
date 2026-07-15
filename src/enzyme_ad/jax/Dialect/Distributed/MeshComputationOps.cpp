#include "Dialect.h"

namespace mlir::enzyme::distributed {

Region &MeshComputationOp::getDeviceBody(unsigned idx) {
  assert(idx < static_cast<unsigned>(getNumDeviceBodies()) &&
         "device-body index is relative to the start of the device partition");
  return getOperation()->getRegion(idx);
}

const Region &MeshComputationOp::getDeviceBody(unsigned idx) const {
  return const_cast<MeshComputationOp *>(this)->getDeviceBody(idx);
}

// Maps a multi-axis MPMD device coordinate to a flat index in the device-body
// partition. Higher-index MPMD axes are treated as higher-significance.
FailureOr<unsigned> MeshComputationOp::findComputationBodyIndexByDeviceIndex(
    const DeviceIndex &deviceIndex) const {
  if (deviceIndex.size() != getMpmdAxes().size()) {
    return failure();
  }

  unsigned flatIndex = 0;
  unsigned stride = 1;
  for (size_t axisPos = 0; axisPos < getMpmdAxes().size(); ++axisPos) {
    Value axis = getMpmdAxes()[axisPos];
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

  if (flatIndex >= static_cast<unsigned>(getNumDeviceBodies())) {
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
  return const_cast<MeshComputationOp *>(this)->getDeviceBodyByDeviceIndex(
      deviceIndex);
}

Region &MeshComputationOp::getCommunicationBody(unsigned idx) {
  unsigned communicationBodyStart = static_cast<unsigned>(getNumDeviceBodies());
  assert(idx < static_cast<unsigned>(getNumCommunicationBodies()) &&
         "communication-body index is relative to the start of the "
         "communication partition");
  return getOperation()->getRegion(communicationBodyStart + idx);
}

const Region &MeshComputationOp::getCommunicationBody(unsigned idx) const {
  return const_cast<MeshComputationOp *>(this)->getCommunicationBody(idx);
}

// Returns the communication-body index for a communication axis in the
// combined communication partition. The ordering is SPMD axes first, then MPMD
// axes, and the returned index is relative to the start of that partition.
FailureOr<unsigned>
MeshComputationOp::findCommunicationBodyIndexForAxis(Value axis) const {
  unsigned communicationBodyIndex = 0;
  for (bool spmdCase : {true, false}) {
    auto axisRange = spmdCase ? getSpmdAxes() : getMpmdAxes();
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
  return const_cast<MeshComputationOp *>(this)->getCommunicationBodyForAxis(
      axis);
}

// Verifies region partition sizes and axis/body cardinality invariants.
// TODO: keep this verifier minimal and move reusable axis checks into shared
// helpers as they are introduced.
LogicalResult MeshComputationOp::verify() {
  unsigned expectedBodyCount =
      static_cast<unsigned>(getNumDeviceBodies()) +
      static_cast<unsigned>(getNumCommunicationBodies());
  if (getOperation()->getNumRegions() != expectedBodyCount) {
    return emitOpError()
           << "requires the number of regions to equal device bodies + "
              "communication bodies";
  }

  size_t expectedCommunicationBodyCount =
      getSpmdAxes().size() + getMpmdAxes().size();
  if (expectedCommunicationBodyCount !=
      static_cast<size_t>(getNumCommunicationBodies())) {
    return emitOpError()
           << "requires the number of communication bodies to equal the "
              "number of SPMD + MPMD axes";
  }

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

  // TODO: factor axis-disjointness checking into a reusable helper shared with
  // LogicalMeshOp. The check should consider all SPMD and MPMD axes together
  // and reject any overlapping atomic factors.

  return success();
}

} // namespace mlir::enzyme::distributed

#define GET_OP_CLASSES
#include "src/enzyme_ad/jax/Dialect/Distributed/DistributedOps.cpp.inc"
