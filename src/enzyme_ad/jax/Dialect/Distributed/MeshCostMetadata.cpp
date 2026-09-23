#include "MeshCostMetadata.h"

namespace mlir::enzyme::distributed {

namespace {
// The mesh's own attributes are plain SI seconds/Hz, while everything past
// this point works in nanoseconds. Each seconds-denominated attribute is
// converted here, once.
constexpr double kNsPerSecond = 1.0e9;
} // namespace

MeshCostParams meshCostParamsFromPhysicalMesh(PhysicalMeshOp mesh) {
  std::vector<uint64_t> extents;
  for (Attribute axisAttr : mesh.getAxesAttr())
    extents.push_back(
        cast<PhysicalCommAxisType>(cast<TypeAttr>(axisAttr).getValue())
            .getExtent());

  // The verifier guarantees present arrays have one entry per axis.
  auto toBandwidths = [](ArrayAttr values) {
    std::vector<double> result;
    for (const llvm::APFloat &value : values.getAsValueRange<FloatAttr>())
      result.push_back(value.convertToDouble() / kNsPerSecond);
    return result;
  };
  auto toRoundLatenciesNs = [](ArrayAttr values) {
    std::vector<int64_t> result;
    for (const llvm::APFloat &value : values.getAsValueRange<FloatAttr>())
      result.push_back(std::llround(value.convertToDouble() * kNsPerSecond));
    return result;
  };
  MeshCostParams params(extents);
  if (ArrayAttr bandwidths = mesh.getAxisBandwidthsAttr())
    params.bandwidth = toBandwidths(bandwidths);
  if (ArrayAttr latencies = mesh.getAxisLatenciesAttr())
    params.roundLatency = toRoundLatenciesNs(latencies);
  if (FloatAttr launch = mesh.getLaunchLatencyAttr())
    params.launchLatency =
        std::llround(launch.getValueAsDouble() * kNsPerSecond);
  return params;
}

DeviceCostParams deviceCostParamsFromPhysicalMesh(PhysicalMeshOp mesh) {
  DeviceCostParams params;
  if (FloatAttr flops = mesh.getDeviceFlopsAttr())
    params.flops = flops.getValueAsDouble() / kNsPerSecond;
  if (FloatAttr memBandwidth = mesh.getDeviceMemBandwidthAttr())
    params.memBandwidth = memBandwidth.getValueAsDouble() / kNsPerSecond;
  if (FloatAttr clockFrequency = mesh.getDeviceClockFrequencyAttr())
    params.clockFrequency = clockFrequency.getValueAsDouble() / kNsPerSecond;
  if (FloatAttr actCostScale = mesh.getActCostScaleAttr())
    params.actCostScale = actCostScale.getValueAsDouble();
  return params;
}

} // namespace mlir::enzyme::distributed
