#include "MeshCostMetadata.h"

namespace mlir::enzyme::distributed {

MeshCostParams meshCostParamsFromPhysicalMesh(PhysicalMeshOp mesh) {
  std::vector<uint64_t> extents;
  for (Attribute axisAttr : mesh.getAxesAttr())
    extents.push_back(
        cast<PhysicalCommAxisType>(cast<TypeAttr>(axisAttr).getValue())
            .getExtent());

  // The verifier guarantees present arrays have one entry per axis.
  auto toDoubles = [](ArrayAttr values) {
    std::vector<double> result;
    for (const llvm::APFloat &value : values.getAsValueRange<FloatAttr>())
      result.push_back(value.convertToDouble());
    return result;
  };
  MeshCostParams params(extents);
  if (ArrayAttr bandwidths = mesh.getAxisBandwidthsAttr())
    params.bandwidth = toDoubles(bandwidths);
  if (ArrayAttr latencies = mesh.getAxisLatenciesAttr())
    params.roundLatency = toDoubles(latencies);
  if (FloatAttr launch = mesh.getLaunchLatencyAttr())
    params.launchLatency = launch.getValueAsDouble();
  return params;
}

DeviceCostParams deviceCostParamsFromPhysicalMesh(PhysicalMeshOp mesh) {
  DeviceCostParams params;
  if (FloatAttr flops = mesh.getDeviceFlopsAttr())
    params.flops = flops.getValueAsDouble();
  if (FloatAttr memBandwidth = mesh.getDeviceMemBandwidthAttr())
    params.memBandwidth = memBandwidth.getValueAsDouble();
  return params;
}

} // namespace mlir::enzyme::distributed
