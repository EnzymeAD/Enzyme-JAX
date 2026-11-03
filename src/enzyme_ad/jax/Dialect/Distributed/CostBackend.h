#ifndef ENZYME_AD_JAX_DIALECT_DISTRIBUTED_COST_BACKEND_H
#define ENZYME_AD_JAX_DIALECT_DISTRIBUTED_COST_BACKEND_H

#include "MeshCostMetadata.h"

#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/StringRef.h"

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace mlir::enzyme::distributed {

// Discardable attribute holding a schedulable op's single-device latency: an
// f64 count of nanoseconds, at least 0 (checked by
// DistributedDialect::verifyOperationAttribute). ScheduleGraph prices a
// Compute node from it alone.
inline constexpr llvm::StringLiteral kLatencyAttrName = "distributed.latency";

// The latency `op` carries, if any.
std::optional<double> getLatency(Operation *op);
void setLatency(Operation *op, double nanoseconds);

// What a backend costs a kernel for: the mesh's device_target, which chose
// the backend, and the device's roofline parameters.
struct CostTarget {
  std::string deviceTarget;
  DeviceCostParams device;
};

// Single-device kernel latency for one kind of device.
//
// The input is the module buildKernelBodyModule builds from a kernel: one
// func.func @kernel at local shapes, holding plain StableHLO. annotate
// rewrites it in place so that afterwards:
//   - @kernel keeps its signature;
//   - every top-level op of @kernel except the terminator carries
//     kLatencyAttrName;
//   - ops that must be scheduled together are moved into a new sibling
//     func.func and replaced by one func.call carrying the group's latency.
//     The sibling funcs are the backend's own functions and stay in the
//     program after splicing (spliceKernelBodyModule).
class CostBackend {
public:
  virtual ~CostBackend() = default;

  virtual llvm::StringRef name() const = 0;

  // Fails with a diagnostic already emitted when the backend cannot cost the
  // kernel.
  virtual LogicalResult annotate(ModuleOp kernelModule,
                                 const CostTarget &target) = 0;
};

using CostBackendFactory = std::function<std::unique_ptr<CostBackend>()>;

// Registers `factory` under the device_target `name`, which must not be
// registered yet. The roofline backend is registered as "cpu" and
// "roofline".
void registerCostBackend(llvm::StringRef name, CostBackendFactory factory);

// The backend registered for `deviceTarget`, created on first lookup and
// owned by the registry, or null if none is registered.
CostBackend *lookupCostBackend(llvm::StringRef deviceTarget);

// Every registered device_target name, sorted.
std::vector<std::string> registeredCostBackendNames();

} // namespace mlir::enzyme::distributed

#endif // ENZYME_AD_JAX_DIALECT_DISTRIBUTED_COST_BACKEND_H
