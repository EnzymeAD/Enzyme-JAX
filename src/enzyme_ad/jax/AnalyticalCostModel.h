#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Verifier.h"

#include "xla/mlir_hlo/mhlo/transforms/passes.h"
#include "xla/service/backend.h"
#include "xla/service/gpu/gpu_latency_hiding_scheduler.h"
#include "xla/service/gpu/model/gpu_hlo_cost_analysis.h"
#include "xla/service/gpu/model/gpu_performance_model.h"
#include "xla/service/platform_util.h"
#include "xla/stream_executor/cuda/cuda_platform_id.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/translate/mhlo_to_hlo/mlir_hlo_to_hlo.h"

#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

class AnalyticalCostModel {
public:
  static uint64_t getAnalyticalCost(mlir::ModuleOp &wrapperModule);

private:
  /**
   * Create XLA internal HloModule for the analytical cost model
   */
  static std::unique_ptr<xla::HloModule>
  wrapperModuleToHloModule(mlir::ModuleOp &wrapperModule);

  static stream_executor::Platform *getXlaPlatform();

  /**
   * Get DeviceDescription for current device.
   */
  static std::unique_ptr<stream_executor::DeviceDescription>
  getDeviceDescription();
};
