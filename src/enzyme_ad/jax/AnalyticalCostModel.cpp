#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

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

#include "AnalyticalCostModel.h"
#include "RunXlaGpuPasses.h"

using namespace mlir;

uint64_t AnalyticalCostModel::getAnalyticalCost(ModuleOp &wrapperModule) {
  std::unique_ptr<xla::HloModule> preOpt =
      wrapperModuleToHloModule(wrapperModule);

  // Run XLA passes (layout, fusion, simplification) to ensure what is being
  // measured is what will be run
  auto hloModule = runXlaGpuPasses(std::move(preOpt));

  auto deviceDescription = getDeviceDescription();

  xla::HloCostAnalysis::ShapeSizeFunction shapeSizeFunction =
      [](const xla::Shape &shape) {
        return xla::gpu::GetSizeOfShape(shape, 4);
      };
  xla::gpu::GpuHloCostAnalysis costAnalysis(
      xla::gpu::GpuHloCostAnalysis::Options{shapeSizeFunction, {}, {}, true},
      *deviceDescription);

  assert(hloModule->computation_count() == 1);

  // std::cout << hloModule->ToString() << std::endl;

  uint64_t cost = 0;

  auto computation = hloModule->entry_computation();
  computation->Accept(&costAnalysis);

  auto options =
      xla::gpu::GpuPerformanceModelOptions::ForModule(hloModule.get());

  // Sum up the costs for all the ops (assume additivity post-fusion and
  // optimisation)
  for (auto op : computation->instructions()) {
    // std::cout << "measuring " << op->ToString() << std::endl;
    auto runtime = xla::gpu::GpuPerformanceModel::EstimateRunTimeForInstruction(
        op, *deviceDescription, &costAnalysis, options);

    // std::cout << runtime.ToString() << std::endl;
    cost += absl::ToInt64Nanoseconds(runtime.exec_time);
  }

  return cost;
}

/**
 * Create XLA internal HloModule for the analytical cost model
 */
std::unique_ptr<xla::HloModule>
AnalyticalCostModel::wrapperModuleToHloModule(ModuleOp &wrapperModule) {
  auto context = wrapperModule.getContext();
  PassManager pm(context);
  pm.addPass(mlir::mhlo::createStablehloLegalizeToHloPass());
  pm.run(wrapperModule);

  MlirToHloConversionOptions options;
  options.propagate_layouts = true;
  options.return_tuple = false;

  auto hloModule = ConvertMlirHloToHloModule(wrapperModule, options);
  if (!hloModule.ok()) {
    llvm::errs() << "Couldn't create hloModule: "
                 << hloModule.status().message();
    return nullptr;
  } else {
    return std::move(hloModule.value());
  }
}

stream_executor::Platform *AnalyticalCostModel::getXlaPlatform() {
  return xla::PlatformUtil::GetPlatform("cuda").value();
}

/**
 * Get DeviceDescription for current device.
 */
std::unique_ptr<stream_executor::DeviceDescription>
AnalyticalCostModel::getDeviceDescription() {
  // assume ordinal 0
  return std::move(getXlaPlatform()->DescriptionForDevice(0).value());
}
