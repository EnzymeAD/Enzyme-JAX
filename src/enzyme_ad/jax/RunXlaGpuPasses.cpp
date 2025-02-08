#include "xla/mlir_hlo/mhlo/transforms/passes.h"
#include "xla/service/backend.h"
#include "xla/service/compiler.h"
#include "xla/service/gpu/gpu_latency_hiding_scheduler.h"
#include "xla/service/gpu/model/gpu_hlo_cost_analysis.h"
#include "xla/service/gpu/model/gpu_performance_model.h"
#include "xla/service/gpu/nvptx_compiler.h"
#include "xla/service/platform_util.h"
#include "xla/stream_executor/cuda/cuda_platform_id.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/translate/mhlo_to_hlo/mlir_hlo_to_hlo.h"

#include "RunXlaGpuPasses.h"

/**
 * Get StreamExecutor for current device.
 */
stream_executor::StreamExecutor *getStreamExecutor() {
  auto platform = xla::PlatformUtil::GetPlatform("cuda").value();
  // assume ordinal 0
  auto executor = platform->ExecutorForDevice(0).value();
  if (executor == nullptr) {
    throw std::runtime_error("Couldn't get executor");
  }

  return executor;
}

std::unique_ptr<xla::HloModule>
runXlaGpuPasses(std::unique_ptr<xla::HloModule> hloModule) {
  xla::gpu::NVPTXCompiler compiler;
  auto executor = getStreamExecutor();
  xla::gpu::NVPTXCompiler::CompileOptions options;
  auto res = compiler.RunHloPasses(std::move(hloModule), executor, options);
  return std::move(res.value());
}
