#include "xla/hlo/ir/hlo_module.h"
#include <memory>

#include "RunXlaGpuPasses.h"

/** Dummy implementation to make it build on Mac. */
std::unique_ptr<xla::HloModule>
runXlaGpuPasses(std::unique_ptr<xla::HloModule> hloModule) {
  std::runtime_error("stub");
}
