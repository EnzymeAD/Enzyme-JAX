#include "xla/hlo/ir/hlo_module.h"
#include <memory>

std::unique_ptr<xla::HloModule>
runXlaGpuPasses(std::unique_ptr<xla::HloModule> hloModule);
