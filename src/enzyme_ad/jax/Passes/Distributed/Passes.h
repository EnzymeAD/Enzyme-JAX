#ifndef DISTRIBUTED_PASSES_H
#define DISTRIBUTED_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace enzyme {
namespace distributed {

void registerdistributedPasses();

} // namespace distributed
} // namespace enzyme
} // namespace mlir

#endif
