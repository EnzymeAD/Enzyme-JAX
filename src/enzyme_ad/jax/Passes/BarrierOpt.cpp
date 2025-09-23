#include "src/enzyme_ad/jax/Passes/BarrierOpt.h"

namespace mlir {
namespace enzymexla {

llvm::cl::opt<bool> BarrierOpt("barrier-opt",
                               llvm::cl::init(true),
                               llvm::cl::desc("Optimize barriers"));

} // namespace enzymexla
} // namespace mlir