#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "src/enzyme_ad/jax/Passes/Comm/CommPasses.h"

namespace mlir {
namespace enzyme {
#define GEN_PASS_DECL_COMMFLATTENSPLITSPASS
#include "src/enzyme_ad/jax/Passes/Comm/CommPasses.h.inc"
} // namespace enzyme
} // namespace mlir
