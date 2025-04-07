#include "mhlo/IR/hlo_ops.h"
#include "src/enzyme_ad/jax/Dialect/Dialect.h"
#include "src/enzyme_ad/jax/Dialect/Ops.h"
#include "src/enzyme_ad/jax/Passes/Passes.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "llvm/ADT/DynamicAPInt.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/MathExtras.h"
#include <algorithm>
#include <cstdint>

#define DEBUG_TYPE "shard-p2p"

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_SHARDPOINTTOPOINT
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

using namespace mlir;
using namespace mlir::enzyme;

struct ShardPointToPointPass
    : public enzyme::impl::ShardPointToPointBase<ShardPointToPointPass> {
  using Base::Base;
  void runOnOperation() override {
    // do nothing for now
  }
};
