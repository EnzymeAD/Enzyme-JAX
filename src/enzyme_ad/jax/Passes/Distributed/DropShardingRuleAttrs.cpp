#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h"

#include "shardy/dialect/sdy/ir/utils.h"

namespace mlir::enzyme::distributed {

#define GEN_PASS_DEF_DROPSHARDINGRULEATTRSPASS
#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h.inc"

namespace {

struct DropShardingRuleAttrsPass
    : public impl::DropShardingRuleAttrsPassBase<DropShardingRuleAttrsPass> {
  using DropShardingRuleAttrsPassBase::DropShardingRuleAttrsPassBase;

  void runOnOperation() override {
    ::mlir::sdy::removeShardingRules(getOperation());
  }
};

} // namespace

} // namespace mlir::enzyme::distributed
