#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h"

namespace mlir::enzyme::distributed {

#define GEN_PASS_DEF_ATOMIZECOLLECTIVESPASS
#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h.inc"

namespace {

struct AtomizeCollectivesPass
    : public impl::AtomizeCollectivesPassBase<AtomizeCollectivesPass> {
  using AtomizeCollectivesPassBase::AtomizeCollectivesPassBase;

  void runOnOperation() override { markAllAnalysesPreserved(); }
};

} // namespace

} // namespace mlir::enzyme::distributed
