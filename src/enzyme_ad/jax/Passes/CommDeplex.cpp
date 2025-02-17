#include "src/enzyme_ad/jax/Dialect/Comm/CommDialect.h"
#include "src/enzyme_ad/jax/Passes/Passes.h"

#define DEBUG_TYPE "enzyme"

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_COMMDEPLEX
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

using namespace mlir;
using namespace mlir::comm;
using namespace enzyme;
using namespace mlir::enzyme; // one of the upstream includes we need is wrapped
                              // in this namespace

namespace {
struct CommDeplex : public  enzyme::impl::CommDeplexBase<CommDeplex> {

  /**
   * Reassigns each use of this multiplex's token to one of the contributing
   * tokens.
   *
   * TODO: this can potentially be a complex decision based on device load,
   * communication latency, potential for removing communcations/computations
   * outright, etc.
   */
  static void chooseMultiplexMapping(CommMultiplexMessage plex) {
    plex.getToken().replaceAllUsesWith(plex.getInTokens().front());
  }

  void runOnOperation() override {
    mlir::Operation *op = getOperation();
    op->walk([](CommMultiplexMessage plex) {
      chooseMultiplexMapping(plex);
      plex.erase();
    });
  }
};

} // end anonymous namespace