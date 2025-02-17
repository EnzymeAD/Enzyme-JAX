#include "src/enzyme_ad/jax/Dialects/Comm/CommDialect.h"
#include "src/enzyme_ad/jax/Passes/Passes.h"

#define DEBUG_TYPE "enzyme"

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_COMMREMOVEFOO
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

using namespace mlir;
using namespace mlir::comm;
using namespace enzyme;
using namespace mlir::enzyme; // one of the upstream includes we need is wrapped
                              // in this namespace

namespace {
struct CommRemoveFoo : public  enzyme::impl::CommRemoveFooBase<CommRemoveFoo> {

  void runOnOperation() override {
    mlir::Operation *op = getOperation();
    op->walk([](CommFoo foop) { foop->erase(); });
  }
};

} // end anonymous namespace