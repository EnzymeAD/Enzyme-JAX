#include "src/enzyme_ad/jax/Dialects/CommDialect.h"
#include "src/enzyme_ad/jax/Passes/PassDetails.h"
#include "src/enzyme_ad/jax/Passes/Passes.h"

#define DEBUG_TYPE "enzyme"

using namespace mlir;
using namespace mlir::comm;
using namespace enzyme; using namespace mlir::enzyme; // one of the upstream includes we need is wrapped in this namespace

namespace {
struct CommRemoveFoo : public CommRemoveFooBase<CommRemoveFoo> {

  void runOnOperation() override {
    mlir::Operation* op = getOperation();

    // if op is CommFoo, erase
    if(isa<CommFoo>(op)){
        op->erase();
    }
  }
};

} // end anonymous namespace

namespace mlir {
namespace comm {
std::unique_ptr<Pass> createCommRemoveFooPass() {
  return std::make_unique<CommRemoveFoo>();
}
} // namespace enzyme
} // namespace mlir
