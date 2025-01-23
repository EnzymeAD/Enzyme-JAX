#include "src/enzyme_ad/jax/Dialects/Comm/CommDialect.h"
#include "src/enzyme_ad/jax/Passes/PassDetails.h"
#include "src/enzyme_ad/jax/Passes/Passes.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "enzyme"

using namespace mlir;
using namespace mlir::comm;
using namespace enzyme;
using namespace mlir::enzyme; // one of the upstream includes we need is wrapped
                              // in this namespace

namespace {
struct CommExplodeSplits : public CommExplodeSplitsBase<CommExplodeSplits> {

  /**
   * After a branch is split and contains only one device, look through it for
   * split nodes and inline them.
   */
  static void inlineSplits(CommBranch branch) {
    assert(branch.getDeviceIds().size() == 1 &&
           "Shouldn't inline on branch with multiple devices");

    for (CommSplit subsplit : llvm::to_vector(branch.getOps<CommSplit>())) {
      // locate the branch corresponding to the current one
      for (CommBranch subbranch : subsplit.getBranches()) {
        assert(subbranch.getDeviceIds().size() == 1 &&
               "Sub splits should already have been exploded");
        if (!(subbranch.getDeviceIds().front() ==
              branch.getDeviceIds().front()))
          continue;

        // We've found our branch. Now we want to copy its code up in place of
        // the split branch. If we only have one basic block this is easy.
        if (subbranch.getRegion().hasOneBlock()) {
          // Can simply copy all instructions before split and then erase it
          Block &block = subbranch.getRegion().front();
          while (!block.empty()) {
            // TODO what about other terminators?
            auto &op = block.front();
            if (isa<CommJoin>(op)) {
              op.erase();
            } else {
              op.moveBefore(subsplit);
            }
          }
          subsplit.erase();

        } else {
          assert(0 && "TODO inline multiple blocks");
          // TODO split the basic block, add a jump to entry, and link joins to next statement.
        }

        break;
      }
    }
  }

  /**
   * Functor to perform the explode split, i.e. create a distinct branch for
   * each device under a single top-level split. Must be called as a post-order
   * traversal so that all sub-splits are already exploded.
   */
  static void explodeSplit(CommSplit split) {

    for (CommBranch branch : llvm::to_vector(split.getBranches())) {
      llvm::dbgs() << "Running on branch " << branch << "\n";
      // Look for any subsplits and move their message declarations up
      branch.walk([&](CommSplit subsplit) {
        for (auto msg : llvm::to_vector(subsplit.getMessages())) {
          msg->moveBefore(branch);
        }
      });

      // Copy the branch for each additional device. For the last one we can
      // just mutate instead of cloning.
      for (int i = 0; i < branch.getDeviceIds().size() - 1; i++) {
        llvm::dbgs() << "Creating deep clone of branch\n";
        CommBranch cloned = branch.clone();

        llvm::dbgs() << "Setting branch ids\n";
        cloned.setDeviceIds({branch.getDeviceIds()[i]});
        llvm::dbgs() << "Moving branch into split block at loc "
                     << branch.getLoc() << "\n";
        // can't use insertBefore/After immediately since it segfaults trying to
        // unlink.
        branch->getBlock()->push_back(cloned);
        cloned->moveBefore(branch);
        inlineSplits(cloned);
      }

      branch.setDeviceIds({branch.getDeviceIds().back()});
      inlineSplits(branch);

      llvm::dbgs() << "Done with current branch\n";
    }

    llvm::dbgs() << "Done with all branches\n";
  }

  void runOnOperation() override {
    llvm::dbgs() << "Running pass on op " << getOperation() << "\n";
    getOperation()->walk(explodeSplit);
  }
};

} // end anonymous namespace

namespace mlir {
namespace comm {
std::unique_ptr<Pass> createCommExplodeSplitsPass() {
  llvm::dbgs() << "creating explode pass\n";
  return std::make_unique<CommExplodeSplits>();
}
} // namespace comm
} // namespace mlir
