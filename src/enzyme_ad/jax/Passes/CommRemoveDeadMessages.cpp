#include "src/enzyme_ad/jax/Dialects/Comm/CommDialect.h"
#include "src/enzyme_ad/jax/Passes/Passes.h"

#define DEBUG_TYPE "enzyme"

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_COMMREMOVEDEADMESSAGES
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

using namespace mlir;
using namespace mlir::comm;
using namespace enzyme;
using namespace mlir::enzyme; // one of the upstream includes we need is wrapped
                              // in this namespace

namespace {
struct CommRemoveDeadMessages
    : public  enzyme::impl::CommRemoveDeadMessagesBase<CommRemoveDeadMessages> {

  static bool is_live(CommMessage msg, llvm::DenseSet<CommMessage> &live_set,
                      llvm::DenseSet<CommMessage> &dead_set) {
    if (live_set.contains(msg))
      return true;
    if (dead_set.contains(msg))
      return false;

    bool live = false;
    for (auto user : msg->getUsers()) {
      if (isa<CommRecv>(user)) {
        live = true;
        break;
      } else if (CommMessage using_msg = dyn_cast<CommMessage>(user)) {
        if (is_live(using_msg, live_set, dead_set)) {
          live = true;
          break;
        }
      } else if (!isa<CommSend>(user)) {
        // Unhandled case, assume live
        live = true;
        break;
      }
    }

    if (live) {
      live_set.insert(msg);
      return true;
    } else {
      dead_set.insert(msg);
      return false;
    }
  }

  void runOnOperation() override {
    mlir::Operation *op = getOperation();

    llvm::DenseSet<CommMessage> live_messages;
    llvm::DenseSet<CommMessage> dead_messages;

    op->walk([&](CommMessage msg) {
      if (!is_live(msg, live_messages, dead_messages)) {
        for (auto user : llvm::to_vector(msg->getUsers())) {
          // Erase only the sends- the other messages need to be walked over so
          // their own cleanup triggers
          if (isa<CommSend>(user)) {
            user->erase();
          }
        }
        // Todo: with multiple messages, will this cause exceptions when we go
        // to erase those?
        msg->erase();
      }
    });
  }
};

} // end anonymous namespace
