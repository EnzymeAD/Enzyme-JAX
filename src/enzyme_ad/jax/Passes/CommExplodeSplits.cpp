#include "src/enzyme_ad/jax/Dialect/Comm/CommDialect.h"
#include "src/enzyme_ad/jax/Passes/Passes.h"

#include "mlir/IR/Builders.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"

#include <utility>

#define DEBUG_TYPE "enzyme"

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_COMMEXPLODESPLITS
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

using namespace mlir;
using namespace mlir::comm;
using namespace enzyme;
using namespace mlir::enzyme; // one of the upstream includes we need is wrapped
                              // in this namespace

namespace {
struct CommExplodeSplits : public  enzyme::impl::CommExplodeSplitsBase<CommExplodeSplits> {

  /**
   * Creates a multiplexmessage wrapping this one and sets all recieves from
   * this message to the multiplex.
   *
   */
  static CommMultiplexMessage createMultiplex(OpBuilder &builder,
                                              CommMessage msg) {
    OpBuilder::InsertionGuard insert_guard(builder);

    // Find all recieve points
    llvm::SmallVector<OpOperand *> receiving_uses;
    for (auto &use : msg->getUses()) {
      Operation *user = use.getOwner();

      if (!isa<CommSend>(user)) {
        // Complex messages count as receivers too, not just recv statements
        receiving_uses.push_back(&use);
      }
    }

    // Create multiplex just after this one
    builder.setInsertionPointAfter(msg);

    Type out_tok_type = MessageTokenType::get(msg.getContext());
    ValueRange in_tokens = (ValueRange({msg->getOpResult(0)}));
    TypeAttr out_data_type = TypeAttr::get(msg.getOutputType());
    CommMultiplexMessage plex = builder.create<CommMultiplexMessage>(
        msg.getLoc(), out_tok_type, out_data_type, in_tokens);

    for (auto use : receiving_uses) {
      use->assign(plex.getToken());
    }
    return plex;
  }

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
          // TODO split the basic block, add a jump to entry, and link joins to
          // next statement.
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
      int n_clones = branch.getDeviceIds().size();

      // Look for any subsplits and move their message declarations up
      branch.walk([&](CommSplit subsplit) {
        for (auto msg : llvm::to_vector(subsplit.getMessages())) {
          msg->moveBefore(branch);
        }
      });

      // Don't care about branches with only one device
      if (n_clones == 1) {
        inlineSplits(branch);
        continue;
      }

      // Look for any send statements that appear in this branch in a
      // multi-device context. When we clone the branches, we will want to
      // transform their simple messages into multiplex messages. Since sub
      // branches have already been exploded they can be excluded from
      // consideration. mapping of tokens to their multiplex op
      OpBuilder builder(branch->getContext());
      llvm::DenseMap<TypedValue<MessageTokenType>, CommMultiplexMessage>
          tok_to_plex;
      split->walk<WalkOrder::PreOrder>([&](Operation *op) {
        if (isa<CommSplit>(op) && op != split) {
          // sub branches exploded already
          return WalkResult::skip();
        }
        if (CommSend send = dyn_cast<CommSend>(op)) {
          if (!tok_to_plex.contains(send.getToken())) {
            CommMultiplexMessage multiplex =
                createMultiplex(builder, send.getMessage());
            tok_to_plex.insert(std::make_pair(send.getToken(), multiplex));
          }
        }
        return WalkResult::advance();
      });

      // Copy the branch for each additional device. For the last one we can
      // just mutate instead of cloning.
      for (int i = 0; i < n_clones - 1; i++) {
        CommBranch cloned = branch.clone();

        cloned.setDeviceIds({branch.getDeviceIds()[i]});

        // can't use insertBefore/After immediately since it segfaults trying to
        // unlink- need to add to a block first
        branch->getBlock()->push_back(cloned);
        cloned->moveBefore(branch);
        inlineSplits(cloned);

        // Walk over all sends and if they have been mapped to a multiplex
        // replace them
        llvm::DenseMap<TypedValue<MessageTokenType>,
                       TypedValue<MessageTokenType>>
            token_replacements;
        cloned.walk([&](CommSend send) {
          auto orig_token = send.getToken();
          if (token_replacements.contains(orig_token)) {
            // if we've already cloned the message for this branch use the clone
            send.getTokenMutable().assign(token_replacements[orig_token]);
          } else if (tok_to_plex.contains(orig_token)) {
            // if this token needs to be multiplexed, create a clone of its
            // message and add to the multiplex

            auto orig_msg = orig_token.getDefiningOp();
            auto cloned_msg = llvm::cast<CommSimpleMessage>(orig_msg->clone());
            orig_msg->getBlock()->push_back(cloned_msg);
            cloned_msg->moveAfter(orig_token.getDefiningOp());
            auto new_token = cloned_msg.getToken();
            tok_to_plex[orig_token].getInTokensMutable().append({new_token});
            send.getTokenMutable().assign(new_token);
            token_replacements[orig_token] = new_token;
          }
        });
      }

      branch.setDeviceIds({branch.getDeviceIds().back()});
      inlineSplits(branch);
    }
  }

  void runOnOperation() override { getOperation()->walk(explodeSplit); }
};

} // end anonymous namespace
