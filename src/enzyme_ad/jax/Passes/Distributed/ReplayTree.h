#ifndef ENZYME_AD_JAX_PASSES_DISTRIBUTED_REPLAYTREE_H
#define ENZYME_AD_JAX_PASSES_DISTRIBUTED_REPLAYTREE_H

#include <memory>
#include <type_traits>
#include <vector>

namespace mlir::enzyme::distributed {
enum ReplayQueryShortCircuit { Continue, Stop };

/**
 * For our search, we need to query a list of decisions at each leaf,
 * and we need to copy leaves / grow the tree. If we do this nievely by:
 *  - Replaying the decision list from root to leaf: O(n) traversal time
 *    for each leaf
 *  - Storing the complete state at each leaf: O(n) size per leaf,
 *    O(n) time to copy
 * So extending down from root in a linear chain in either approach would take
 * O(n^2) time. (Worse if we are expanding a full tree, of course).
 *
 * We can do a hybrid checkpoint approach, though, where instead of depth D
 * pointing to depth (D -1), we point to the parent differing only by taking the
 * lowest-significance 1 bit and replacing it with a 0: i.e. a chain might go
 * 0000 <-- 0100 <-- 0101, with a hop here compressing 4 and 1 steps
 * respectively. This should give an overall traversal time of O(log(n)) "hops"
 * and an ammortized O(1) (?) construction time for each leaf.
 *
 * However, despite reducing the number of hops, the overall decisions stored
 * in the chain is still the same (in our 0101 example, we still walked over
 * deltas compressing 5 steps). So this only gets us any benefit if we can
 * internally do better than O(n) operations within each hop. An O(1) map
 * lookup, for example, gets us an O(log(n)) query over the entire replay;
 * an O(n) operation still creates an O(n) query over the entire replay.
 *
 * For this structure we require a DeltaType with an associative "apply"
 * operation allowing one delta to be applied to another. Empty-construting this
 * delta should give us an identity element.
 */
template <typename Derived, typename DeltaType> class ReplayTree {
protected:
  using SelfT = Derived;
  DeltaType delta;
  std::shared_ptr<const SelfT> parent;
  uint32_t depth;

public:
  ReplayTree(std::shared_ptr<const SelfT> predecessor, uint32_t depth)
      : depth(depth), delta() {
    int stepsUp = llvm::countr_zero(depth);
    std::vector<std::shared_ptr<const SelfT>> toCompress;
    for (int i = 0; i < stepsUp; ++i) {
      toCompress.push_back(predecessor);
      predecessor = predecessor->parent;
    }
    parent = predecessor;
    for (auto it = toCompress.rbegin(); it != toCompress.rend(); ++it) {
      // Replay onto the delta, in original order
      delta.apply((*it)->delta);
    }
  }

  // Define the necessary members and methods for the ReplayTree here.
  static std::shared_ptr<SelfT> makeRoot() {
    return std::make_shared<SelfT>(nullptr, 0);
  }
  static std::shared_ptr<SelfT> makeChild(std::shared_ptr<SelfT> predecessor) {
    return std::make_shared<SelfT>(predecessor, predecessor->getDepth() + 1);
  }

  // Any mutating operations to the replay tree can be done
  // on this getDelta reference.
  DeltaType &getDelta() { return delta; }

  uint32_t getDepth() const { return depth; }

  // Currently return-by-reference or state on the query object.
  // Query must return a short-circuit status indicating whether to
  // continue or stop. Will always search the latest part of the replay
  // first.
  template <typename Query, typename... Args>
  void queryReplayReverse(Query q, Args... args) const {
    // Query the current delta first
    ReplayQueryShortCircuit status = q((const DeltaType)delta, args...);
    if (status == Stop) {
      return;
    }
    // Continue querying the parent if it exists
    if (parent) {
      parent->queryReplayReverse(q, args...);
    }
  }
};
} // namespace mlir::enzyme::distributed
#endif // ENZYME_AD_JAX_PASSES_DISTRIBUTED_REPLAYTREE_H