
#ifndef ENZYME_PASSES_DISTRIBUTED_BEAMSEARCHDRIVER_H
#define ENZYME_PASSES_DISTRIBUTED_BEAMSEARCHDRIVER_H

#include <functional>
#include <memory>
#include <queue>
#include <vector>

#include "llvm/ADT/ArrayRef.h"

namespace mlir::enzyme::distributed {

class BeamSearchNodeBase {
public:
  double score;
  virtual bool finalized() const = 0;
  virtual ~BeamSearchNodeBase() = default;
};

/**
 * "The beam": in charge of maintaining the current set of candidates and
 * choosing pruning, exploration order.
 */
template <typename NodeType> class BeamSearchQueueBase {
  using NodePtr = std::shared_ptr<NodeType>;

public:
  virtual void push(NodePtr node) = 0;
  virtual NodePtr pop() = 0;
  virtual bool done() = 0;
  virtual ~BeamSearchQueueBase() = default;
};

template <typename NodeType> class BeamSearchScorerBase {
public:
  virtual double score(const std::shared_ptr<NodeType> &node) = 0;
  virtual ~BeamSearchScorerBase() = default;
};

template <typename NodeType> class BeamSearchExplorerBase {
public:
  using NodePtr = std::shared_ptr<NodeType>;
  virtual std::vector<NodePtr> generateCandidatesFromNode(NodePtr node) = 0;
  virtual ~BeamSearchExplorerBase() = default;
};

/**
 * Build a beam search driver independent of things like actual
 * search strategy, decision space, heuristics, etc. The driver should
 * allow us to tune the search without needing to mess with much
 * implementation, and allow us to place metrics within the driver without
 * needing to touch the search strategy itself.
 *
 * Driver parameterizes on:
 *  - Candidate spec: the state the candidate is in (i.e. the decisions
 * needed to construct the candidate)
 *  - Scoring function: how to score a candidate spec
 *  - Candidate generator: how to generate new candidate specs from a given
 *    candidate spec
 *  - Search node evaluation order (i.e. best first, breadth first, etc.)
 *
 * Partially parameterizes on:
 *  - Search node state: we always store score,"finalized" flag, candidate
 * spec, but may allow the user to store additional state in the search
 * node.
 */
template <typename NodeType> class BeamSearchDriver {
public:
  using NodePtr = std::shared_ptr<NodeType>;
  using QueueType = BeamSearchQueueBase<NodeType>;
  using ScorerType = BeamSearchScorerBase<NodeType>;
  using ExplorerType = BeamSearchExplorerBase<NodeType>;

protected:
  QueueType &queue;
  ScorerType &scorer;
  ExplorerType &explorer;
  std::vector<NodePtr> finalizedCandidates;
  NodePtr bestFinalizedCandidate;

public:
  BeamSearchDriver(QueueType &queue, ScorerType &scorer, ExplorerType &explorer)
      : queue(queue), scorer(scorer), explorer(explorer){};

  NodePtr getBest() { return bestFinalizedCandidate; }
  llvm::ArrayRef<NodePtr> getFinalized() { return finalizedCandidates; }

  void run() {
    while (!queue.done()) {
      NodePtr node = queue.pop();
      if (node->finalized()) {
        // Finalized candidates get recorded
        finalizedCandidates.push_back(node);
        if (!bestFinalizedCandidate ||
            node->score > bestFinalizedCandidate->score) {
          bestFinalizedCandidate = node;
        }
      } else {
        // In-progress candidates must be expanded
        std::vector<NodePtr> newCandidates =
            explorer.generateCandidatesFromNode(node);
        for (NodePtr newNode : newCandidates) {
          double newScore = scorer.score(newNode);
          newNode->score = newScore;
          queue.push(newNode);
        }
      }
    }
  }
};

template <typename NodeType>
class BeamSearchBreadthFirstQueue : public BeamSearchQueueBase<NodeType> {
  using NodePtr = std::shared_ptr<NodeType>;
  int max_residency;

  std::queue<NodePtr> queue;
  struct CompareScore {
    bool operator()(const NodePtr &lhs, const NodePtr &rhs) const {
      return lhs->score > rhs->score;
    }
  };
  std::priority_queue<NodePtr, std::vector<NodePtr>, CompareScore> incoming;
  // Called with every node in a freshly-refilled generation, right as it
  // becomes the new queue (see pop() below) -- the one point where the whole
  // beam for this round is known at once. Left null (the default), this is a
  // no-op; a caller that wants per-generation progress markers (e.g. the
  // deepest/shallowest search depth reached so far) supplies one at
  // construction.
  std::function<void(llvm::ArrayRef<NodePtr>)> onGenerationStart;

public:
  BeamSearchBreadthFirstQueue(
      int max_residency,
      std::function<void(llvm::ArrayRef<NodePtr>)> onGenerationStart = nullptr)
      : max_residency(max_residency), queue(), incoming(),
        onGenerationStart(std::move(onGenerationStart)) {}

  void push(NodePtr node) override {
    incoming.push(node);
    if (incoming.size() > max_residency) {
      // pop worst node off the incoming queue
      incoming.pop();
    }
  }

  NodePtr pop() override {
    if (queue.empty()) {
      // refill the queue from the incoming queue
      std::vector<NodePtr> generation;
      while (!incoming.empty()) {
        NodePtr node = incoming.top();
        incoming.pop();
        queue.push(node);
        if (onGenerationStart) {
          generation.push_back(node);
        }
      }
      if (onGenerationStart && !generation.empty()) {
        onGenerationStart(generation);
      }
    }
    NodePtr node = queue.front();
    queue.pop();
    return node;
  }
  bool done() override { return queue.empty() && incoming.empty(); }
};

} // namespace mlir::enzyme::distributed

#endif // ENZYME_PASSES_DISTRIBUTED_BEAMSEARCHDRIVER_H