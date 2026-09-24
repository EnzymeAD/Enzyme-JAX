#ifndef ENZYME_AD_JAX_PASSES_DISTRIBUTED_SEARCHSTRATEGYRUNNERS_H
#define ENZYME_AD_JAX_PASSES_DISTRIBUTED_SEARCHSTRATEGYRUNNERS_H

#include "src/enzyme_ad/jax/Passes/Distributed/BeamSearchDriver.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <queue>
#include <random>
#include <utility>
#include <vector>

// Search strategies other than beam. Each takes the root, the explorer and the
// scorer, spends a budget of scorer calls, and returns the best finalized node.
// `stop` is checked between evaluations so fail-on-bad-candidate halts them
// early, the same way it halts the beam driver via BeamSearchQueueBase::abort.
// Per-candidate logging happens for free inside the scorer itself
// (StrategyScorer::score, gated on distributed.logging_root) whenever a
// runner calls scorer.score(), so nothing here needs its own metrics hook.

namespace mlir::enzyme::distributed {

using SurrogateSample = std::pair<std::vector<uint64_t>, double>;

// Additive surrogate over the decision sequence: feature (axis, option) means
// the option-th child was taken at that axis.
struct SurrogateModel {
  static uint64_t key(size_t axis, size_t option) {
    return (static_cast<uint64_t>(axis) << 32) | static_cast<uint32_t>(option);
  }

  llvm::DenseMap<uint64_t, unsigned> index;
  std::vector<double> weights;
  bool valid = false;
  // Fit log(cost) so every prediction stays positive. A linear fit predicts
  // negative costs for a few percent of candidates, and top-K selection lands
  // on exactly those.
  bool logTarget = true;

  // Unseen features contribute 0; rejecting them would bias the pool.
  double weightFor(size_t axis, size_t option) const {
    auto it = index.find(key(axis, option));
    return it == index.end() ? 0.0 : weights[it->second];
  }

  double predict(llvm::ArrayRef<uint64_t> path) const {
    double sum = 0;
    for (uint64_t k : path) {
      auto it = index.find(k);
      if (it != index.end())
        sum += weights[it->second];
    }
    return logTarget ? std::exp(sum) : sum;
  }

  void dump(llvm::StringRef path,
            llvm::ArrayRef<SurrogateSample> samples) const {
    std::error_code ec;
    llvm::raw_fd_ostream os(path, ec, llvm::sys::fs::OF_Text);
    if (ec) {
      llvm::errs()
          << "distributed-search-strategies: cannot write surrogate dump '"
          << path << "': " << ec.message() << "\n";
      return;
    }
    os << "# log_target=" << (logTarget ? 1 : 0) << " features=" << index.size()
       << " samples=" << samples.size() << "\n";
    os << "weight,axis,option,value\n";
    for (const auto &kv : index)
      os << "w," << (kv.first >> 32) << ',' << (uint32_t)kv.first << ','
         << llvm::format("%.17g", weights[kv.second]) << '\n';
    for (const auto &smp : samples) {
      os << "s," << llvm::format("%.17g", smp.second);
      for (uint64_t k : smp.first)
        os << ',' << (k >> 32) << ':' << (uint32_t)k;
      os << '\n';
    }
  }

  // Ridge on the normal equations, solved by Cholesky; the indicator matrix is
  // rank-deficient because path lengths vary. Lambda scales with the trace.
  bool fit(llvm::ArrayRef<SurrogateSample> samples) {
    index.clear();
    weights.clear();
    valid = false;
    for (const auto &s : samples)
      for (uint64_t k : s.first)
        index.insert({k, static_cast<unsigned>(index.size())});
    const unsigned p = index.size();
    if (p == 0 || samples.size() < p / 4) {
      // A populated index with no weights would make weightFor and predict
      // read past the end; callers check `valid`, but the class should not
      // depend on that.
      index.clear();
      return false;
    }

    std::vector<double> A(static_cast<size_t>(p) * p, 0.0), b(p, 0.0);
    std::vector<double> row(p);
    for (const auto &s : samples) {
      // No logarithm for a non-positive cost; a NaN would mis-rank the pool.
      if (logTarget && !(s.second > 0.0))
        continue;
      double target = logTarget ? std::log(s.second) : s.second;
      std::fill(row.begin(), row.end(), 0.0);
      for (uint64_t k : s.first)
        row[index.find(k)->second] += 1.0;
      for (unsigned i = 0; i < p; ++i) {
        if (row[i] == 0.0)
          continue;
        b[i] += row[i] * target;
        for (unsigned j = 0; j <= i; ++j)
          A[static_cast<size_t>(i) * p + j] += row[i] * row[j];
      }
    }
    double trace = 0;
    for (unsigned i = 0; i < p; ++i)
      trace += A[static_cast<size_t>(i) * p + i];
    double lam = 1e-6 * std::max(1.0, trace / p);
    for (unsigned i = 0; i < p; ++i)
      A[static_cast<size_t>(i) * p + i] += lam;

    // Not positive definite means abandon the fit rather than mis-rank.
    for (unsigned i = 0; i < p; ++i) {
      for (unsigned j = 0; j <= i; ++j) {
        double sum = A[static_cast<size_t>(i) * p + j];
        for (unsigned k = 0; k < j; ++k)
          sum -= A[static_cast<size_t>(i) * p + k] *
                 A[static_cast<size_t>(j) * p + k];
        if (i == j) {
          if (sum <= 0.0 || !std::isfinite(sum)) {
            index.clear();
            return false;
          }
          A[static_cast<size_t>(i) * p + i] = std::sqrt(sum);
        } else {
          A[static_cast<size_t>(i) * p + j] =
              sum / A[static_cast<size_t>(j) * p + j];
        }
      }
    }
    weights.assign(p, 0.0);
    for (unsigned i = 0; i < p; ++i) {
      double sum = b[i];
      for (unsigned k = 0; k < i; ++k)
        sum -= A[static_cast<size_t>(i) * p + k] * weights[k];
      weights[i] = sum / A[static_cast<size_t>(i) * p + i];
    }
    for (unsigned ii = p; ii-- > 0;) {
      double sum = weights[ii];
      for (unsigned k = ii + 1; k < p; ++k)
        sum -= A[static_cast<size_t>(k) * p + ii] * weights[k];
      weights[ii] = sum / A[static_cast<size_t>(ii) * p + ii];
    }
    for (double w : weights)
      if (!std::isfinite(w)) {
        index.clear();
        weights.clear();
        return false;
      }
    valid = true;
    return true;
  }
};

// Random walk to a leaf. `path`, when non-null, collects the (axis, chosen
// option) feature key at every step, for surrogate fitting/ranking.
template <typename Node>
std::shared_ptr<Node> randomDescent(std::shared_ptr<Node> node,
                                    BeamSearchExplorerBase<Node> &explorer,
                                    std::mt19937 &rng,
                                    std::vector<uint64_t> *path) {
  if (path)
    path->clear();
  while (!node->finalized()) {
    auto children = explorer.generateCandidatesFromNode(node);
    if (children.empty())
      break;
    std::uniform_int_distribution<size_t> pick(0, children.size() - 1);
    size_t choice = pick(rng);
    auto child = children[choice];
    if (path)
      path->push_back(SurrogateModel::key(child->getAxisIndex(), choice));
    node = child;
  }
  return node;
}

// Scores one leaf and keeps the best.
template <typename Node>
double scoreLeaf(const std::shared_ptr<Node> &leaf,
                 BeamSearchScorerBase<Node> &scorer,
                 std::shared_ptr<Node> &best) {
  double sc = scorer.score(leaf);
  leaf->score = sc;
  if (!best || sc > best->score)
    best = leaf;
  return sc;
}

// Real evaluations: they count against the budget and update `best`. Each
// decision path is kept for the fit.
template <typename Node>
std::vector<SurrogateSample> collectTrainingSamples(
    std::shared_ptr<Node> root, BeamSearchExplorerBase<Node> &explorer,
    BeamSearchScorerBase<Node> &scorer, unsigned count, std::mt19937 &rng,
    const bool &stop, std::shared_ptr<Node> &best) {
  std::vector<SurrogateSample> out;
  out.reserve(count);
  std::vector<uint64_t> path;
  for (unsigned i = 0; i < count && !stop; ++i) {
    auto leaf = randomDescent(root, explorer, rng, &path);
    if (!leaf->finalized())
      continue;
    double sc = scoreLeaf(leaf, scorer, best);
    // The surrogate is fit on cost (-score), so a weight is what a decision
    // adds to the predicted cost.
    if (std::isfinite(sc))
      out.push_back({path, -sc});
  }
  return out;
}

template <typename Node>
void randomRolloutsInto(std::shared_ptr<Node> root,
                        BeamSearchExplorerBase<Node> &explorer,
                        BeamSearchScorerBase<Node> &scorer, unsigned count,
                        std::mt19937 &rng, const bool &stop,
                        std::shared_ptr<Node> &best) {
  for (unsigned s = 0; s < count && !stop; ++s) {
    auto leaf = randomDescent(root, explorer, rng, /*path=*/nullptr);
    if (leaf->finalized())
      scoreLeaf(leaf, scorer, best);
  }
}

// Best of `samples` independent random rollouts. Serves as a control
// baseline, since it samples the tree independently of any scoring feedback,
// unlike the beam, which only visits what its own scores led it to.
template <typename Node>
std::shared_ptr<Node>
runRandom(std::shared_ptr<Node> root, BeamSearchExplorerBase<Node> &explorer,
          BeamSearchScorerBase<Node> &scorer, unsigned samples, unsigned seed,
          const bool &stop) {
  std::mt19937 rng(seed);
  std::shared_ptr<Node> best;
  randomRolloutsInto(root, explorer, scorer, samples, rng, stop, best);
  return best;
}

// Fit a surrogate on a training sample, then exactly evaluate the best K of a
// much larger cheaply ranked pool. Budget is trainSamples + K.
template <typename Node>
std::shared_ptr<Node>
runTwoStage(std::shared_ptr<Node> root, BeamSearchExplorerBase<Node> &explorer,
            BeamSearchScorerBase<Node> &scorer, unsigned budget, unsigned seed,
            unsigned trainSamples, unsigned poolSize, bool logTarget,
            llvm::StringRef dumpPath, const bool &stop) {
  std::mt19937 rng(seed);
  std::shared_ptr<Node> best;
  trainSamples = std::min(trainSamples, budget);
  auto training = collectTrainingSamples(root, explorer, scorer, trainSamples,
                                         rng, stop, best);

  const unsigned K = budget - trainSamples;
  SurrogateModel model;
  model.logTarget = logTarget;
  if (K > 0 && !stop && !model.fit(training)) {
    llvm::errs() << "distributed-search-strategies: surrogate fit failed on "
                 << training.size() << " samples; remaining " << K
                 << " evaluations fall back to random rollouts\n";
    randomRolloutsInto(root, explorer, scorer, K, rng, stop, best);
  } else if (K > 0 && !stop) {
    if (!dumpPath.empty())
      model.dump(dumpPath, training);

    // Only K nodes stay pinned; the allocator recycles the rest.
    struct Candidate {
      double predicted;
      std::shared_ptr<Node> node;
    };
    auto worseFirst = [](const Candidate &a, const Candidate &b) {
      return a.predicted < b.predicted; // max-heap on predicted cost
    };
    std::priority_queue<Candidate, std::vector<Candidate>, decltype(worseFirst)>
        keep(worseFirst);

    std::vector<uint64_t> path;
    for (unsigned i = 0; i < poolSize; ++i) {
      auto leaf = randomDescent(root, explorer, rng, &path);
      if (!leaf->finalized())
        continue;
      double pred = model.predict(path);
      if (keep.size() < K) {
        keep.push({pred, leaf});
      } else if (pred < keep.top().predicted) {
        keep.pop();
        keep.push({pred, leaf});
      }
    }

    while (!keep.empty() && !stop) {
      Candidate c = keep.top();
      keep.pop();
      scoreLeaf(c.node, scorer, best);
    }
  }
  return best;
}

// Kept outside the node. The children vector owns the tree, so addresses are
// never recycled under the map. Keys are stable; values are not, since the map
// moves them when it grows, so nothing holds a reference to one across a
// lookup that could trigger a rehash (see the expand/select loop below).
template <typename Node> struct UctStats {
  unsigned visits = 0;
  double sumValue = 0;
  double bestValue = -std::numeric_limits<double>::infinity();
  std::vector<std::shared_ptr<Node>> children;
  bool expanded = false;
};

// Monte Carlo Tree Search (UCT). Rollouts are random because the completion
// heuristic is deterministic, and ties break at random because sibling UCB
// values tie often. An iteration that lands on a node already scored in an
// earlier iteration reuses that score rather than evaluating again, so the
// number of real evaluations can be less than `iterations`.
template <typename Node>
std::shared_ptr<Node>
runMcts(std::shared_ptr<Node> root, BeamSearchExplorerBase<Node> &explorer,
        BeamSearchScorerBase<Node> &scorer, unsigned iterations, unsigned seed,
        double explorationC, bool maxBackup, double surrogateWeight,
        unsigned surrogateTrain, bool logTarget, const bool &stop) {
  std::mt19937 rng(seed);
  std::shared_ptr<Node> best;

  // Optional PUCT-style prior on the UCB score. Off by default.
  SurrogateModel model;
  model.logTarget = logTarget;
  if (surrogateWeight != 0.0 && surrogateTrain > 0) {
    surrogateTrain = std::min(surrogateTrain, iterations);
    auto training = collectTrainingSamples(root, explorer, scorer,
                                           surrogateTrain, rng, stop, best);
    iterations -= surrogateTrain; // training rollouts are real evaluations
    if (!model.fit(training))
      llvm::errs() << "distributed-search-strategies: surrogate fit failed on "
                   << training.size()
                   << " samples; MCTS continues without a prior\n";
  }
  llvm::DenseMap<const Node *, UctStats<Node>> stats;
  // Normalised into [0,1]; a constant means nothing against raw magnitudes.
  double obsMin = std::numeric_limits<double>::infinity();
  double obsMax = -std::numeric_limits<double>::infinity();

  // Returns the children by value. A later lookup inserts, and DenseMap moves
  // every value when it grows, so a reference into a bucket does not survive
  // one.
  auto expand = [&](const std::shared_ptr<Node> &n) {
    UctStats<Node> &st = stats[n.get()];
    if (!st.expanded && !n->finalized()) {
      st.children = explorer.generateCandidatesFromNode(n);
      st.expanded = true;
    }
    return st.children;
  };

  auto normalise = [&](double v) {
    if (!std::isfinite(v))
      return 0.0;
    if (obsMax <= obsMin)
      return 0.5;
    return (v - obsMin) / (obsMax - obsMin);
  };

  for (unsigned iter = 0; iter < iterations && !stop; ++iter) {
    // Select.
    std::vector<std::shared_ptr<Node>> path;
    auto node = root;
    path.push_back(node);
    while (!node->finalized()) {
      std::vector<std::shared_ptr<Node>> children = expand(node);
      if (children.empty())
        break;
      // Visit every child once before trusting any UCB comparison.
      std::vector<std::shared_ptr<Node>> unvisited;
      for (auto &c : children)
        if (stats[c.get()].visits == 0)
          unvisited.push_back(c);
      if (!unvisited.empty()) {
        std::uniform_int_distribution<size_t> pick(0, unvisited.size() - 1);
        node = unvisited[pick(rng)];
        path.push_back(node);
        break;
      }
      // Normalised within this sibling set only, since predicted magnitude
      // depends on the axis.
      std::vector<double> prior(children.size(), 0.0);
      if (model.valid && surrogateWeight != 0.0) {
        double lo = std::numeric_limits<double>::infinity();
        double hi = -std::numeric_limits<double>::infinity();
        for (size_t i = 0; i < children.size(); ++i) {
          prior[i] = model.weightFor(children[i]->getAxisIndex(), i);
          lo = std::min(lo, prior[i]);
          hi = std::max(hi, prior[i]);
        }
        for (double &p : prior)
          p = (hi > lo) ? (hi - p) / (hi - lo) : 0.5; // cheaper => larger
      }

      // Read once: the lookups below insert, which can move this value.
      unsigned parentVisits = stats[node.get()].visits;
      double bestU = -std::numeric_limits<double>::infinity();
      std::vector<std::shared_ptr<Node>> tied;
      for (size_t i = 0; i < children.size(); ++i) {
        auto &c = children[i];
        UctStats<Node> &cs = stats[c.get()];
        double exploit =
            normalise(maxBackup ? cs.bestValue : cs.sumValue / cs.visits);
        double explore = explorationC *
                         std::sqrt(std::log((double)parentVisits) / cs.visits);
        double u = exploit + explore + surrogateWeight * prior[i];
        if (u > bestU + 1e-12) {
          bestU = u;
          tied.assign(1, c);
        } else if (std::abs(u - bestU) <= 1e-12) {
          tied.push_back(c);
        }
      }
      if (tied.empty())
        break;
      std::uniform_int_distribution<size_t> pick(0, tied.size() - 1);
      node = tied[pick(rng)];
      path.push_back(node);
    }

    // Simulate.
    auto leaf = node;
    while (!leaf->finalized()) {
      auto children = explorer.generateCandidatesFromNode(leaf);
      if (children.empty())
        break;
      std::uniform_int_distribution<size_t> pick(0, children.size() - 1);
      leaf = children[pick(rng)];
    }
    if (!leaf->finalized())
      continue;

    // The select loop's UCB comparison can revisit the same finalized node
    // on a later iteration. Since scorer.score() clones the module and runs
    // the full nested lowering pipeline, a revisit reuses the value
    // recorded on the node's first visit instead of paying for that again.
    //
    // `visits > 0` on a finalized node is a reliable revisit check because:
    //   - backprop (below) is what increments visits, so it's only nonzero
    //     once a select path has ended here and been scored.
    //   - a finalized node only ever appears as the last element of a
    //     select path, never as an interior one.
    UctStats<Node> &leafStats = stats[leaf.get()];
    double value;
    if (leafStats.visits > 0) {
      value = leaf->score;
    } else {
      value = scoreLeaf(leaf, scorer, best);
      if (std::isfinite(value)) {
        obsMin = std::min(obsMin, value);
        obsMax = std::max(obsMax, value);
      }
    }

    // Backprop.
    for (auto &n : path) {
      UctStats<Node> &st = stats[n.get()];
      st.visits++;
      if (std::isfinite(value)) {
        st.sumValue += value;
        st.bestValue = std::max(st.bestValue, value);
      }
    }
  }
  return best;
}

} // namespace mlir::enzyme::distributed

#endif // ENZYME_AD_JAX_PASSES_DISTRIBUTED_SEARCHSTRATEGYRUNNERS_H
