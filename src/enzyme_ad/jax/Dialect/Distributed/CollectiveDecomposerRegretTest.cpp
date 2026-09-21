// Regret harness for the heuristic step orderer and search budget (PlanBudget
// in CollectiveDecomposer.h). It generates small synthetic collectives directly
// as NormalizedCollective values (no MLIR), plans each one exactly and under
// budgets, and reports how often the budgeted chain matches the exact cost and
// the worst cost ratio.

#include "src/enzyme_ad/jax/Dialect/Distributed/CollectiveDecomposer.h"

#include "gtest/gtest.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <optional>
#include <random>
#include <string>
#include <vector>

namespace mlir::enzyme::distributed {
namespace {

// A generated collective with the mesh parameters it is costed under.
struct Case {
  NormalizedCollective collective;
  MeshCostParams params;
  std::string description;
};

AtomLabel label(AtomSpace space, size_t axis, size_t atom) {
  return {space, axis, atom};
}

// Builds random collectives out of independent gadgets, each of which is a
// well-formed row (or component) of the normal form: every role the decomposer
// handles occurs, and gadgets of equal extent on the same axis give the
// symmetric atoms that symmetry dedupe acts on.
class CaseGenerator {
public:
  explicit CaseGenerator(uint32_t seed) : rng(seed) {}

  Case generate() {
    size_t numAxes = 2 + pick(3);
    std::vector<uint64_t> axisExtents(numAxes, 1);
    std::vector<double> bandwidth;
    bool uniform = pick(3) == 0;
    for (size_t a = 0; a < numAxes; ++a)
      bandwidth.push_back(uniform ? 1.0 : double(1u << pick(3)));

    NormalizedCollective collective;
    std::string description;
    size_t nextTileDim = 0;
    int64_t inputElements = 1;
    size_t gadgets = 1 + pick(4);
    std::vector<size_t> atomsOnAxis(numAxes, 0);

    // Appends a mesh atom of `extent` on a random axis and returns its index.
    auto addAtom = [&](uint64_t extent, MeshAtomRole in, MeshAtomRole out) {
      size_t axis = pick(numAxes);
      MeshAtom atom;
      atom.axis = axis;
      atom.atom = atomsOnAxis[axis]++;
      atom.extent = extent;
      atom.stride = axisExtents[axis];
      axisExtents[axis] *= extent;
      atom.in = in;
      atom.out = out;
      collective.meshAtoms.push_back(atom);
      return collective.meshAtoms.size() - 1;
    };
    auto meshRole = [&](size_t index) {
      const MeshAtom &atom = collective.meshAtoms[index];
      return MeshAtomRole{AtomRole::Mesh,
                          label(AtomSpace::Mesh, atom.axis, atom.atom)};
    };
    auto atomLabel = [&](size_t index) {
      const MeshAtom &atom = collective.meshAtoms[index];
      return label(AtomSpace::Mesh, atom.axis, atom.atom);
    };
    // A tile atom paired with mesh atom `mesh`.
    auto addTile = [&](bool isInput, uint64_t extent, size_t mesh) {
      size_t dim = nextTileDim++;
      collective.tileAtoms.push_back(
          {isInput, dim, 0, extent, atomLabel(mesh)});
      if (isInput)
        inputElements *= extent;
      return label(isInput ? AtomSpace::InTile : AtomSpace::OutTile, dim, 0);
    };

    for (size_t g = 0; g < gadgets; ++g) {
      uint64_t extent = pick(2) ? 2 : 4;
      switch (pick(9)) {
      case 0: { // (Reduced, Replicate): all-reduce, or a peel
        addAtom(extent, {AtomRole::Reduced}, {AtomRole::Replicate});
        description += "AR ";
        break;
      }
      case 1: { // (Reduced, Tile): reduce-scatter
        size_t m = addAtom(extent, {AtomRole::Reduced}, {AtomRole::Replicate});
        collective.meshAtoms[m].out = {AtomRole::Tile,
                                       addTile(true, extent, m)};
        description += "RS ";
        break;
      }
      case 2: { // (Tile, Replicate): all-gather
        size_t m =
            addAtom(extent, {AtomRole::Replicate}, {AtomRole::Replicate});
        collective.meshAtoms[m].in = {AtomRole::Tile,
                                      addTile(false, extent, m)};
        description += "AG ";
        break;
      }
      case 3: { // (Tile, Tile): all-to-all or gather + slice
        size_t m =
            addAtom(extent, {AtomRole::Replicate}, {AtomRole::Replicate});
        collective.meshAtoms[m].in = {AtomRole::Tile,
                                      addTile(false, extent, m)};
        collective.meshAtoms[m].out = {AtomRole::Tile,
                                       addTile(true, extent, m)};
        description += "A2A ";
        break;
      }
      case 4: { // (Replicate, Tile): free slice
        size_t m =
            addAtom(extent, {AtomRole::Replicate}, {AtomRole::Replicate});
        collective.meshAtoms[m].out = {AtomRole::Tile,
                                       addTile(true, extent, m)};
        description += "SL ";
        break;
      }
      case 5: { // idle atom (Replicate, Replicate): a conjugation target
        addAtom(extent, {AtomRole::Replicate}, {AtomRole::Replicate});
        description += "IDLE ";
        break;
      }
      case 6: { // a swap of two atoms: one permute
        size_t a =
            addAtom(extent, {AtomRole::Replicate}, {AtomRole::Replicate});
        size_t b =
            addAtom(extent, {AtomRole::Replicate}, {AtomRole::Replicate});
        collective.meshAtoms[a].in = collective.meshAtoms[a].out = meshRole(b);
        collective.meshAtoms[b].in = collective.meshAtoms[b].out = meshRole(a);
        description += "SWAP ";
        break;
      }
      case 7: { // path: a (Reduced, Mesh b), b (Mesh a, Replicate)
        size_t a = addAtom(extent, {AtomRole::Reduced}, {AtomRole::Replicate});
        size_t b =
            addAtom(extent, {AtomRole::Replicate}, {AtomRole::Replicate});
        collective.meshAtoms[a].out = meshRole(b);
        collective.meshAtoms[b].in = meshRole(a);
        description += "PATH ";
        break;
      }
      case 8: { // a (Tile, Mesh b), b (Mesh a, Replicate): gather before slice
        size_t a =
            addAtom(extent, {AtomRole::Replicate}, {AtomRole::Replicate});
        size_t b =
            addAtom(extent, {AtomRole::Replicate}, {AtomRole::Replicate});
        collective.meshAtoms[a].in = {AtomRole::Tile,
                                      addTile(false, extent, a)};
        collective.meshAtoms[a].out = meshRole(b);
        collective.meshAtoms[b].in = meshRole(a);
        description += "GS ";
        break;
      }
      }
    }
    // A pass-through tile dimension carries every mesh extent as a factor, so
    // any combination of peels and scatters divides the payload evenly. The
    // decomposer does not check that a peel (D17) and a reduce-scatter
    // that both draw on the same payload bytes fit together; that is outside
    // what this harness measures.
    uint64_t passthrough = 1;
    for (const MeshAtom &atom : collective.meshAtoms)
      passthrough *= atom.extent;
    int64_t elementBytes = 1 + pick(4);
    collective.payloadBytes = elementBytes * inputElements;
    if (passthrough > 1) {
      size_t dimIn = nextTileDim++, dimOut = nextTileDim++;
      collective.tileAtoms.push_back(
          {true, dimIn, 0, passthrough, label(AtomSpace::OutTile, dimOut, 0)});
      collective.tileAtoms.push_back(
          {false, dimOut, 0, passthrough, label(AtomSpace::InTile, dimIn, 0)});
      collective.payloadBytes *= passthrough;
    }
    collective.reductionKind = ReductionKind::Add;
    for (uint64_t &e : axisExtents)
      e = std::max<uint64_t>(e, 1);
    return {std::move(collective),
            MeshCostParams(axisExtents, bandwidth,
                           std::vector<double>(numAxes, 0.01), 0.1),
            description};
  }

private:
  size_t pick(size_t n) { return rng() % n; }
  std::mt19937 rng;
};

double chainCost(const std::vector<PrimitiveStep> &chain) {
  double total = 0;
  for (const PrimitiveStep &step : chain)
    total += step.isolatedDuration();
  return total;
}

struct Config {
  std::string name;
  size_t kOrder, kVariant;
};

struct Tally {
  size_t cases = 0, exact = 0, failed = 0;
  double worst = 1.0, sum = 0.0;
  std::string worstCase;
};

// Regret of every configuration over generated collectives. Prints a report and
// asserts that the pieces of the orderer that claim exactness are exact and
// that every returned chain passes the independent semantic check.
TEST(CollectiveDecomposerRegret, GeneratedCollectives) {
  const size_t kUnlimited = PlanBudget::kUnlimited;
  std::vector<Config> configs;
  for (size_t k : {1, 2, 4})
    configs.push_back({"k_order=" + std::to_string(k), k, kUnlimited});
  for (size_t k : {1, 2, 4})
    configs.push_back({"k_variant=" + std::to_string(k), kUnlimited, k});
  for (size_t k : {1, 2, 4})
    configs.push_back({"both=" + std::to_string(k), k, k});
  configs.push_back({"k_order=inf (tiers only)", kUnlimited, kUnlimited});

  std::vector<Tally> tallies(configs.size());
  size_t total = 0, unsupported = 0;
  for (uint32_t seed = 1; seed <= 400; ++seed) {
    Case c = CaseGenerator(seed).generate();
    std::string reason;
    std::optional<std::vector<PrimitiveStep>> exact =
        plan(c.collective, c.params, reason);
    if (!exact) {
      ++unsupported;
      continue;
    }
    ++total;
    double exactCost = chainCost(*exact);
    for (size_t i = 0; i < configs.size(); ++i) {
      PlanOptions options;
      PlanBudget budget;
      budget.kOrder = configs[i].kOrder;
      budget.kVariant = configs[i].kVariant;
      options.budget = budget;
      std::optional<std::vector<PrimitiveStep>> chain =
          plan(c.collective, c.params, reason, options);
      Tally &tally = tallies[i];
      ++tally.cases;
      if (!chain) {
        ++tally.failed;
        continue;
      }
      std::string why;
      EXPECT_TRUE(verifyChainRealizesCollective(c.collective, *chain, why))
          << configs[i].name << " seed " << seed << ": " << why;
      double ratio = exactCost == 0 ? 1.0 : chainCost(*chain) / exactCost;
      if (ratio < 1.0 + 1e-9 && ratio > 1.0 - 1e-9)
        ++tally.exact;
      tally.sum += ratio;
      if (ratio > tally.worst) {
        tally.worst = ratio;
        tally.worstCase = "seed " + std::to_string(seed) + " " + c.description;
      }
      // With every budget field unlimited the orderer restricts the search
      // to tier order only, which is valid on every generated case (see
      // orderCandidates), so its optimum equals the exact one.
      if (configs[i].kOrder == kUnlimited && configs[i].kVariant == kUnlimited)
        EXPECT_NEAR(chainCost(*chain), exactCost, 1e-9 * (1 + exactCost))
            << "tier order lost the optimum, seed " << seed << " "
            << c.description;
    }
  }
  std::printf("\nregret harness: %zu supported collectives (%zu unsupported)\n",
              total, unsupported);
  std::printf("%-26s %8s %8s %10s %10s  worst case\n", "config", "exact%",
              "failed", "mean", "worst");
  for (size_t i = 0; i < configs.size(); ++i) {
    const Tally &t = tallies[i];
    size_t ok = t.cases - t.failed;
    std::printf("%-26s %7.1f%% %8zu %10.4f %10.4f  %s\n",
                configs[i].name.c_str(),
                100.0 * t.exact / std::max<size_t>(1, ok), t.failed,
                ok ? t.sum / ok : 0.0, t.worst, t.worstCase.c_str());
    EXPECT_EQ(t.failed, 0u) << configs[i].name;
    EXPECT_LE(t.worst, 2.0) << configs[i].name;
  }
  EXPECT_GT(total, 100u);
}

MeshAtom meshAtom(size_t axis, MeshAtomRole in, MeshAtomRole out) {
  return {axis, 0, 2, 1, in, out};
}
MeshAtomRole mesh(size_t axis) {
  return {AtomRole::Mesh, label(AtomSpace::Mesh, axis, 0)};
}

// A component with several non-baseline variants (a path closure and one
// conjugation per gather or all-reduce unit) next to two idle atoms on faster
// axes: keeping only the top-ranked variant must still find the exact
// optimum, the variant the exact search picks.
TEST(CollectiveDecomposerRegret, VariantRankingKeepsTheOptimum) {
  MeshCostParams params({2, 2, 2, 2}, {1, 1, 4, 16}, {0.01, 0.01, 0.01, 0.01},
                        0.1);
  std::vector<NormalizedCollective> collectives(2);
  // A reduce whose result moves to another atom (D15 against the half-split).
  collectives[0].meshAtoms = {
      meshAtom(0, {AtomRole::Reduced}, mesh(1)),
      meshAtom(1, mesh(0), {AtomRole::Replicate}),
      meshAtom(2, {AtomRole::Replicate}, {AtomRole::Replicate}),
      meshAtom(3, {AtomRole::Replicate}, {AtomRole::Replicate})};
  collectives[0].payloadBytes = 8;
  collectives[0].reductionKind = ReductionKind::Add;
  // A gather whose digit is placed on another atom.
  collectives[1].meshAtoms = {
      meshAtom(0, {AtomRole::Tile, label(AtomSpace::OutTile, 0, 0)}, mesh(1)),
      meshAtom(1, mesh(0), {AtomRole::Replicate}),
      meshAtom(2, {AtomRole::Replicate}, {AtomRole::Replicate}),
      meshAtom(3, {AtomRole::Replicate}, {AtomRole::Replicate})};
  collectives[1].tileAtoms = {{false, 0, 0, 2, label(AtomSpace::Mesh, 0, 0)}};
  collectives[1].payloadBytes = 4;
  for (const NormalizedCollective &collective : collectives) {
    std::string reason;
    auto exact = plan(collective, params, reason);
    ASSERT_TRUE(exact.has_value()) << reason;
    for (size_t k : {1, 2}) {
      PlanOptions options;
      options.budget = PlanBudget();
      options.budget->kVariant = k;
      auto ranked = plan(collective, params, reason, options);
      ASSERT_TRUE(ranked.has_value()) << reason;
      std::printf("variant ranking k_variant=%zu: exact %g, ranked %g\n", k,
                  chainCost(*exact), chainCost(*ranked));
      EXPECT_NEAR(chainCost(*ranked), chainCost(*exact), 1e-9);
    }
  }
}

// max_states turns an oversized search into a clean failure with a reason.
TEST(CollectiveDecomposerRegret, StateBudgetFailsCleanly) {
  std::optional<Case> found;
  for (uint32_t seed = 1; seed <= 400 && !found; ++seed) {
    Case candidate = CaseGenerator(seed).generate();
    if (candidate.collective.meshAtoms.size() >= 4)
      found = std::move(candidate);
  }
  ASSERT_TRUE(found.has_value());
  Case &c = *found;
  PlanOptions options;
  PlanBudget budget;
  budget.maxStates = 1;
  options.budget = budget;
  std::string reason;
  std::optional<std::vector<PrimitiveStep>> chain =
      plan(c.collective, c.params, reason, options);
  EXPECT_FALSE(chain.has_value());
  EXPECT_NE(reason.find("budget"), std::string::npos) << reason;
}

} // namespace
} // namespace mlir::enzyme::distributed
