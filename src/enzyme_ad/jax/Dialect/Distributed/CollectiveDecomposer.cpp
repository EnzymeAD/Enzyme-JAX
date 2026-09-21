#include "CollectiveDecomposer.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ErrorHandling.h"

#include <cassert>
#include <limits>
#include <map>
#include <set>

namespace mlir::enzyme::distributed {

namespace {

// The units of work the DP orders (D7). Each unit owns one atom, except the
// permute group, which owns every atom of a nontrivial cycle.
enum class UnitKind { Slice, Gather, TileToTile, Permute };

struct Unit {
  UnitKind kind;
  std::vector<StepAtom> atoms;
  // Extent of the (single) atom; unused by Permute.
  uint64_t extent;

  // The stage at which the unit has nothing left to do. Slice, Gather and
  // Permute have stages {pending, done}; TileToTile adds the intermediate
  // {gathered, slice pending}.
  uint8_t finalStage() const { return kind == UnitKind::TileToTile ? 2 : 1; }
};

StepAtom stepAtomOf(const MeshAtom &atom) {
  return {atom.axis, atom.atom, atom.extent};
}

// Splits the mesh atoms into units, or reports why the collective is outside
// the supported rows.
bool buildUnits(const NormalizedCollective &collective,
                std::vector<Unit> &units, std::string &failureReason) {
  auto describeAtom = [](const MeshAtom &atom) {
    return "mesh" + std::to_string(atom.axis) + "." + std::to_string(atom.atom);
  };
  std::vector<StepAtom> permuted;
  for (const MeshAtom &atom : collective.meshAtoms) {
    switch (classify(atom)) {
    case RolePair::NoOp:
      break;
    case RolePair::LocalSlice:
      units.push_back({UnitKind::Slice, {stepAtomOf(atom)}, atom.extent});
      break;
    case RolePair::AllGather:
      units.push_back({UnitKind::Gather, {stepAtomOf(atom)}, atom.extent});
      break;
    case RolePair::TileToTile:
      units.push_back({UnitKind::TileToTile, {stepAtomOf(atom)}, atom.extent});
      break;
    case RolePair::Permute:
      if (atom.in.kind != AtomRole::Mesh || atom.out.kind != AtomRole::Mesh) {
        failureReason = "unsupported mixed row on " + describeAtom(atom) +
                        " (in " + toString(atom.in.kind) + ", out " +
                        toString(atom.out.kind) + ")";
        return false;
      }
      permuted.push_back(stepAtomOf(atom));
      break;
    case RolePair::AllReduce:
    case RolePair::ReduceScatter:
    case RolePair::ReduceThenPermute:
      failureReason = "unsupported reduction on " + describeAtom(atom) + " (" +
                      toString(classify(atom)) + ")";
      return false;
    }
  }
  if (!permuted.empty())
    units.push_back({UnitKind::Permute, std::move(permuted), 0});

  // Tile atoms must be moved by a mesh atom or relabeled to the other side's
  // tile; anything else (e.g. output data drawn from nowhere) has no
  // primitive here.
  for (const TileAtom &tile : collective.tileAtoms) {
    bool ok = false;
    if (tile.partner) {
      AtomSpace space = tile.partner->space;
      ok = space == AtomSpace::Mesh ||
           space == (tile.isInput ? AtomSpace::OutTile : AtomSpace::InTile);
    }
    if (!ok) {
      failureReason = std::string("unsupported ") +
                      (tile.isInput ? "input" : "output") + " tile atom " +
                      std::to_string(tile.dim) + "." +
                      std::to_string(tile.atom) +
                      " (partner is not a mesh atom or the other tile)";
      return false;
    }
  }
  return true;
}

// Exact DP over DecomposerState (D6, D7). Candidate generation is separate
// from the recursion so a filter can rank or truncate candidates.
class ChainSearch {
public:
  ChainSearch(const std::vector<Unit> &units, int64_t payloadBytes,
              const MeshCostParams &params, const CandidateFilter &filter)
      : units(units), payloadBytes(payloadBytes), params(params),
        filter(filter) {}

  DecomposerState initialState() const {
    return {std::vector<uint8_t>(units.size(), 0)};
  }

  // Per-device bytes in `state` (D3).
  int64_t payloadOf(const DecomposerState &state) const {
    int64_t divisor = 1, factor = 1;
    for (size_t i = 0; i < units.size(); ++i) {
      uint8_t stage = state.stage[i];
      int64_t e = units[i].extent;
      switch (units[i].kind) {
      case UnitKind::Slice:
        if (stage == 1)
          divisor *= e;
        break;
      case UnitKind::Gather:
        if (stage == 1)
          factor *= e;
        break;
      case UnitKind::TileToTile:
        if (stage >= 1)
          factor *= e;
        if (stage == 2)
          divisor *= e;
        break;
      case UnitKind::Permute:
        break;
      }
    }
    // Every slice divides out a distinct input tile atom, all of which the
    // initial payload contains.
    assert(payloadBytes % divisor == 0 && "slices exceed the tile payload");
    return payloadBytes / divisor * factor;
  }

  // The steps available in `state`. A pending free step (D2) is returned
  // alone; otherwise there is one candidate per way to advance each unit.
  // Empty exactly when every unit is at its final stage.
  std::vector<CandidateStep> candidates(const DecomposerState &state) const {
    int64_t payload = payloadOf(state);
    std::vector<CandidateStep> result;
    for (size_t i = 0; i < units.size(); ++i) {
      const Unit &unit = units[i];
      bool sliceReady =
          (unit.kind == UnitKind::Slice && state.stage[i] == 0) ||
          (unit.kind == UnitKind::TileToTile && state.stage[i] == 1);
      if (!sliceReady)
        continue;
      DecomposerState next = advance(state, i, unit.finalStage());
      return {
          {localSliceFootprint(unit.atoms, payload, payloadOf(next), params),
           std::move(next)}};
    }
    for (size_t i = 0; i < units.size(); ++i) {
      const Unit &unit = units[i];
      if (state.stage[i] != 0)
        continue;
      switch (unit.kind) {
      case UnitKind::Slice:
        llvm_unreachable("slices are consumed as free steps above");
      case UnitKind::Gather:
        result.push_back({allGatherFootprint(unit.atoms, payload, params),
                          advance(state, i, 1)});
        break;
      case UnitKind::TileToTile:
        result.push_back({allToAllFootprint(unit.atoms, payload, params),
                          advance(state, i, 2)});
        result.push_back({allGatherFootprint(unit.atoms, payload, params),
                          advance(state, i, 1)});
        break;
      case UnitKind::Permute: {
        std::vector<double> change;
        for (const StepAtom &atom : unit.atoms)
          change.push_back(permuteSwapChangeFraction(atom.extent));
        result.push_back({permuteFootprint(unit.atoms, change, payload, params),
                          advance(state, i, 1)});
        break;
      }
      }
    }
    if (result.size() > 1 && filter) {
      filter(state, result);
      assert(!result.empty() && "the candidate filter dropped every candidate");
    }
    return result;
  }

  // Least summed isolated duration from `state` to completion.
  double solve(const DecomposerState &state) {
    if (auto it = memo.find(state); it != memo.end())
      return it->second.cost;
    std::vector<CandidateStep> options = candidates(state);
    Entry best{0.0, 0};
    if (!options.empty()) {
      best.cost = std::numeric_limits<double>::infinity();
      for (size_t i = 0; i < options.size(); ++i) {
        double cost =
            options[i].step.isolatedDuration() + solve(options[i].next);
        // Strict comparison keeps the earlier candidate on ties, so results
        // do not depend on floating point noise between equal-cost orders.
        if (cost < best.cost)
          best = {cost, i};
      }
    }
    memo.emplace(state, best);
    return best.cost;
  }

  // The chain achieving solve(initialState()).
  std::vector<PrimitiveStep> extractChain() {
    std::vector<PrimitiveStep> chain;
    DecomposerState state = initialState();
    solve(state);
    while (true) {
      std::vector<CandidateStep> options = candidates(state);
      if (options.empty())
        return chain;
      size_t choice = memo.at(state).choice;
      chain.push_back(options[choice].step);
      state = options[choice].next;
    }
  }

private:
  struct Entry {
    double cost;
    size_t choice;
  };

  static DecomposerState advance(DecomposerState state, size_t unit,
                                 uint8_t stage) {
    state.stage[unit] = stage;
    return state;
  }

  const std::vector<Unit> &units;
  int64_t payloadBytes;
  const MeshCostParams &params;
  const CandidateFilter &filter;
  std::map<DecomposerState, Entry> memo;
};

} // namespace

std::optional<std::vector<PrimitiveStep>>
plan(const NormalizedCollective &collective, const MeshCostParams &params,
     std::string &failureReason, const PlanOptions &options) {
  std::vector<Unit> units;
  if (!buildUnits(collective, units, failureReason))
    return std::nullopt;
  if (units.size() > kMaxLiveUnits) {
    failureReason = "too many units of work (" + std::to_string(units.size()) +
                    ") for the exact search";
    return std::nullopt;
  }

  ChainSearch search(units, collective.payloadBytes, params, options.filter);
  std::vector<PrimitiveStep> chain = search.extractChain();
#ifndef NDEBUG
  std::string why;
  assert(verifyChainRealizesCollective(collective, chain, why) &&
         "decomposition does not realize the collective");
#endif
  return chain;
}

namespace {

// The symbolic model behind verifyChainRealizesCollective. Digits are indexed
// mesh atoms first, then input tile atoms.
class ChainChecker {
public:
  explicit ChainChecker(const NormalizedCollective &collective)
      : collective(collective) {}

  bool run(const std::vector<PrimitiveStep> &chain, std::string &why) {
    if (!setup(why))
      return false;
    for (size_t i = 0; i < chain.size(); ++i)
      if (!applyStep(chain[i], why)) {
        why = "step " + std::to_string(i) + " (" + toString(chain[i].kind) +
              "): " + why;
        return false;
      }
    return checkFinal(why);
  }

private:
  static constexpr int kNone = -1;

  // Digit of the mesh atom `label`, or -1 when absent.
  int meshDigit(const AtomLabel &label) const {
    if (label.space != AtomSpace::Mesh)
      return kNone;
    for (size_t i = 0; i < collective.meshAtoms.size(); ++i)
      if (collective.meshAtoms[i].axis == label.axis &&
          collective.meshAtoms[i].atom == label.atom)
        return static_cast<int>(i);
    return kNone;
  }

  int inTileDigit(const AtomLabel &label) const {
    if (label.space != AtomSpace::InTile)
      return kNone;
    for (size_t i = 0; i < collective.tileAtoms.size(); ++i) {
      const TileAtom &tile = collective.tileAtoms[i];
      if (tile.isInput && tile.dim == label.axis && tile.atom == label.atom)
        return static_cast<int>(collective.meshAtoms.size() + i);
    }
    return kNone;
  }

  // The digit that a role brings to (out) an atom, or -1.
  int digitOfOutRole(const MeshAtomRole &role) const {
    switch (role.kind) {
    case AtomRole::Tile:
      return inTileDigit(role.partner);
    case AtomRole::Mesh:
      return meshDigit(role.partner);
    case AtomRole::Replicate:
    case AtomRole::Reduced:
      return kNone;
    }
    llvm_unreachable("covered switch");
  }

  int64_t localPayload() const {
    int64_t bytes = elementBytes;
    for (int digit : local)
      bytes *= extent[digit];
    return bytes;
  }

  bool setup(std::string &why) {
    size_t numMesh = collective.meshAtoms.size();
    size_t numDigits = numMesh + collective.tileAtoms.size();
    extent.assign(numDigits, 1);
    occupant.assign(numMesh, kNone);
    requiredOccupant.assign(numMesh, kNone);

    int64_t inputElements = 1;
    for (size_t i = 0; i < numMesh; ++i)
      extent[i] = collective.meshAtoms[i].extent;
    for (size_t i = 0; i < collective.tileAtoms.size(); ++i) {
      const TileAtom &tile = collective.tileAtoms[i];
      extent[numMesh + i] = tile.extent;
      if (tile.isInput) {
        local.insert(static_cast<int>(numMesh + i));
        inputElements *= tile.extent;
      }
    }
    assert(collective.payloadBytes % inputElements == 0);
    elementBytes = collective.payloadBytes / inputElements;

    for (size_t i = 0; i < numMesh; ++i) {
      const MeshAtom &atom = collective.meshAtoms[i];
      if (atom.in.kind == AtomRole::Reduced) {
        why = "reductions are not modelled";
        return false;
      }
      // An input digit that goes nowhere (replicated away) carries no data.
      if (atom.in.kind == AtomRole::Tile || atom.in.kind == AtomRole::Mesh)
        occupant[i] = static_cast<int>(i);
      requiredOccupant[i] = digitOfOutRole(atom.out);
    }
    int64_t outputElements = 1;
    for (size_t i = 0; i < collective.tileAtoms.size(); ++i) {
      const TileAtom &tile = collective.tileAtoms[i];
      if (tile.isInput)
        continue;
      outputElements *= tile.extent;
      if (!tile.partner) {
        why = "output tile atom without a source";
        return false;
      }
      int digit = tile.partner->space == AtomSpace::Mesh
                      ? meshDigit(*tile.partner)
                      : inTileDigit(*tile.partner);
      if (digit == kNone) {
        why = "output tile atom with an unsupported source";
        return false;
      }
      requiredLocal.insert(digit);
    }
    expectedFinalPayload = elementBytes * outputElements;
    return true;
  }

  // The index of the mesh atom a step atom names.
  bool findMeshAtom(const StepAtom &atom, size_t &index, std::string &why) {
    for (size_t i = 0; i < collective.meshAtoms.size(); ++i) {
      const MeshAtom &candidate = collective.meshAtoms[i];
      if (candidate.axis == atom.axis && candidate.atom == atom.atom) {
        if (candidate.extent != atom.extent) {
          why = "atom extent differs from the collective's";
          return false;
        }
        index = i;
        return true;
      }
    }
    why = "atom is not a mesh atom of the collective";
    return false;
  }

  bool applyStep(const PrimitiveStep &step, std::string &why) {
    if (step.payloadIn != localPayload()) {
      why = "payload in is " + std::to_string(step.payloadIn) + ", expected " +
            std::to_string(localPayload());
      return false;
    }
    if (!applySemantics(step, why))
      return false;
    if (step.payloadOut != localPayload()) {
      why = "payload out is " + std::to_string(step.payloadOut) +
            ", expected " + std::to_string(localPayload());
      return false;
    }
    return true;
  }

  bool applySemantics(const PrimitiveStep &step, std::string &why) {
    if (step.kind == PrimitiveKind::Permute)
      return applyPermute(step, why);
    if (step.atoms.size() != 1) {
      why = "expected exactly one atom";
      return false;
    }
    size_t x;
    if (!findMeshAtom(step.atoms.front(), x, why))
      return false;
    int required = requiredOccupant[x];
    switch (step.kind) {
    case PrimitiveKind::AllGather:
      if (occupant[x] == kNone) {
        why = "gathers an atom that holds no digit";
        return false;
      }
      local.insert(occupant[x]);
      occupant[x] = kNone;
      return true;
    case PrimitiveKind::LocalSlice:
      if (occupant[x] != kNone) {
        why = "slices onto an atom that still holds a digit";
        return false;
      }
      return place(x, required, why);
    case PrimitiveKind::AllToAll: {
      if (occupant[x] == kNone) {
        why = "all-to-all on an atom that holds no digit";
        return false;
      }
      int gathered = occupant[x];
      occupant[x] = kNone;
      if (!place(x, required, why))
        return false;
      local.insert(gathered);
      return true;
    }
    case PrimitiveKind::AllReduce:
    case PrimitiveKind::ReduceScatter:
    case PrimitiveKind::Permute:
      why = "kind is not part of a reduction-free chain";
      return false;
    }
    llvm_unreachable("covered switch");
  }

  // Moves the tile digit `digit` onto mesh atom `x`.
  bool place(size_t x, int digit, std::string &why) {
    if (digit == kNone || !local.count(digit)) {
      why = "the digit the output needs on the atom is not local";
      return false;
    }
    if (extent[digit] != collective.meshAtoms[x].extent) {
      why = "digit and atom extents differ";
      return false;
    }
    local.erase(digit);
    occupant[x] = digit;
    return true;
  }

  bool applyPermute(const PrimitiveStep &step, std::string &why) {
    std::set<size_t> inStep;
    for (const StepAtom &atom : step.atoms) {
      size_t x;
      if (!findMeshAtom(atom, x, why))
        return false;
      inStep.insert(x);
    }
    std::vector<int> renamed = occupant;
    for (size_t x : inStep) {
      const MeshAtom &atom = collective.meshAtoms[x];
      if (atom.out.kind != AtomRole::Mesh) {
        why = "permutes an atom whose output is not another atom's digit";
        return false;
      }
      int source = meshDigit(atom.out.partner);
      if (source == kNone || !inStep.count(source)) {
        why = "a digit's source atom is not part of the permute";
        return false;
      }
      renamed[x] = occupant[source];
    }
    occupant = renamed;
    return true;
  }

  bool checkFinal(std::string &why) {
    for (size_t x = 0; x < occupant.size(); ++x)
      if (occupant[x] != requiredOccupant[x]) {
        why = "mesh" + std::to_string(collective.meshAtoms[x].axis) + "." +
              std::to_string(collective.meshAtoms[x].atom) +
              " ends holding the wrong digit";
        return false;
      }
    if (local != requiredLocal) {
      why = "the local tile ends with the wrong digits";
      return false;
    }
    if (localPayload() != expectedFinalPayload) {
      why = "final payload " + std::to_string(localPayload()) +
            " differs from the output tile size " +
            std::to_string(expectedFinalPayload);
      return false;
    }
    return true;
  }

  const NormalizedCollective &collective;
  std::vector<uint64_t> extent;
  std::vector<int> occupant, requiredOccupant;
  std::set<int> local, requiredLocal;
  int64_t elementBytes = 1;
  int64_t expectedFinalPayload = 0;
};

} // namespace

bool verifyChainRealizesCollective(const NormalizedCollective &collective,
                                   const std::vector<PrimitiveStep> &chain,
                                   std::string &why) {
  return ChainChecker(collective).run(chain, why);
}

double CollectivePlan::totalDuration() const {
  double total = 0;
  for (const PrimitiveStep &step : chain)
    total += step.isolatedDuration();
  return total;
}

std::optional<CollectivePlan>
planCollective(DistributedCollectiveOp collective,
               ArrayRef<PhysicalCommAxisType> meshAxisTypes,
               const MeshCostParams &params, std::string &failureReason,
               const PlanOptions &options) {
  std::optional<NormalizedCollective> normalized =
      normalizeCollective(collective, meshAxisTypes, failureReason);
  if (!normalized)
    return std::nullopt;
  std::optional<std::vector<PrimitiveStep>> chain =
      plan(*normalized, params, failureReason, options);
  if (!chain)
    return std::nullopt;
  // normalizeCollective established the single Await user.
  auto await = cast<DistributedAwait>(*collective->getUsers().begin());
  return CollectivePlan{collective, await, collective.getInputObject(),
                        await.getValue(), std::move(*chain)};
}

} // namespace mlir::enzyme::distributed
