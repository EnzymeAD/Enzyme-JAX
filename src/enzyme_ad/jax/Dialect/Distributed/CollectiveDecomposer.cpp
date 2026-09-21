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
// permute group, which owns every atom of the pure permutation cycles.
enum class UnitKind {
  Slice,
  Gather,
  TileToTile,
  AllReduce,
  ReduceScatter,
  Permute
};

struct Unit {
  UnitKind kind;
  std::vector<StepAtom> atoms;
  // Extent of the (single) atom; unused by Permute.
  uint64_t extent;
  // Units that must have communicated (stage >= 1) before this unit's
  // dependent step may start (D12). For TileToTile and ReduceScatter units the
  // dependent steps are the direct exchange and the pending slice; their first
  // half (gather, all-reduce) never waits. Indices are into the unit's UnitSet
  // when built and into the assembled unit vector once a combination of
  // variants has been assembled.
  std::vector<size_t> after;

  // The stage at which the unit has nothing left to do. Slice, Gather,
  // AllReduce and Permute have stages {pending, done}. TileToTile and
  // ReduceScatter add the intermediate {communicated, slice pending}: the
  // gather (resp. all-reduce) of the alternative that ends in a free slice,
  // which is then the only way forward from that stage.
  uint8_t finalStage() const {
    return kind == UnitKind::TileToTile || kind == UnitKind::ReduceScatter ? 2
                                                                           : 1;
  }
};

// One way of realizing a component, with its internal dependencies.
using UnitSet = std::vector<Unit>;

// A realization of a component: its units, and the mesh atoms it borrows for
// a conjugation (D16). Two chosen variants must not borrow the same atom.
struct Variant {
  UnitSet units;
  std::vector<size_t> borrowed;
};

// A group of mesh atoms decomposed together (D11, D13). Components share
// nothing but the payload and the borrowed atoms, so their units are
// independent DP dimensions. `variants` lists alternative realizations; exactly
// one is used per plan, and the first is always the half-split.
struct Component {
  std::vector<Variant> variants;
};

StepAtom stepAtomOf(const MeshAtom &atom) {
  return {atom.axis, atom.atom, atom.extent};
}

// Index of the mesh atom `label` names.
size_t meshIndexOf(const NormalizedCollective &collective,
                   const AtomLabel &label) {
  assert(label.space == AtomSpace::Mesh && "a mesh partner names a mesh atom");
  for (size_t i = 0; i < collective.meshAtoms.size(); ++i)
    if (collective.meshAtoms[i].axis == label.axis &&
        collective.meshAtoms[i].atom == label.atom)
      return i;
  llvm_unreachable("mesh partner is not an atom of the collective");
}

// True for the rows whose cost depends on a mesh partner's role, which the
// role pair alone does not determine. classify() labels them Permute (or
// ReduceThenPermute for a reduced atom).
bool isMeshCoupled(const MeshAtom &atom) {
  RolePair pair = classify(atom);
  return pair == RolePair::Permute || pair == RolePair::ReduceThenPermute;
}

// D9: the log-round reduction algorithms need an associative and commutative
// body of a recognized kind.
bool checkReductionKind(const NormalizedCollective &collective,
                        const MeshAtom &atom, std::string &failureReason) {
  assert(collective.reductionKind != ReductionKind::None &&
         "a reduced atom implies a reduction body");
  if (collective.reductionKind != ReductionKind::Unknown)
    return true;
  failureReason = "unsupported reduction body on mesh" +
                  std::to_string(atom.axis) + "." + std::to_string(atom.atom) +
                  " (not a single recognized associative operation)";
  return false;
}

// The half-split of a mesh-coupled component (D11): every atom (in = P,
// out = Q) becomes a first half by P followed by a second half by Q, with the
// atom holding replicated data in between. The first halves are ordinary
// units; each second half is a free slice that waits for the digit it places
// (D12).
//
// With `fuse`, an atom whose first half releases a digit (P is Tile or Mesh)
// and whose second half places one (Q is Tile or Mesh) is one TileToTile unit,
// and a reduced atom placing a digit (Q is Mesh) is one ReduceScatter unit
// (D14). The digit they place must be local, so they wait for its source's
// unit exactly as the slice would.
bool halfSplit(const NormalizedCollective &collective,
               const std::vector<size_t> &members, bool fuse, UnitSet &set,
               std::string &failureReason) {
  // Atom index -> the unit of its first half (or fused exchange).
  std::map<size_t, size_t> mainUnit;
  for (size_t i : members) {
    const MeshAtom &atom = collective.meshAtoms[i];
    bool places = atom.out.kind != AtomRole::Replicate;
    UnitKind kind;
    switch (atom.in.kind) {
    case AtomRole::Reduced:
      if (!checkReductionKind(collective, atom, failureReason))
        return false;
      kind = fuse && places ? UnitKind::ReduceScatter : UnitKind::AllReduce;
      break;
    case AtomRole::Tile:
    case AtomRole::Mesh:
      kind = fuse && places ? UnitKind::TileToTile : UnitKind::Gather;
      break;
    case AtomRole::Replicate:
      continue;
    }
    mainUnit[i] = set.size();
    set.push_back({kind, {stepAtomOf(atom)}, atom.extent, {}});
  }
  for (size_t i : members) {
    const MeshAtom &atom = collective.meshAtoms[i];
    if (atom.out.kind == AtomRole::Replicate)
      continue;
    // A digit that came from another atom must be local by the time it is
    // placed. An input tile digit is local from the start.
    std::vector<size_t> sourceLocal;
    if (atom.out.kind == AtomRole::Mesh) {
      auto it = mainUnit.find(meshIndexOf(collective, atom.out.partner));
      assert(it != mainUnit.end() &&
             "a digit that moves between atoms is gathered by its source");
      sourceLocal.push_back(it->second);
    }
    auto own = mainUnit.find(i);
    if (own != mainUnit.end()) {
      Unit &main = set[own->second];
      if (main.kind == UnitKind::TileToTile ||
          main.kind == UnitKind::ReduceScatter) {
        main.after = std::move(sourceLocal);
        continue;
      }
      // The atom must have released its own digit (or finished reducing).
      sourceLocal.push_back(own->second);
    }
    set.push_back({UnitKind::Slice,
                   {stepAtomOf(atom)},
                   atom.extent,
                   std::move(sourceLocal)});
  }
  return true;
}

// The path closure of a non-pure mesh-coupled component (D15). Data flows
// along the component's mesh links from a start atom s (whose output is a
// tile digit or a replicate) through atoms that hand their digit on, to an end
// atom e (whose input is reduced, a tile digit or a replicate). Once e holds
// no digit, the whole path is one cyclic shift of atom contents: every atom
// takes its source's digit and s takes e's empty content. That is one permute,
// followed by the free slice of s's tile digit when s's output is a tile atom.
//
// e is emptied first: an all-reduce when it is reduced, a gather when its own
// digit goes to the output tile, nothing when its input is replicated.
void pathClosure(const NormalizedCollective &collective,
                 const std::vector<size_t> &members, UnitSet &set) {
  size_t start = members.front(), end = members.front();
  for (size_t i : members) {
    const MeshAtom &atom = collective.meshAtoms[i];
    if (atom.out.kind != AtomRole::Mesh)
      start = i;
    if (atom.in.kind != AtomRole::Mesh)
      end = i;
  }
  const MeshAtom &s = collective.meshAtoms[start];
  const MeshAtom &e = collective.meshAtoms[end];
  assert(s.out.kind != AtomRole::Mesh && e.in.kind != AtomRole::Mesh &&
         "a non-pure component is a path with a start and an end");

  std::vector<size_t> dependency;
  switch (e.in.kind) {
  case AtomRole::Reduced:
    set.push_back({UnitKind::AllReduce, {stepAtomOf(e)}, e.extent, {}});
    dependency.push_back(set.size() - 1);
    break;
  case AtomRole::Tile:
    set.push_back({UnitKind::Gather, {stepAtomOf(e)}, e.extent, {}});
    dependency.push_back(set.size() - 1);
    break;
  case AtomRole::Replicate:
    break;
  case AtomRole::Mesh:
    llvm_unreachable("the end of a path hands no digit on");
  }
  std::vector<StepAtom> atoms;
  for (size_t i : members)
    atoms.push_back(stepAtomOf(collective.meshAtoms[i]));
  set.push_back({UnitKind::Permute, std::move(atoms), 0, dependency});
  if (s.out.kind == AtomRole::Tile)
    set.push_back(
        {UnitKind::Slice, {stepAtomOf(s)}, s.extent, {set.size() - 1}});
}

// The mesh atom a step on `atom` can be conjugated onto (D16): an atom of the
// same extent that holds no digit and has no role of its own, on an axis with
// strictly more bandwidth. Returns the fastest such atom (ties to the lowest
// index), or meshAtoms.size() when there is none.
size_t conjugationTarget(const NormalizedCollective &collective,
                         const MeshCostParams &params, const StepAtom &atom) {
  size_t best = collective.meshAtoms.size();
  for (size_t i = 0; i < collective.meshAtoms.size(); ++i) {
    const MeshAtom &candidate = collective.meshAtoms[i];
    if (candidate.in.kind != AtomRole::Replicate ||
        candidate.out.kind != AtomRole::Replicate ||
        candidate.extent != atom.extent ||
        params.bandwidth[candidate.axis] <= params.bandwidth[atom.axis])
      continue;
    if (best == collective.meshAtoms.size() ||
        params.bandwidth[candidate.axis] >
            params.bandwidth[collective.meshAtoms[best].axis])
      best = i;
  }
  return best;
}

// `base` with unit `u` (a gather or all-reduce on atom A) run on atom
// `target` instead: a swap permute of A and `target` first, then the step on
// `target`. The target holds no digit, so after the step both atoms are empty
// again, exactly as after the unconjugated step, and no swap back is needed.
Variant conjugate(const Variant &base, size_t u, const MeshAtom &target,
                  size_t targetIndex) {
  // Unit u becomes two units (swap, then the step), shifting later indices.
  auto newIndex = [&](size_t old) { return old < u ? old : old + 1; };
  Variant result;
  result.borrowed = {targetIndex};
  for (size_t i = 0; i < base.units.size(); ++i) {
    Unit unit = base.units[i];
    for (size_t &dep : unit.after)
      dep = newIndex(dep);
    if (i == u) {
      result.units.push_back(
          {UnitKind::Permute, {unit.atoms.front(), stepAtomOf(target)}, 0, {}});
      unit.atoms = {stepAtomOf(target)};
      unit.after.push_back(u);
    }
    result.units.push_back(std::move(unit));
  }
  return result;
}

// Appends to `component` one variant per gather or all-reduce unit of its
// first variant that has a conjugation target (D16). At most one unit is
// conjugated per variant.
void addConjugations(const NormalizedCollective &collective,
                     const MeshCostParams &params, Component &component) {
  const Variant base = component.variants.front();
  for (size_t u = 0; u < base.units.size(); ++u) {
    const Unit &unit = base.units[u];
    if (unit.kind != UnitKind::Gather && unit.kind != UnitKind::AllReduce)
      continue;
    size_t target = conjugationTarget(collective, params, unit.atoms.front());
    if (target == collective.meshAtoms.size())
      continue;
    component.variants.push_back(
        conjugate(base, u, collective.meshAtoms[target], target));
  }
}

// Splits the mesh atoms into components of units, or reports why the
// collective is outside the supported rows.
//
// Atoms with no mesh partner have a dedicated row of their own. Atoms tied by
// mesh partners form connected components (D11): one made only of (Mesh, Mesh)
// atoms is a set of permutation cycles, and all cycles share one permute unit
// (D5); any other component is half-split (D11) and, unless
// `options.relayVariants` is off, also offered as a path closure (D15).
// Components are ordered by their first atom, with the permute component last.
// The variant lists are trimmed so that their product stays within
// kMaxVariantCombinations, keeping the earliest variants of the earliest
// components.
bool buildComponents(const NormalizedCollective &collective,
                     const MeshCostParams &params, const PlanOptions &options,
                     std::vector<Component> &components,
                     std::string &failureReason) {
  auto single = [&](UnitKind kind, const MeshAtom &atom) {
    Component component;
    component.variants.push_back(
        {UnitSet{{kind, {stepAtomOf(atom)}, atom.extent, {}}}, {}});
    if (options.relayVariants)
      addConjugations(collective, params, component);
    components.push_back(std::move(component));
  };

  // Union-find over the mesh-coupled atoms, joined by mesh partners.
  size_t numAtoms = collective.meshAtoms.size();
  std::vector<size_t> parent(numAtoms);
  for (size_t i = 0; i < numAtoms; ++i)
    parent[i] = i;
  auto find = [&](size_t x) {
    while (parent[x] != x)
      x = parent[x] = parent[parent[x]];
    return x;
  };
  for (size_t i = 0; i < numAtoms; ++i) {
    const MeshAtom &atom = collective.meshAtoms[i];
    if (!isMeshCoupled(atom))
      continue;
    for (const MeshAtomRole *role : {&atom.in, &atom.out})
      if (role->kind == AtomRole::Mesh)
        parent[find(i)] = find(meshIndexOf(collective, role->partner));
  }
  std::map<size_t, std::vector<size_t>> members;
  for (size_t i = 0; i < numAtoms; ++i)
    if (isMeshCoupled(collective.meshAtoms[i]))
      members[find(i)].push_back(i);

  std::vector<StepAtom> permuted;
  std::set<size_t> emitted;
  for (size_t i = 0; i < numAtoms; ++i) {
    const MeshAtom &atom = collective.meshAtoms[i];
    switch (classify(atom)) {
    case RolePair::NoOp:
      break;
    case RolePair::LocalSlice:
      single(UnitKind::Slice, atom);
      break;
    case RolePair::AllGather:
      single(UnitKind::Gather, atom);
      break;
    case RolePair::TileToTile:
      single(UnitKind::TileToTile, atom);
      break;
    case RolePair::AllReduce:
      if (!checkReductionKind(collective, atom, failureReason))
        return false;
      single(UnitKind::AllReduce, atom);
      break;
    case RolePair::ReduceScatter:
      if (!checkReductionKind(collective, atom, failureReason))
        return false;
      single(UnitKind::ReduceScatter, atom);
      break;
    case RolePair::ReduceThenPermute:
    case RolePair::Permute: {
      const std::vector<size_t> &group = members.at(find(i));
      bool pure = llvm::all_of(group, [&](size_t j) {
        const MeshAtom &member = collective.meshAtoms[j];
        return member.in.kind == AtomRole::Mesh &&
               member.out.kind == AtomRole::Mesh;
      });
      if (pure) {
        permuted.push_back(stepAtomOf(atom));
        break;
      }
      if (!emitted.insert(find(i)).second)
        break;
      Component component;
      component.variants.push_back({});
      if (!halfSplit(collective, group, options.relayVariants,
                     component.variants.front().units, failureReason))
        return false;
      if (options.relayVariants) {
        Variant closure;
        pathClosure(collective, group, closure.units);
        component.variants.push_back(std::move(closure));
        addConjugations(collective, params, component);
      }
      components.push_back(std::move(component));
      break;
    }
    }
  }
  if (!permuted.empty()) {
    UnitSet set;
    set.push_back({UnitKind::Permute, std::move(permuted), 0, {}});
    components.push_back({{{std::move(set), {}}}});
  }

  // Keep the product of the variant counts within the cap. The first variant
  // of every component (the half-split) always stays, so no collective becomes
  // unsupported.
  size_t product = 1;
  for (Component &component : components) {
    size_t allowed = std::max<size_t>(1, kMaxVariantCombinations / product);
    if (component.variants.size() > allowed)
      component.variants.resize(allowed);
    product *= component.variants.size();
  }

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

// Concatenates the chosen variant of every component into one unit vector,
// shifting each unit's dependencies to the assembled indices.
std::vector<Unit> assemble(const std::vector<Component> &components,
                           const std::vector<size_t> &choice) {
  std::vector<Unit> units;
  for (size_t c = 0; c < components.size(); ++c) {
    size_t offset = units.size();
    for (Unit unit : components[c].variants[choice[c]].units) {
      for (size_t &dep : unit.after)
        dep += offset;
      units.push_back(std::move(unit));
    }
  }
  return units;
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
      case UnitKind::ReduceScatter:
        if (stage == 2)
          divisor *= e;
        break;
      case UnitKind::AllReduce:
      case UnitKind::Permute:
        break;
      }
    }
    // Every slice and reduce-scatter divides out a digit that is local by
    // then: an input tile atom the initial payload contains, or a digit a
    // gather brought in (D12 orders the gather first).
    assert(payloadBytes * factor % divisor == 0 &&
           "slices exceed the tile payload");
    return payloadBytes * factor / divisor;
  }

  bool finished(const DecomposerState &state) const {
    for (size_t i = 0; i < units.size(); ++i)
      if (state.stage[i] != units[i].finalStage())
        return false;
    return true;
  }

  // True when every unit `unit` waits for has communicated (D12).
  bool dependenciesDone(const DecomposerState &state, const Unit &unit) const {
    return llvm::all_of(unit.after,
                        [&](size_t dep) { return state.stage[dep] >= 1; });
  }

  // The steps available in `state`. A pending free step (D2) is returned
  // alone; otherwise there is one candidate per way to advance each unit.
  // Empty exactly when every unit is at its final stage.
  std::vector<CandidateStep> candidates(const DecomposerState &state) const {
    int64_t payload = payloadOf(state);
    std::vector<CandidateStep> result;
    for (size_t i = 0; i < units.size(); ++i) {
      const Unit &unit = units[i];
      bool sliceReady = (unit.kind == UnitKind::Slice && state.stage[i] == 0 &&
                         dependenciesDone(state, unit)) ||
                        ((unit.kind == UnitKind::TileToTile ||
                          unit.kind == UnitKind::ReduceScatter) &&
                         state.stage[i] == 1 && dependenciesDone(state, unit));
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
      // The exchange of a TileToTile or ReduceScatter unit waits for its
      // dependencies; its gather or all-reduce does not.
      bool ready = dependenciesDone(state, unit);
      if (!ready && unit.kind != UnitKind::TileToTile &&
          unit.kind != UnitKind::ReduceScatter)
        continue;
      switch (unit.kind) {
      case UnitKind::Slice:
        llvm_unreachable("ready slices are consumed as free steps above");
      case UnitKind::Gather:
        result.push_back({allGatherFootprint(unit.atoms, payload, params),
                          advance(state, i, 1)});
        break;
      case UnitKind::TileToTile:
        if (ready)
          result.push_back({allToAllFootprint(unit.atoms, payload, params),
                            advance(state, i, 2)});
        result.push_back({allGatherFootprint(unit.atoms, payload, params),
                          advance(state, i, 1)});
        break;
      case UnitKind::AllReduce:
        result.push_back({allReduceFootprint(unit.atoms, payload, params),
                          advance(state, i, 1)});
        break;
      case UnitKind::ReduceScatter:
        if (ready)
          result.push_back({reduceScatterFootprint(unit.atoms, payload, params),
                            advance(state, i, 2)});
        result.push_back({allReduceFootprint(unit.atoms, payload, params),
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
    assert((!options.empty() || finished(state)) &&
           "unfinished units wait on dependencies that can never complete");
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
  std::vector<Component> components;
  if (!buildComponents(collective, params, options, components, failureReason))
    return std::nullopt;

  // Components are independent but share the payload, so their variants are
  // chosen jointly: one exact search per combination, cheapest chain wins
  // (ties keep the earlier combination). buildComponents keeps the number of
  // combinations within kMaxVariantCombinations.
  size_t combinations = 1;
  for (const Component &component : components)
    combinations *= component.variants.size();
  assert(combinations <= kMaxVariantCombinations &&
         "component variant lists exceed the combination cap");

  std::optional<std::vector<PrimitiveStep>> best;
  std::string oversizeReason;
  double bestCost = std::numeric_limits<double>::infinity();
  for (size_t combination = 0; combination < combinations; ++combination) {
    std::vector<size_t> choice;
    size_t rest = combination;
    for (const Component &component : components) {
      choice.push_back(rest % component.variants.size());
      rest /= component.variants.size();
    }
    // Two components cannot conjugate onto the same atom (D16).
    std::set<size_t> borrowed;
    bool conflict = false;
    for (size_t c = 0; c < components.size(); ++c)
      for (size_t atom : components[c].variants[choice[c]].borrowed)
        conflict |= !borrowed.insert(atom).second;
    if (conflict)
      continue;
    std::vector<Unit> units = assemble(components, choice);
    if (units.size() > kMaxLiveUnits) {
      oversizeReason = "too many units of work (" +
                       std::to_string(units.size()) + ") for the exact search";
      continue;
    }
    ChainSearch search(units, collective.payloadBytes, params, options.filter);
    double cost = search.solve(search.initialState());
    if (cost < bestCost) {
      bestCost = cost;
      best = search.extractChain();
    }
  }
  if (!best) {
    // Every combination was skipped: the first variants never conflict, so the
    // only cause is the unit limit.
    failureReason = oversizeReason;
    return std::nullopt;
  }
  failureReason.clear();
#ifndef NDEBUG
  std::string why;
  assert(verifyChainRealizesCollective(collective, *best, why) &&
         "decomposition does not realize the collective");
#endif
  return best;
}

namespace {

// The symbolic model behind verifyChainRealizesCollective. Digits are indexed
// mesh atoms first, then input tile atoms. A reduced input atom carries no
// digit (its digit is summed away); it is tracked by a pending flag that a
// reduction step clears.
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

  size_t meshIndex(const AtomLabel &label) const {
    int digit = meshDigit(label);
    assert(digit != kNone && "a mesh partner names a mesh atom");
    return static_cast<size_t>(digit);
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
    reductionPending.assign(numMesh, false);

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
        if (collective.reductionKind == ReductionKind::None ||
            collective.reductionKind == ReductionKind::Unknown) {
          why = "reduction body is not a recognized associative kind";
          return false;
        }
        reductionPending[i] = true;
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
      if (reductionPending[x]) {
        why = "slices onto an atom whose reduction has not run";
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
      // The atom then holds replicated data; an output that needs a tile
      // digit on it must be followed by a slice.
      return consumeReduction(x, why);
    case PrimitiveKind::ReduceScatter:
      return consumeReduction(x, why) && place(x, required, why);
    case PrimitiveKind::Permute:
      llvm_unreachable("permutes are handled above");
    }
    llvm_unreachable("covered switch");
  }

  // Sums the reduced atom `x`'s digit away. Each reduced atom is consumed by
  // exactly one step.
  bool consumeReduction(size_t x, std::string &why) {
    if (!reductionPending[x]) {
      why = "reduces an atom that is not reduced, or reduces it twice";
      return false;
    }
    reductionPending[x] = false;
    return true;
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

  // A permute moves the contents of atoms (their digit, or nothing, and their
  // pending reduction) among the atoms it names, so every atom of the step
  // must have equal extent to its source. Two readings:
  //   - The atoms are a union of complete mesh-linked components: every mesh
  //     partner of an atom is in the step. The step is the collective's own
  //     movement. A cycle shifts each atom's content to the atom that takes its
  //     digit. A path (see the path closure) shifts along the path and its
  //     start atom takes the content of its end atom, which is what closes the
  //     path into a cycle.
  //   - The atoms are exactly two atoms, at least one without a mesh link. The
  //     step swaps their contents; it is how a step is run on a different atom
  //     (conjugation), so the digit or pending reduction that a later step
  //     finds on an atom is whatever the swap put there.
  bool applyPermute(const PrimitiveStep &step, std::string &why) {
    std::set<size_t> inStep;
    for (const StepAtom &atom : step.atoms) {
      size_t x;
      if (!findMeshAtom(atom, x, why))
        return false;
      inStep.insert(x);
    }
    bool closed = true, linked = true;
    for (size_t x : inStep) {
      const MeshAtom &atom = collective.meshAtoms[x];
      bool hasLink = false;
      for (const MeshAtomRole *role : {&atom.in, &atom.out})
        if (role->kind == AtomRole::Mesh) {
          hasLink = true;
          closed &= inStep.count(meshIndex(role->partner)) > 0;
        }
      linked &= hasLink;
    }
    if (closed && linked)
      return applyComponentPermute(inStep, why);
    if (inStep.size() == 2)
      return applySwap(inStep, why);
    why = "permute atoms are neither complete components nor a swap";
    return false;
  }

  bool applyComponentPermute(const std::set<size_t> &inStep, std::string &why) {
    std::vector<int> renamed = occupant;
    for (size_t x : inStep) {
      const MeshAtom &atom = collective.meshAtoms[x];
      if (reductionPending[x]) {
        why = "permutes an atom whose reduction has not run";
        return false;
      }
      size_t source;
      if (atom.out.kind == AtomRole::Mesh) {
        source = meshIndex(atom.out.partner);
      } else {
        // The start of a path takes the content of the path's end.
        source = x;
        while (collective.meshAtoms[source].in.kind == AtomRole::Mesh)
          source = meshIndex(collective.meshAtoms[source].in.partner);
      }
      renamed[x] = occupant[source];
    }
    occupant = renamed;
    return true;
  }

  bool applySwap(const std::set<size_t> &inStep, std::string &why) {
    size_t x = *inStep.begin(), y = *std::next(inStep.begin());
    if (collective.meshAtoms[x].extent != collective.meshAtoms[y].extent) {
      why = "swaps atoms of different extents";
      return false;
    }
    std::swap(occupant[x], occupant[y]);
    bool pending = reductionPending[x];
    reductionPending[x] = reductionPending[y];
    reductionPending[y] = pending;
    return true;
  }

  bool checkFinal(std::string &why) {
    for (size_t x = 0; x < reductionPending.size(); ++x)
      if (reductionPending[x]) {
        why = "mesh" + std::to_string(collective.meshAtoms[x].axis) + "." +
              std::to_string(collective.meshAtoms[x].atom) +
              " is never reduced";
        return false;
      }
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
  // Reduced input atoms whose digit has not been summed away yet.
  std::vector<bool> reductionPending;
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
