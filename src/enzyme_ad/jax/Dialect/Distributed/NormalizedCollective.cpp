#include "NormalizedCollective.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ErrorHandling.h"

#include "src/enzyme_ad/jax/Utils.h"

#include <map>
#include <set>
#include <tuple>

namespace mlir::enzyme::distributed {

namespace {

using LabelKey = std::tuple<int, size_t, size_t>;

LabelKey keyOf(const AtomLabel &label) {
  return {static_cast<int>(label.space), label.axis, label.atom};
}

// The single atom of an atomic factor.
AtomLabel atomOf(const CollectiveAtoms &atoms, const ResolvedFactor &factor) {
  SmallVector<AtomLabel> labels = atoms.labelsOf(factor);
  assert(labels.size() == 1 && "factor of an atomic collective is one atom");
  return labels.front();
}

} // namespace

// A pair (L -> R) sends input digit L to output digit R, so `in` is read from
// pairs with the atom as L and `out` from pairs with it as R. in Tile means
// (atom -> OutTile atom), a gather into the tile; out Tile means (InTile atom
// -> atom), a slice or scatter of the tile.
RolePair classify(const MeshAtom &atom) {
  const MeshAtomRole &in = atom.in, &out = atom.out;
  switch (in.kind) {
  case AtomRole::Reduced:
    switch (out.kind) {
    case AtomRole::Replicate:
      return RolePair::AllReduce;
    case AtomRole::Tile:
      return RolePair::ReduceScatter;
    case AtomRole::Mesh:
      return RolePair::ReduceThenPermute;
    case AtomRole::Reduced:
      llvm_unreachable("an output digit is never reduced");
    }
    llvm_unreachable("covered switch");
  case AtomRole::Tile:
    switch (out.kind) {
    case AtomRole::Replicate:
      return RolePair::AllGather;
    case AtomRole::Tile:
      return RolePair::TileToTile;
    case AtomRole::Mesh:
      return RolePair::Permute;
    case AtomRole::Reduced:
      llvm_unreachable("an output digit is never reduced");
    }
    llvm_unreachable("covered switch");
  case AtomRole::Replicate:
    switch (out.kind) {
    case AtomRole::Tile:
      return RolePair::LocalSlice;
    case AtomRole::Replicate:
      return RolePair::NoOp;
    case AtomRole::Mesh:
      return RolePair::Permute;
    case AtomRole::Reduced:
      llvm_unreachable("an output digit is never reduced");
    }
    llvm_unreachable("covered switch");
  case AtomRole::Mesh: {
    AtomLabel self{AtomSpace::Mesh, atom.axis, atom.atom};
    if (out.kind == AtomRole::Mesh && in.partner == self && out.partner == self)
      return RolePair::NoOp;
    return RolePair::Permute;
  }
  }
  llvm_unreachable("covered switch");
}

const char *toString(AtomRole role) {
  switch (role) {
  case AtomRole::Reduced:
    return "reduced";
  case AtomRole::Tile:
    return "tile";
  case AtomRole::Mesh:
    return "mesh";
  case AtomRole::Replicate:
    return "replicate";
  }
  llvm_unreachable("covered switch");
}

const char *toString(RolePair pair) {
  switch (pair) {
  case RolePair::AllReduce:
    return "all-reduce";
  case RolePair::ReduceScatter:
    return "reduce-scatter";
  case RolePair::ReduceThenPermute:
    return "reduce-then-permute";
  case RolePair::AllGather:
    return "all-gather";
  case RolePair::TileToTile:
    return "tile-to-tile";
  case RolePair::LocalSlice:
    return "local-slice";
  case RolePair::NoOp:
    return "no-op";
  case RolePair::Permute:
    return "permute";
  }
  llvm_unreachable("covered switch");
}

const char *toString(ReductionKind kind) {
  switch (kind) {
  case ReductionKind::None:
    return "none";
  case ReductionKind::Unknown:
    return "unknown";
  case ReductionKind::Add:
    return "add";
  case ReductionKind::Min:
    return "min";
  case ReductionKind::Max:
    return "max";
  case ReductionKind::Mul:
    return "mul";
  case ReductionKind::And:
    return "and";
  case ReductionKind::Or:
    return "or";
  case ReductionKind::Xor:
    return "xor";
  }
  llvm_unreachable("covered switch");
}

NormalizedCollective
buildNormalizedCollective(const CollectiveResolution &resolution,
                          ArrayRef<int64_t> inputTile,
                          ArrayRef<int64_t> outputTile, int64_t elementBytes,
                          ReductionKind reductionKind) {
  assert(isCollectiveAtomic(resolution) &&
         "buildNormalizedCollective requires an atomic collective");
  assert((reductionKind == ReductionKind::None) ==
             resolution.reductionGroups.empty() &&
         "reduction kind must be None exactly when there are no reductions");
  const CollectiveAtoms &atoms = resolution.atoms;

  NormalizedCollective result;
  result.reductionKind = reductionKind;
  result.payloadBytes = elementBytes;
  for (int64_t dim : inputTile)
    result.payloadBytes *= dim;

  // Atoms with extent 1 carry no data: mapping pairs and mesh operands drop
  // them, so they never get a role.
  auto isLive = [&](const AtomLabel &label) {
    return atoms.extentOf(label) != 1;
  };

  // Role of each mesh atom's input digit (from reductions and pair lhs's) and
  // output digit (from pair rhs's), plus the partner of each tile atom.
  std::map<LabelKey, MeshAtomRole> inRoles, outRoles;
  std::map<LabelKey, AtomLabel> tilePartner;

  for (const ResolvedGroup &group : resolution.reductionGroups)
    for (const ResolvedFactor &factor : group) {
      AtomLabel label = atomOf(atoms, factor);
      if (!isLive(label))
        continue;
      // Reductions are over mesh atoms only; device-local axes were inlined
      // away before this point.
      assert(label.space == AtomSpace::Mesh &&
             "reduction over a non-mesh atom");
      bool inserted =
          inRoles.try_emplace(keyOf(label), MeshAtomRole{AtomRole::Reduced})
              .second;
      assert(inserted && "atom reduced twice");
      (void)inserted;
    }

  for (const auto &[lhsGroup, rhsGroup] : resolution.pairs) {
    AtomLabel lhs = atomOf(atoms, lhsGroup.front());
    AtomLabel rhs = atomOf(atoms, rhsGroup.front());
    if (!isLive(lhs))
      continue;

    if (lhs.space == AtomSpace::Mesh) {
      MeshAtomRole role;
      switch (rhs.space) {
      case AtomSpace::OutTile:
        role = {AtomRole::Tile, rhs};
        break;
      case AtomSpace::Mesh:
        role = {AtomRole::Mesh, rhs};
        break;
      case AtomSpace::Replicate:
        role = {AtomRole::Replicate};
        break;
      case AtomSpace::InTile:
        llvm_unreachable("a pair's rhs is never an input-tile atom");
      }
      bool inserted = inRoles.try_emplace(keyOf(lhs), role).second;
      assert(inserted && "mesh atom's input digit goes to two places");
      (void)inserted;
    } else if (lhs.space == AtomSpace::InTile) {
      tilePartner[keyOf(lhs)] = rhs;
    } else {
      assert(lhs.space == AtomSpace::Replicate);
    }

    if (rhs.space == AtomSpace::Mesh) {
      MeshAtomRole role;
      switch (lhs.space) {
      case AtomSpace::InTile:
        role = {AtomRole::Tile, lhs};
        break;
      case AtomSpace::Mesh:
        role = {AtomRole::Mesh, lhs};
        break;
      case AtomSpace::Replicate:
        role = {AtomRole::Replicate};
        break;
      case AtomSpace::OutTile:
        llvm_unreachable("a pair's lhs is never an output-tile atom");
      }
      bool inserted = outRoles.try_emplace(keyOf(rhs), role).second;
      assert(inserted && "mesh atom's output digit comes from two places");
      (void)inserted;
    } else if (rhs.space == AtomSpace::OutTile) {
      tilePartner[keyOf(rhs)] = lhs;
    }
  }

  // Every atom of every mesh axis the collective's mesh operands touch. An
  // atomic collective covers each such atom on both operands.
  std::set<size_t> meshAxes;
  for (const ResolvedFactor &factor : resolution.inputMeshFactors)
    meshAxes.insert(factor.key.second);
  for (const ResolvedFactor &factor : resolution.outputMeshFactors)
    meshAxes.insert(factor.key.second);
  for (size_t axis : meshAxes)
    for (const AtomLabel &label : atoms.labelsOfAxis({AtomSpace::Mesh, axis})) {
      if (!isLive(label))
        continue;
      auto in = inRoles.find(keyOf(label));
      auto out = outRoles.find(keyOf(label));
      assert(in != inRoles.end() && out != outRoles.end() &&
             "mesh atom lacks an in or out role: explicit replication has not "
             "run");
      MeshAtom atom{
          axis,       label.atom, atoms.extentOf(label), atoms.strideOf(label),
          in->second, out->second};
      result.meshAtoms.push_back(atom);
    }

  auto addTileAtoms = [&](AtomSpace space, size_t rank) {
    for (size_t dim = 0; dim < rank; ++dim)
      for (const AtomLabel &label : atoms.labelsOfAxis({space, dim})) {
        if (!isLive(label))
          continue;
        TileAtom tile{space == AtomSpace::InTile, dim, label.atom,
                      atoms.extentOf(label), std::nullopt};
        if (auto it = tilePartner.find(keyOf(label)); it != tilePartner.end())
          tile.partner = it->second;
        result.tileAtoms.push_back(tile);
      }
  };
  addTileAtoms(AtomSpace::InTile, inputTile.size());
  addTileAtoms(AtomSpace::OutTile, outputTile.size());
  return result;
}

std::optional<NormalizedCollective>
normalizeCollective(DistributedCollectiveOp collective,
                    ArrayRef<PhysicalCommAxisType> meshAxisTypes,
                    std::string &failureReason) {
  // The collective's async handle has exactly one Await consumer, whose
  // result type is the output tile.
  assert(llvm::hasSingleElement(collective->getUsers()) &&
         "a DistributedCollective's async handle must have exactly one "
         "DistributedAwait consumer");
  auto await = cast<DistributedAwait>(*collective->getUsers().begin());
  auto inputType =
      cast<RankedTensorType>(collective.getInputObject().getType());
  ArrayRef<int64_t> inputTile = inputType.getShape();
  ArrayRef<int64_t> outputTile =
      cast<RankedTensorType>(await.getValue().getType()).getShape();

  CollectiveResolutionError error;
  FailureOr<CollectiveResolution> resolution = resolveCollectiveAtoms(
      collective, meshAxisTypes, inputTile, outputTile, error);
  if (failed(resolution)) {
    failureReason =
        "could not resolve (" + llvm::join(error.reasons, "; ") + ")";
    return std::nullopt;
  }
  if (!isCollectiveAtomic(*resolution)) {
    failureReason = "collective is not atomic";
    return std::nullopt;
  }

  Type elementType = inputType.getElementType();
  int64_t elementBytes = (elementType.getIntOrFloatBitWidth() + 7) / 8;

  ReductionKind kind = ReductionKind::None;
  for (Region &body : collective.getReductionBodies()) {
    ReductionKind bodyKind;
    switch (stablehlo::classifyReduceBlockKind(body.front())) {
    case stablehlo::ReduceOpKind::Add:
      bodyKind = ReductionKind::Add;
      break;
    case stablehlo::ReduceOpKind::Min:
      bodyKind = ReductionKind::Min;
      break;
    case stablehlo::ReduceOpKind::Max:
      bodyKind = ReductionKind::Max;
      break;
    case stablehlo::ReduceOpKind::Mul:
      bodyKind = ReductionKind::Mul;
      break;
    case stablehlo::ReduceOpKind::And:
      bodyKind = ReductionKind::And;
      break;
    case stablehlo::ReduceOpKind::Or:
      bodyKind = ReductionKind::Or;
      break;
    case stablehlo::ReduceOpKind::Xor:
      bodyKind = ReductionKind::Xor;
      break;
    case stablehlo::ReduceOpKind::Unknown:
      bodyKind = ReductionKind::Unknown;
      break;
    }
    assert((kind == ReductionKind::None || kind == bodyKind) &&
           "multi-kind reductions unsupported, generalize if needed");
    kind = bodyKind;
  }
  return buildNormalizedCollective(*resolution, inputTile, outputTile,
                                   elementBytes, kind);
}

} // namespace mlir::enzyme::distributed
