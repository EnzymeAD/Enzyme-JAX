#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

#include "src/enzyme_ad/jax/Dialect/Distributed/NormalizedCollective.h"
#include "src/enzyme_ad/jax/Dialect/Distributed/Utilities.h"

namespace mlir::enzyme::distributed {

#define GEN_PASS_DEF_PRINTCOLLECTIVEPLANPASS
#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h.inc"

namespace {

// Same `<space><axis>.<atom>` handle as distributed-atomize-collectives's
// remarks, so the two passes' output can be read side by side.
void printLabel(llvm::raw_ostream &os, const AtomLabel &label) {
  switch (label.space) {
  case AtomSpace::Mesh:
    os << "mesh";
    break;
  case AtomSpace::InTile:
    os << "in";
    break;
  case AtomSpace::OutTile:
    os << "out";
    break;
  case AtomSpace::Replicate:
    os << "replicate";
    break;
  case AtomSpace::MidTile:
    os << "mid";
    break;
  }
  os << label.axis << "." << label.atom;
}

void printRole(llvm::raw_ostream &os, const MeshAtomRole &role) {
  os << toString(role.kind);
  if (role.kind == AtomRole::Tile || role.kind == AtomRole::Mesh) {
    os << "(";
    printLabel(os, role.partner);
    os << ")";
  }
}

// One line per mesh atom, tile atom and reduction group, then the payload.
std::string describe(const NormalizedCollective &normalized) {
  std::string message;
  llvm::raw_string_ostream os(message);
  for (const MeshAtom &atom : normalized.meshAtoms) {
    os << "mesh" << atom.axis << "." << atom.atom << ": extent " << atom.extent
       << " stride " << atom.stride << " in ";
    printRole(os, atom.in);
    os << " out ";
    printRole(os, atom.out);
    os << " => " << toString(classify(atom)) << "\n";
  }
  for (const TileAtom &tile : normalized.tileAtoms) {
    os << (tile.isInput ? "in" : "out") << tile.dim << "." << tile.atom
       << ": extent " << tile.extent;
    if (tile.partner) {
      os << " pairs ";
      printLabel(os, *tile.partner);
    }
    os << "\n";
  }
  if (normalized.reductionKind != ReductionKind::None)
    os << "reduction: " << toString(normalized.reductionKind) << "\n";
  os << "payload bytes: " << normalized.payloadBytes << "\n";
  return message;
}

struct PrintCollectivePlanPass
    : public impl::PrintCollectivePlanPassBase<PrintCollectivePlanPass> {
  using PrintCollectivePlanPassBase::PrintCollectivePlanPassBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();

    FailureOr<PhysicalMeshOp> physicalMesh = findUniquePhysicalMesh(module);
    if (failed(physicalMesh)) {
      // findUniquePhysicalMesh already emitted a diagnostic.
      signalPassFailure();
      return;
    }
    SmallVector<PhysicalCommAxisType> meshAxisTypes;
    for (Attribute axisAttr : physicalMesh->getAxesAttr())
      meshAxisTypes.push_back(
          cast<PhysicalCommAxisType>(cast<TypeAttr>(axisAttr).getValue()));

    module.walk([&](DistributedCollectiveOp collective) {
      std::string failureReason;
      std::optional<NormalizedCollective> normalized =
          normalizeCollective(collective, meshAxisTypes, failureReason);
      if (!normalized) {
        collective->emitRemark()
            << "print-collective-plan: no plan (" << failureReason << ")";
        return;
      }
      collective->emitRemark() << describe(*normalized);
    });

    markAllAnalysesPreserved();
  }
};

} // namespace

} // namespace mlir::enzyme::distributed
