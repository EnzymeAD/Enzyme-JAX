#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

#include "src/enzyme_ad/jax/Dialect/Distributed/CollectiveAtoms.h"
#include "src/enzyme_ad/jax/Dialect/Distributed/Utilities.h"

namespace mlir::enzyme::distributed {

#define GEN_PASS_DEF_ATOMIZECOLLECTIVESPASS
#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h.inc"

namespace {

// Prints one atom as `<space><axis>.<atom>` (e.g. `mesh0.1`, `in2.0`): a
// handle for a lit test (or a future NormalizedCollective builder) to key
// off of that only depends on the resolution's own structure, never on an
// SSA value name.
void printAtomLabel(llvm::raw_ostream &os, const AtomLabel &label) {
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
  }
  os << label.axis << "." << label.atom;
}

void printAtomLabels(llvm::raw_ostream &os, ArrayRef<AtomLabel> labels) {
  llvm::interleaveComma(
      labels, os, [&](const AtomLabel &label) { printAtomLabel(os, label); });
}

// Renders a resolved collective as one line per mesh axis and per tile
// dimension (its atoms' extents, major-first) followed by one line per
// reduction group and per mapping pair (the atoms it covers, as `pair k: lhs
// -> rhs`). This is deliberately the same information a NormalizedCollective
// builder (a later pass) will need to rebuild the atomic form, so pinning
// this text in a lit test doubles as a check on the resolution itself, not
// just on this pass's own formatting.
std::string describeResolution(ArrayRef<PhysicalCommAxisType> meshAxisTypes,
                               ArrayRef<int64_t> inputTile,
                               ArrayRef<int64_t> outputTile,
                               const CollectiveResolution &resolution) {
  std::string message;
  llvm::raw_string_ostream os(message);
  const CollectiveAtoms &atoms = resolution.atoms;

  auto printAxisAtoms = [&](StringRef prefix, AtomSpace space, size_t axis) {
    os << prefix << axis << ": ";
    llvm::interleaveComma(
        atoms.labelsOfAxis({space, axis}), os,
        [&](const AtomLabel &label) { os << atoms.extentOf(label); });
    os << "\n";
  };
  for (size_t a = 0; a < meshAxisTypes.size(); ++a)
    printAxisAtoms("mesh", AtomSpace::Mesh, a);
  for (size_t d = 0; d < inputTile.size(); ++d)
    printAxisAtoms("in", AtomSpace::InTile, d);
  for (size_t d = 0; d < outputTile.size(); ++d)
    printAxisAtoms("out", AtomSpace::OutTile, d);

  for (auto [i, group] : llvm::enumerate(resolution.reductionGroups)) {
    os << "reduction" << i << ": ";
    printAtomLabels(os, atoms.labelsOf(group));
    os << "\n";
  }
  for (auto [i, pair] : llvm::enumerate(resolution.pairs)) {
    os << "pair" << i << ": ";
    printAtomLabels(os, atoms.labelsOf(pair.first));
    os << " -> ";
    printAtomLabels(os, atoms.labelsOf(pair.second));
    os << "\n";
  }
  return message;
}

struct AtomizeCollectivesPass
    : public impl::AtomizeCollectivesPassBase<AtomizeCollectivesPass> {
  using AtomizeCollectivesPassBase::AtomizeCollectivesPassBase;

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
      // A DistributedCollective's async handle always has exactly one
      // DistributedAwait consumer (see createCollectiveAndAwait); that
      // Await's result is the output tile shape (no mesh prefix, same as
      // input_object), and is the dialect-idiomatic way to reach it rather
      // than re-deriving it from output_type.
      assert(llvm::hasSingleElement(collective->getUsers()) &&
             "a DistributedCollective's async handle must have exactly one "
             "DistributedAwait consumer (see createCollectiveAndAwait)");
      auto await = cast<DistributedAwait>(*collective->getUsers().begin());

      auto inputTile =
          cast<RankedTensorType>(collective.getInputObject().getType())
              .getShape();
      auto outputTile =
          cast<RankedTensorType>(await.getValue().getType()).getShape();

      CollectiveResolutionError error;
      FailureOr<CollectiveResolution> resolution = resolveCollectiveAtoms(
          collective, meshAxisTypes, inputTile, outputTile, error);
      if (failed(resolution)) {
        if (error.kind != CollectiveResolutionError::Kind::NoCommonAtoms) {
          // Structural errors (a factor group not built from axis.product, a
          // factor with no provenance axis, or a physical axis outside the
          // module's mesh) are exactly what the op verifier and this pass's
          // own precondition -- running after
          // distributed-make-replications-explicit, against the same single
          // physical mesh -- already rule out for IR that reaches this pass.
          llvm::report_fatal_error(
              llvm::Twine("distributed-atomize-collectives: "
                          "resolveCollectiveAtoms hit a structural error on "
                          "verifier-legal IR: ") +
              llvm::join(error.reasons, "; "));
        }
        // An indivisible mapping pair or non-nesting cuts: a legitimate
        // infeasible sharding candidate, not a bug (see this pass's own
        // description in Passes.td).
        collective->emitRemark()
            << "atomize-collectives: could not split into common atoms ("
            << error.reasons.front() << ")";
        return;
      }
      collective->emitRemark() << describeResolution(meshAxisTypes, inputTile,
                                                     outputTile, *resolution);
    });

    markAllAnalysesPreserved();
  }
};

} // namespace

} // namespace mlir::enzyme::distributed
