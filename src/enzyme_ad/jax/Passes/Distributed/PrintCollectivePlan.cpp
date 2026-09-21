#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"

#include "src/enzyme_ad/jax/Dialect/Distributed/CollectiveDecomposer.h"
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

// Reports the decomposer's chain for `plan`, the outcome of the independent
// semantic check, and the ports that attach the chain to the program.
std::string describeChain(const CollectivePlan &plan,
                          const NormalizedCollective &normalized) {
  std::string message;
  llvm::raw_string_ostream os(message);
  os << "chain: " << plan.chain.size() << " steps, total duration "
     << llvm::format("%g", plan.totalDuration()) << "\n";
  for (size_t i = 0; i < plan.chain.size(); ++i)
    os << "step " << i << ": " << describeStep(plan.chain[i]);
  std::string why;
  if (verifyChainRealizesCollective(normalized, plan.chain, why))
    os << "semantics: verified\n";
  else
    os << "semantics: FAILED (" << why << ")\n";
  // The chain is sequential, so these two values are its only external
  // dependencies: the input feeds step 0 and the result is produced by the
  // last step.
  os << "input port: " << plan.inputPort.getType()
     << " (the collective's input_object operand) -> "
     << (plan.chain.empty() ? "result port" : "step 0") << "\n";
  os << "result port: " << plan.resultPort.getType() << " (the await result, "
     << llvm::range_size(plan.resultPort.getUses()) << " uses) <- "
     << (plan.chain.empty() ? "input port"
                            : "step " + std::to_string(plan.chain.size() - 1))
     << "\n";
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

    std::optional<MeshCostParams> params;
    if (chain) {
      std::vector<uint64_t> extents;
      for (PhysicalCommAxisType axis : meshAxisTypes)
        extents.push_back(axis.getExtent());
      params = MeshCostParams(extents);
      if (!bandwidths.empty()) {
        if (bandwidths.size() != meshAxisTypes.size()) {
          module.emitError("bandwidths needs one entry per physical axis");
          signalPassFailure();
          return;
        }
        params->bandwidth.assign(bandwidths.begin(), bandwidths.end());
      }
    }

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
      if (!chain)
        return;
      // Removes the candidates whose kind is disabled, so the alternatives a
      // cost-minimizing search never selects can still be exercised.
      std::vector<PrimitiveKind> disabled;
      if (disableAllToAll)
        disabled.push_back(PrimitiveKind::AllToAll);
      if (disableReduceScatter)
        disabled.push_back(PrimitiveKind::ReduceScatter);
      PlanOptions options;
      options.relayVariants = !disableRelayVariants;
      options.peelVariants = !disablePeelVariants;
      if (kOrder >= 0 || kVariant >= 0 || maxStates >= 0) {
        PlanBudget budget;
        if (kOrder >= 0)
          budget.kOrder = static_cast<size_t>(kOrder);
        if (kVariant >= 0)
          budget.kVariant = static_cast<size_t>(kVariant);
        if (maxStates >= 0)
          budget.maxStates = static_cast<size_t>(maxStates);
        options.budget = budget;
      }
      if (!disabled.empty())
        options.filter = [disabled](const DecomposerState &,
                                    std::vector<CandidateStep> &candidates) {
          llvm::erase_if(candidates, [&](const CandidateStep &candidate) {
            return llvm::is_contained(disabled, candidate.step.kind);
          });
        };
      std::optional<CollectivePlan> plan = planCollective(
          collective, meshAxisTypes, *params, failureReason, options);
      if (!plan) {
        collective->emitRemark()
            << "print-collective-plan: no chain (" << failureReason << ")";
        return;
      }
      collective->emitRemark() << describeChain(*plan, *normalized);
    });

    markAllAnalysesPreserved();
  }
};

} // namespace

} // namespace mlir::enzyme::distributed
