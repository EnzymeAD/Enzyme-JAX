#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h"

namespace mlir {
namespace enzyme {
namespace distributed {

#define GEN_PASS_DEF_LOCALIZEDISTRIBUTEDMODULEPASS
#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h.inc"

struct LocalizeDistributedModulePass
    : public enzyme::distributed::impl::LocalizeDistributedModulePassBase<
          LocalizeDistributedModulePass> {
  using LocalizeDistributedModulePassBase::LocalizeDistributedModulePassBase;

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();

    llvm::SmallVector<MeshComputationOp> meshComputations;
    moduleOp.walk([&](MeshComputationOp meshComputationOp) {
      meshComputations.push_back(meshComputationOp);
    });

    for (MeshComputationOp meshComputationOp : meshComputations) {
      if (failed(localizeMeshComputation(meshComputationOp))) {
        meshComputationOp.emitError()
            << "failed to run localize-distributed on mesh computation";
        signalPassFailure();
        return;
      }
    }
  }
};

} // namespace distributed
} // namespace enzyme
} // namespace mlir
