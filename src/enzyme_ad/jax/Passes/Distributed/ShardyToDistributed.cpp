#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h"

#include "llvm/Support/ErrorHandling.h"

#include "src/enzyme_ad/jax/Dialect/Distributed/Dialect.h"

namespace mlir {
namespace enzyme {
namespace distributed {

#define GEN_PASS_DEF_SHARDYTODISTRIBUTEDPASS
#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h.inc"


struct ShardyToDistributedPass
    : public enzyme::distributed::impl::ShardyToDistributedPassBase<
          ShardyToDistributedPass> {
  using ShardyToDistributedPassBase::ShardyToDistributedPassBase;

  std::optional<PhysicalMeshOp> findPhysicalMesh(ModuleOp moduleOp) {
    std::optional<PhysicalMeshOp> physicalMesh;
    for (auto mesh : moduleOp.getOps<PhysicalMeshOp>()) {
      if (physicalMesh.has_value()) {
        llvm::report_fatal_error("multiple PhysicalMeshOps found in module");
      }
      physicalMesh = mesh;
    }
    return physicalMesh;
  }

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    OpBuilder builder(moduleOp.getContext());

    (void)builder;
    (void)findPhysicalMesh(moduleOp);
  }
};

} // namespace distributed
} // namespace enzyme
} // namespace mlir