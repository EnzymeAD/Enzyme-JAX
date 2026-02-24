#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h"

#include "src/enzyme_ad/jax/Dialect/Distributed/Dialect.h"

#include "mlir/Pass/PassManager.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace enzyme {
namespace distributed {

#define GEN_PASS_DEF_SHARDYTODISTRIBUTEDPASS
#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h.inc"

struct ShardyToDistributedPass
    : public enzyme::distributed::impl::ShardyToDistributedPassBase<
          ShardyToDistributedPass> {
  using ShardyToDistributedPassBase::ShardyToDistributedPassBase;

  FailureOr<PhysicalMeshOp> findTopLevelPhysicalMesh(ModuleOp moduleOp) {
    PhysicalMeshOp foundMesh;
    for (PhysicalMeshOp meshOp : moduleOp.getOps<PhysicalMeshOp>()) {
      if (foundMesh) {
        meshOp.emitError()
            << "multiple physical meshes in module not supported";
        return failure();
      }
      foundMesh = meshOp;
    }
    if (!foundMesh) {
      moduleOp.emitError() << "no top-level physical mesh found";
      return failure();
    }
    return foundMesh;
  }

  /**
   * Finds all functions where an argument or internal tensor value has a shardy
   * sharding. This may be a little fragile: revisit in the future.
   */
  llvm::SmallVector<func::FuncOp> findShardyFunctions(ModuleOp moduleOp) {
    llvm::SmallVector<func::FuncOp> shardyFunctions;
    // Walk or just iterate?
    moduleOp.walk([&](func::FuncOp funcOp) {
      bool hasShardySharding = false;
      // Iterate over ops. If any have a sharding attribute,
      // consider the function shardy and add to the list.
      // Check the function arguments for sharding attributes as well.
      //   for (auto arg : funcOp.getArguments()) {
      //     if (arg.getAttrOfType<StringAttr>("sdy.sharding")) {
      //       hasShardySharding = true;
      //       break;
      //     }
      //   }
      funcOp.walk([&](Operation *op) {
        for (auto attr : op->getAttrs()) {
          if (attr.getName().getValue() == "sdy.sharding") {
            hasShardySharding = true;
            return WalkResult::interrupt();
          }
        }
        return WalkResult::advance();
      });
      if (hasShardySharding) {
        shardyFunctions.push_back(funcOp);
      }
    });
    return shardyFunctions;
  }

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();

    FailureOr<PhysicalMeshOp> physicalMesh = findTopLevelPhysicalMesh(moduleOp);
    if (failed(physicalMesh)) {
      signalPassFailure();
      return;
    }

    llvm::SmallVector<func::FuncOp> shardyFunctions =
        findShardyFunctions(moduleOp);

    ShardyFunctionToDistributedPassOptions functionPassOptions;
    functionPassOptions.physicalMeshSymName =
        (*physicalMesh).getSymNameAttr().getValue().str();

    for (func::FuncOp funcOp : shardyFunctions) {
      OpPassManager functionPassManager(func::FuncOp::getOperationName());
      functionPassManager.addPass(
          createShardyFunctionToDistributedPass(functionPassOptions));

      if (failed(runPipeline(functionPassManager, funcOp))) {
        funcOp.emitError() << "failed to run shardy function conversion pass";
        signalPassFailure();
        return;
      }
    }
  }
};

} // namespace distributed
} // namespace enzyme
} // namespace mlir