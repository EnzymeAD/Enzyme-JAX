#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h"

#include "mlir/IR/SymbolTable.h"

#include "shardy/dialect/sdy/ir/dialect.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"

namespace mlir {
namespace enzyme {
namespace distributed {

#define GEN_PASS_DEF_SHARDYFUNCTIONTODISTRIBUTEDPASS
#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h.inc"

struct ShardyFunctionToDistributedPass
    : public enzyme::distributed::impl::ShardyFunctionToDistributedPassBase<
          ShardyFunctionToDistributedPass> {
  using ShardyFunctionToDistributedPassBase::
      ShardyFunctionToDistributedPassBase;

  FailureOr<llvm::SmallVector<sdy::MeshAttr>>
  collectShardyMeshes(func::FuncOp funcOp) {
    llvm::SmallVector<sdy::MeshAttr> shardyMeshes;
    bool hasCollectionFailure = false;
    // The function contains symbol references to some outside mesh definition.
    // Collect unique mesh definitions referenced here.
    funcOp.walk([&](Operation *op) -> WalkResult {
      for (auto attr : op->getAttrs()) {
        if (attr.getName().getValue() == "sdy.sharding") {
          auto collectMeshFromRef = [&](Attribute meshOrRef) -> LogicalResult {
            sdy::MeshAttr meshAttr;
            if (auto meshAttrDirect =
                    llvm::dyn_cast<sdy::MeshAttr>(meshOrRef)) {
              meshAttr = meshAttrDirect;
            } else if (auto meshRef =
                           llvm::dyn_cast<FlatSymbolRefAttr>(meshOrRef)) {
              auto meshOp = llvm::dyn_cast_or_null<sdy::MeshOp>(
                  SymbolTable::lookupNearestSymbolFrom(op, meshRef));
              if (!meshOp) {
                return failure();
              }
              meshAttr = meshOp.getMeshAttr();
            } else {
              return failure();
            }

            bool alreadyCollected = false;
            for (auto collectedMesh : shardyMeshes) {
              if (collectedMesh == meshAttr) {
                alreadyCollected = true;
                break;
              }
            }
            if (!alreadyCollected) {
              shardyMeshes.push_back(meshAttr);
            }
            return success();
          };

          auto shardingAttr = attr.getValue();
          if (auto tensorShardingAttr =
                  llvm::dyn_cast<sdy::TensorShardingAttr>(shardingAttr)) {
            if (failed(collectMeshFromRef(tensorShardingAttr.getMeshOrRef()))) {
              hasCollectionFailure = true;
              return WalkResult::interrupt();
            }
          } else if (auto perValueShardingAttr =
                         llvm::dyn_cast<sdy::TensorShardingPerValueAttr>(
                             shardingAttr)) {
            for (sdy::TensorShardingAttr valueSharding :
                 perValueShardingAttr.getShardings()) {
              if (failed(collectMeshFromRef(valueSharding.getMeshOrRef()))) {
                hasCollectionFailure = true;
                return WalkResult::interrupt();
              }
            }
          } else {
            hasCollectionFailure = true;
            return WalkResult::interrupt();
          }
        }
      }
      return WalkResult::advance();
    });

    if (hasCollectionFailure) {
      return failure();
    }
    return shardyMeshes;
  }

  FailureOr<PhysicalMeshOp> findPhysicalMesh(func::FuncOp funcOp) {
    auto meshRefAttr =
        FlatSymbolRefAttr::get(funcOp.getContext(), physicalMeshSymName);
    Operation *meshOp =
        SymbolTable::lookupNearestSymbolFrom(funcOp, meshRefAttr);
    if (!meshOp) {
      return failure();
    }

    auto physicalMesh = dyn_cast<PhysicalMeshOp>(meshOp);
    if (!physicalMesh) {
      return failure();
    }
    return physicalMesh;
  }

  // True if a divides b (b / a).
  bool isDivisible(int64_t a, int64_t b) { return a != 0 && b % a == 0; }

  FailureOr<Value> projectToPhysicalMesh(func::FuncOp funcOp,
                                         OpBuilder &builder,
                                         sdy::MeshAttr shardyMesh,
                                         PhysicalMeshOp physicalMesh) {
    // For each physical axis, take as much of the shardy mesh axis as we can
    // provide, then hand the next factor off to the next physical axis.
    // Requires that the logical mesh cleanly factors/divides. May result in a
    // mix of multiple products and factors.
    llvm::SmallVector<int64_t> cuts;
    auto sdyAxes = shardyMesh.getAxes();
    auto physAxes = physicalMesh.getAxes();

    if (sdyAxes.empty() || physAxes.empty()) {
      return failure();
    }

    size_t shardyAxisIndex = 0;
    size_t physicalAxisIndex = 0;
    int64_t shardyAxisSize = sdyAxes[shardyAxisIndex].getSize();
    FailureOr<PhysicalCommAxisOpInterface> physicalAxis =
        resolvePhysicalAxisInterfaceFromAttr(physicalMesh,
                                             physAxes[physicalAxisIndex]);
    if (failed(physicalAxis)) {
      return failure();
    }
    int64_t physicalAxisSize = (*physicalAxis).getPhysicalAxisSize();

    while (shardyAxisIndex < sdyAxes.size() &&
           physicalAxisIndex < physAxes.size()) {
      bool advancePhysicalAxis = false;
      bool advanceShardyAxis = false;
      if (shardyAxisSize == physicalAxisSize) {
        cuts.push_back(shardyAxisSize);
        advancePhysicalAxis = true;
        advanceShardyAxis = true;
      } else if (shardyAxisSize < physicalAxisSize) {
        if (!isDivisible(shardyAxisSize, physicalAxisSize)) {
          return failure();
        }
        cuts.push_back(shardyAxisSize);
        advanceShardyAxis = true;
        physicalAxisSize /= shardyAxisSize;
      } else { // shardyAxisSize > physicalAxisSize
        if (!isDivisible(physicalAxisSize, shardyAxisSize)) {
          return failure();
        }
        cuts.push_back(physicalAxisSize);
        advancePhysicalAxis = true;
        shardyAxisSize /= physicalAxisSize;
      }
      if (advancePhysicalAxis) {
        ++physicalAxisIndex;
        if (physicalAxisIndex < physAxes.size()) {
          physicalAxis = resolvePhysicalAxisInterfaceFromAttr(
              physicalMesh, physAxes[physicalAxisIndex]);
          if (failed(physicalAxis)) {
            return failure();
          }
          physicalAxisSize = (*physicalAxis).getPhysicalAxisSize();
        }
      }
      if (advanceShardyAxis) {
        ++shardyAxisIndex;
        if (shardyAxisIndex < sdyAxes.size()) {
          shardyAxisSize = sdyAxes[shardyAxisIndex].getSize();
        }
      }
    }
    if (shardyAxisIndex != sdyAxes.size() ||
        physicalAxisIndex != physAxes.size()) {
      return failure();
    }

    // Insert factor ops for each physical axis corresponding to the cuts.
    // Divisibility should be guaranteed by the above logic.
    llvm::SmallVector<Value> axisFactors;
    size_t cutIndex = 0;
    for (auto physicalAxisAttr : physAxes) {
      auto physicalAxisRef = cast<FlatSymbolRefAttr>(physicalAxisAttr);
      FailureOr<PhysicalCommAxisOpInterface> axisInterface =
          resolvePhysicalAxisInterfaceFromAttr(physicalMesh, physicalAxisRef);
      if (failed(axisInterface)) {
        return failure();
      }

      int64_t size = (*axisInterface).getPhysicalAxisSize();
      int64_t product = 1;
      llvm::SmallVector<int32_t> factors;
      while (product < size) {
        if (cutIndex >= cuts.size()) {
          return failure();
        }
        int64_t factor = cuts[cutIndex];
        factors.push_back(static_cast<int32_t>(factor));
        product *= factor;
        ++cutIndex;
      }
      if (product != size) {
        return failure();
      }

      auto factorAttr = builder.getI32ArrayAttr(factors);
      auto factorOp = builder.create<AxisFactorOp>(funcOp.getLoc(),
                                                   physicalAxisRef, factorAttr);
      axisFactors.append(factorOp.getLogicalAxes().begin(),
                         factorOp.getLogicalAxes().end());
    }

    llvm::SmallVector<Value> newLogicalAxes;
    cutIndex = 0;
    for (auto shardyAxis : sdyAxes) {
      int64_t size = shardyAxis.getSize();
      int64_t product = 1;
      llvm::SmallVector<Value> factorValues;
      if (size == 1) {
        assert(cuts[cutIndex] == 1);
        factorValues.push_back(axisFactors[cutIndex]);
        ++cutIndex;
      }
      while (product < size) {
        if (cutIndex >= cuts.size() || cutIndex >= axisFactors.size()) {
          return failure();
        }
        factorValues.push_back(axisFactors[cutIndex]);
        product *= cuts[cutIndex];
        ++cutIndex;
      }
      if (product != size) {
        return failure();
      }

      if (factorValues.size() == 1) {
        newLogicalAxes.push_back(factorValues[0]);
      } else {
        auto productOp =
            builder.create<AxisProductOp>(funcOp.getLoc(), factorValues);
        newLogicalAxes.push_back(productOp.getLogicalAxis());
      }
    }

    if (cutIndex != cuts.size()) {
      return failure();
    }

    auto logicalMeshType = LogicalMeshType::get(funcOp.getContext());
    auto logicalMesh = builder.create<LogicalMeshOp>(
      funcOp.getLoc(), logicalMeshType, physicalMesh.getSymNameAttr(),
      newLogicalAxes);

    return logicalMesh.getMesh();
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();

    if (physicalMeshSymName.empty()) {
      funcOp.emitError() << "physical mesh symbol was not provided to function "
                            "conversion pass";
      signalPassFailure();
      return;
    }

    // If the function has multiple blocks, we fail for now.
    if (!funcOp.getBody().hasOneBlock()) {
      funcOp.emitError() << "cannot convert function with multiple blocks";
      signalPassFailure();
      return;
    }

    FailureOr<PhysicalMeshOp> physicalMesh = findPhysicalMesh(funcOp);
    if (failed(physicalMesh)) {
      funcOp.emitError()
          << "references unknown or invalid physical mesh symbol '"
          << physicalMeshSymName << "'";
      signalPassFailure();
      return;
    }

    FailureOr<llvm::SmallVector<sdy::MeshAttr>> shardyMeshes =
        collectShardyMeshes(funcOp);
    if (failed(shardyMeshes)) {
      funcOp.emitError() << "failed to collect shardy meshes from function "
                            "sharding attributes";
      signalPassFailure();
      return;
    }

    if ((*shardyMeshes).size() != 1) {
      // At some point there is a remeshing. We can support these by
      // partitioning the function, but for now we just fail.
      funcOp.emitError()
          << "cannot convert function referencing zero or multiple meshes";
      signalPassFailure();
      return;
    }
    auto shardyMesh = (*shardyMeshes)[0];

    // Create a new entry block with identical block arguments as the
    // current function body block. We insert new ops into this block, allowing
    // the old block to be copied as-is in the future.
    Block *oldEntryBlock = &funcOp.getBody().front();
    llvm::SmallVector<Location> argLocs;
    argLocs.reserve(oldEntryBlock->getNumArguments());
    for (BlockArgument arg : oldEntryBlock->getArguments()) {
      argLocs.push_back(arg.getLoc());
    }
    Block *newEntryBlock = new Block();
    newEntryBlock->addArguments(oldEntryBlock->getArgumentTypes(), argLocs);
    funcOp.getBody().push_front(newEntryBlock);

    OpBuilder builder(funcOp);
    builder.setInsertionPointToStart(newEntryBlock);

    // Find the logical mesh projection and insert into the new entry block.
    FailureOr<Value> logicalMesh =
        projectToPhysicalMesh(funcOp, builder, shardyMesh, *physicalMesh);
    if (failed(logicalMesh)) {
      funcOp.emitError() << "failed to factor shardy mesh " << shardyMesh
                         << " onto physical mesh " << *physicalMesh;
      signalPassFailure();
      return;
    }

    // Create a new MeshComputationOp with the logical mesh in the new entry
    // block, where the argument and return types match that of the function.
    // Then we can move the old entry block into the region of the
    // MeshComputationOp.
    llvm::SmallVector<Type> meshComputationResultTypes =
        llvm::to_vector(funcOp.getResultTypes());
    llvm::SmallVector<Value> meshComputationInputs(
        newEntryBlock->getArguments().begin(),
        newEntryBlock->getArguments().end());
    auto meshComputationOp = builder.create<MeshComputationOp>(
        funcOp.getLoc(), meshComputationResultTypes, *logicalMesh,
        meshComputationInputs);

    Region &meshBody = meshComputationOp->getRegion(0);
    if (!meshBody.empty()) {
      meshBody.front().erase();
    }
    oldEntryBlock->moveBefore(&meshBody, meshBody.begin());

    auto oldReturn = cast<func::ReturnOp>(oldEntryBlock->getTerminator());

    builder.setInsertionPoint(oldReturn);
    builder.create<DistributedYieldOp>(oldReturn.getLoc(),
                                       oldReturn.getOperands());
    oldReturn.erase();

    builder.setInsertionPointToEnd(newEntryBlock);
    builder.create<func::ReturnOp>(funcOp.getLoc(), meshComputationOp.getResults());
  }
};

} // namespace distributed
} // namespace enzyme
} // namespace mlir
