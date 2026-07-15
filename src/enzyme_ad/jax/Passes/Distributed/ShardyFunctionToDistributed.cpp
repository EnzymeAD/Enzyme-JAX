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
        funcOp.getLoc(), logicalMeshType, newLogicalAxes);

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
      funcOp.emitError() << "references unknown or invalid physical mesh symbol '"
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

    OpBuilder builder(funcOp);
    builder.setInsertionPointToStart(&funcOp.getBody().front());

    FailureOr<Value> logicalMesh =
      projectToPhysicalMesh(funcOp, builder, shardyMesh, *physicalMesh);
    if (failed(logicalMesh)) {
      funcOp.emitError()
          << "failed to factor shardy mesh " << shardyMesh
          << " onto physical mesh " << *physicalMesh;
      signalPassFailure();
      return;
    }
    // We need to convert the (logical) shardy mesh into its projection onto
    // the physical mesh. There may be multiple projections, and in the case
    // of communication heterogeneity, result in different performance!
    // We will try to project logical mesh axis to physical mesh axes using
    // their relative order, but this is a heuristic.

    // Functions are isolated from above, so the live-ins and live-outs
    // are just the function arguments and return values. We can then wrap the
    // body of the function in a mesh computation region for our dialect. We
    // need to:
    // - Define the return types (and counts) of the mesh computation to match
    // the return
    //   types of the function.
    // - Define the arg types of the mesh computation to match the argument
    // types
    //   of the function.
    // - Wire the SSA values between function and mesh.
    // - To do LATER: resolve submeshing, subdevices, and copy function code
    // into the mesh region.
  }
};

} // namespace distributed
} // namespace enzyme
} // namespace mlir
