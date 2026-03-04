#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h"

#include "mlir/IR/SymbolTable.h"

#include "shardy/dialect/sdy/ir/dialect.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"

namespace mlir {
namespace enzyme {
namespace distributed {

static constexpr llvm::StringLiteral kShardyAxisNamesAttr =
  "enzyme.shardy_axis_names";

#define GEN_PASS_DEF_SHARDYFUNCTIONTODISTRIBUTEDPASS
#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h.inc"

// True if a divides b (b / a is int).
bool isDivisible(int64_t a, int64_t b) { return a != 0 && b % a == 0; }

// Finds a "smallest"/least granular set of cuts such that different consecutive
// partitionings of the cut vector produce zones whose products produce a or b
// respectively. For example, cut([6, 4], [2, 12]) -> [2, 3, 4] because [(2, 3),
// 4] gives a and [2, (3, 4)] gives b. Note that if neither a // b or b // a we
// get stuck.
mlir::FailureOr<llvm::SmallVector<uint>>
find_cuts(llvm::SmallVector<uint> sizes_a, llvm::SmallVector<uint> sizes_b) {
  llvm::SmallVector<uint> cuts;
  if (sizes_a.empty() || sizes_b.empty()) {
    if (sizes_a.empty() && sizes_b.empty()) {
      return cuts;
    }
    return mlir::failure();
  }
  auto it_a = sizes_a.begin();
  auto it_b = sizes_b.begin();
  int val_a = *it_a;
  int val_b = *it_b;
  while (it_a != sizes_a.end() && it_b != sizes_b.end()) {
    bool a_div_b = isDivisible(val_a, val_b);
    bool b_div_a = isDivisible(val_b, val_a);
    if (!a_div_b && !b_div_a) {
      return mlir::failure();
    } else if (a_div_b) {
      cuts.push_back(val_a);
      val_b /= val_a;
      val_a = 1;
    } else { // b_div_a
      cuts.push_back(val_b);
      val_a /= val_b;
      val_b = 1;
    }
    // advance the iterator for any value that made the
    // cut, or both for equal values.
    if (val_a == 1) {
      ++it_a;
      val_a = (it_a != sizes_a.end()) ? *it_a : 1;
    }
    if (val_b == 1) {
      ++it_b;
      val_b = (it_b != sizes_b.end()) ? *it_b : 1;
    }
  }
  if (it_a != sizes_a.end() || it_b != sizes_b.end()) {
    // one reached end, other didn't: no cuts solution.
    return mlir::failure();
  }

  return cuts;
}

sdy::MeshAttr attr_to_mesh_attr(Attribute meshOrRef, Operation *op) {
  if (auto meshAttrDirect = llvm::dyn_cast<sdy::MeshAttr>(meshOrRef)) {
    return meshAttrDirect;
  } else if (auto meshRef = llvm::dyn_cast<FlatSymbolRefAttr>(meshOrRef)) {
    auto meshOp = llvm::dyn_cast_or_null<sdy::MeshOp>(
        SymbolTable::lookupNearestSymbolFrom(op, meshRef));
    return meshOp.getMeshAttr();
  } else {
    llvm_unreachable("expected mesh attribute or reference");
  }
}

llvm::LogicalResult
collectMeshFromRef(Attribute meshOrRef, Operation *op,
                   llvm::SmallVector<sdy::MeshAttr> &existing_meshes) {
  sdy::MeshAttr meshAttr = attr_to_mesh_attr(meshOrRef, op);

  bool alreadyCollected = false;
  for (auto collectedMesh : existing_meshes) {
    if (collectedMesh == meshAttr) {
      alreadyCollected = true;
      break;
    }
  }
  if (!alreadyCollected) {
    existing_meshes.push_back(meshAttr);
  }
  return success();
};

FailureOr<llvm::SmallVector<sdy::MeshAttr>>
collectShardyMeshes(func::FuncOp funcOp) {
  llvm::SmallVector<sdy::MeshAttr> shardyMeshes;
  bool hasCollectionFailure = false;
  // The function contains symbol references to some outside mesh definition.
  // Collect unique mesh definitions referenced here.
  funcOp.walk([&](Operation *op) -> WalkResult {
    for (auto attr : op->getAttrs()) {
      if (attr.getName().getValue() == "sdy.sharding") {
        auto shardingAttr = attr.getValue();

        // Look through either type of sharding attribute for mesh references.
        auto result = success();
        if (auto tensorShardingAttr =
                llvm::dyn_cast<sdy::TensorShardingAttr>(shardingAttr)) {
          result = collectMeshFromRef(tensorShardingAttr.getMeshOrRef(), op,
                                      shardyMeshes);
        } else if (auto perValueShardingAttr =
                       llvm::dyn_cast<sdy::TensorShardingPerValueAttr>(
                           shardingAttr)) {
          for (sdy::TensorShardingAttr valueSharding :
               perValueShardingAttr.getShardings()) {
            if (failed(collectMeshFromRef(valueSharding.getMeshOrRef(), op,
                                          shardyMeshes))) {
              result = failure();
              break;
            }
          }
        } else {
          llvm_unreachable("expected sharding attribute");
        }

        if (failed(result)) {
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

struct ShardyFunctionToDistributedPass
    : public enzyme::distributed::impl::ShardyFunctionToDistributedPassBase<
          ShardyFunctionToDistributedPass> {
  using ShardyFunctionToDistributedPassBase::
      ShardyFunctionToDistributedPassBase;

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

  /**
   * For a given logical shardy mesh, attempt to find a projection
   * onto the physical mesh, which may require factoring both physical
   * and logical axes.
   */
  FailureOr<Value> projectToPhysicalMesh(func::FuncOp funcOp,
                                         OpBuilder &builder,
                                         sdy::MeshAttr shardyMesh,
                                         PhysicalMeshOp physicalMesh) {
    // For each physical axis, take as much of the shardy mesh axis as we can
    // provide, then hand the next factor off to the next physical axis.
    // Requires that the logical mesh cleanly factors/divides. May result in a
    // mix of multiple products and factors.
    auto sdyAxes = shardyMesh.getAxes();
    auto physAxes = physicalMesh.getAxes();
    if (sdyAxes.empty() || physAxes.empty()) {
      return failure();
    }
    llvm::SmallVector<uint> sdyAxisSizes;
    llvm::SmallVector<uint> physAxisSizes;
    for (auto sdyAxis : sdyAxes) {
      sdyAxisSizes.push_back(sdyAxis.getSize());
    }
    for (auto physAxisAttr : physAxes) {
      auto physAxisRef = cast<FlatSymbolRefAttr>(physAxisAttr);
      FailureOr<PhysicalCommAxisOpInterface> physAxisInterface =
          resolvePhysicalAxisInterfaceFromAttr(physicalMesh, physAxisRef);
      physAxisSizes.push_back((*physAxisInterface).getPhysicalAxisSize());
    }

    auto maybe_cuts = find_cuts(sdyAxisSizes, physAxisSizes);
    if (failed(maybe_cuts)) {
      return failure("Failed to factor logical mesh onto physical axis.");
    }
    auto cuts = *maybe_cuts;

    // Insert factor ops for each physical axis corresponding to the cuts.
    llvm::SmallVector<Value> axisFactors;
    size_t cutIndex = 0;
    for (auto physicalAxisAttr : physAxes) {
      auto physicalAxisRef = cast<FlatSymbolRefAttr>(physicalAxisAttr);
      FailureOr<PhysicalCommAxisOpInterface> axisInterface =
          resolvePhysicalAxisInterfaceFromAttr(physicalMesh, physicalAxisRef);

      int64_t size = (*axisInterface).getPhysicalAxisSize();
      int64_t product = 1;
      llvm::SmallVector<int32_t> factors;
      while (product < size) {
        int64_t factor = cuts[cutIndex];
        factors.push_back(static_cast<int32_t>(factor));
        product *= factor;
        ++cutIndex;
      }

      // Build the factor op for this physical axis based on the cuts.
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
        // Shardy meshes ocasionally have size 1 axes, which realistiaclly don't
        // need to exist but are easier to pass through as factors with size 1.
        assert(cuts[cutIndex] == 1);
        factorValues.push_back(axisFactors[cutIndex]);
        ++cutIndex;
      }
      while (product < size) {
        factorValues.push_back(axisFactors[cutIndex]);
        product *= cuts[cutIndex];
        ++cutIndex;
      }

      if (factorValues.size() == 1) {
        newLogicalAxes.push_back(factorValues[0]);
      } else {
        auto productOp =
            builder.create<AxisProductOp>(funcOp.getLoc(), factorValues);
        newLogicalAxes.push_back(productOp.getLogicalAxis());
      }
    }

    // Build the logical mesh with the new axes.
    auto logicalMeshType = LogicalMeshType::get(funcOp.getContext());
    auto logicalMesh = builder.create<LogicalMeshOp>(
        funcOp.getLoc(), logicalMeshType, physicalMesh.getSymNameAttr(),
        newLogicalAxes);

    llvm::SmallVector<Attribute> shardyAxisNameAttrs;
    shardyAxisNameAttrs.reserve(sdyAxes.size());
    for (auto shardyAxis : sdyAxes) {
      shardyAxisNameAttrs.push_back(builder.getStringAttr(shardyAxis.getName()));
    }
    logicalMesh->setAttr(kShardyAxisNamesAttr,
                         builder.getArrayAttr(shardyAxisNameAttrs));

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

    // For now assert only one logical shardy mesh for this function.
    if ((*shardyMeshes).size() != 1) {
      // At some point there is a remeshing. We can support these by
      // partitioning the function, but for now we just fail.
      funcOp.emitError()
          << "cannot convert function referencing zero or multiple meshes";
      signalPassFailure();
      return;
    }
    auto shardyMesh = (*shardyMeshes)[0];

    // Copy old code into the body of the mesh op by creating a new entry block
    // and "stealing" the old entry block. Must copy arguments etc.
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
    builder.create<func::ReturnOp>(funcOp.getLoc(),
                                   meshComputationOp.getResults());

    if (failed(rewriteShardyCollectivesInFunction(funcOp))) {
      funcOp.emitError() << "failed to rewrite shardy collectives to "
                            "distributed collectives";
      signalPassFailure();
      return;
    }

    {// TODO remove
      mlir::OpPrintingFlags flags;
      flags.assumeVerified(true);
      llvm::outs() << "After ShardyFunctionToDistributedPass:\n";
      funcOp.print(llvm::outs(), flags);
      llvm::outs() << "\n";
    }
  }
};

} // namespace distributed
} // namespace enzyme
} // namespace mlir
