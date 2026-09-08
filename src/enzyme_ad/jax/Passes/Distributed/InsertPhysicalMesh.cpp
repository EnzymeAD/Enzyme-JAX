#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/MemoryBuffer.h"

#include <limits>
#include <string>

namespace mlir::enzyme::distributed {

#define GEN_PASS_DEF_INSERTPHYSICALMESHPASS
#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h.inc"

namespace {

struct InsertPhysicalMeshPass
    : public impl::InsertPhysicalMeshPassBase<InsertPhysicalMeshPass> {
  using InsertPhysicalMeshPassBase::InsertPhysicalMeshPassBase;

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();

    // Validate that exactly one of config string or file is specified
    bool hasString = !configurationString.empty();
    bool hasFile = !configurationFile.empty();

    if (!hasString && !hasFile) {
      moduleOp.emitError()
          << "insert-physical-mesh: must specify either configuration-string "
             "or configuration-file";
      signalPassFailure();
      return;
    }

    if (hasString && hasFile) {
      moduleOp.emitError()
          << "insert-physical-mesh: cannot specify both configuration-string "
             "and configuration-file";
      signalPassFailure();
      return;
    }

    Block *block = &moduleOp.getBodyRegion().front();
    ParserConfig parserConfig = ParserConfig(moduleOp.getContext(), false);

    // Read the configuration
    LogicalResult parseResult = mlir::success();
    if (hasFile) {
      parseResult =
          mlir::parseSourceFile(configurationFile, block, parserConfig);
    } else {
      parseResult =
          mlir::parseSourceString(configurationString, block, parserConfig);
    }

    if (failed(parseResult)) {
      moduleOp.emitError()
          << "insert-physical-mesh: failed to parse configuration";
      signalPassFailure();
      return;
    }

    // Assert that we read an actual PhysicalMeshOp from the configuration
    // Relies on parsing into a block inserting at the back
    auto lastInsertedOp = &block->back();
    PhysicalMeshOp meshOp = dyn_cast<PhysicalMeshOp>(lastInsertedOp);
    if (!meshOp) {
      // technically we don't know if the last inserted op is the only op added,
      // but that's fine for our purposes.
      moduleOp.emitError() << "insert-physical-mesh: configuration was not "
                              "exactly a PhysicalMeshOp";
      signalPassFailure();
      return;
    }

    // Create a GetPhysicalMeshAxesOp to expose the mesh axes
    OpBuilder builder(moduleOp.getContext());
    builder.setInsertionPoint(
        moduleOp.getBody(),
        std::next(moduleOp.getBodyRegion().front().begin()));

    auto meshSymRef = FlatSymbolRefAttr::get(meshOp.getSymNameAttr());
    SmallVector<Type> axisTypes;
    for (Attribute axisAttr : meshOp.getAxesAttr()) {
      auto typeAttr = cast<TypeAttr>(axisAttr);
      axisTypes.push_back(typeAttr.getValue());
    }

    builder.create<GetPhysicalMeshAxesOp>(meshOp.getLoc(), axisTypes,
                                          meshSymRef);
  }
};

} // namespace
} // namespace mlir::enzyme::distributed
