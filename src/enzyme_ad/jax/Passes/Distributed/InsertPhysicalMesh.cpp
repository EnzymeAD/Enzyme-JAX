#include "src/enzyme_ad/jax/Passes/Distributed/FindShardyFunctionsAnalysis.h"
#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/Support/FormatVariadic.h"

#include <limits>

namespace mlir::enzyme::distributed {

#define GEN_PASS_DEF_INSERTPHYSICALMESHPASS
#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h.inc"

namespace {

struct InsertPhysicalMeshPass
    : public impl::InsertPhysicalMeshPassBase<InsertPhysicalMeshPass> {
  using InsertPhysicalMeshPassBase::InsertPhysicalMeshPassBase;

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();

    unsigned physicalMeshCount = 0;
    for (PhysicalMeshOp meshOp : moduleOp.getOps<PhysicalMeshOp>()) {
      (void)meshOp;
      ++physicalMeshCount;
      if (physicalMeshCount > 1) {
        moduleOp.emitError() << "expected at most one distributed physical "
                                "mesh in module, found "
                             << physicalMeshCount;
        signalPassFailure();
        return;
      }
    }

    if (physicalMeshCount == 1) {
      return;
    }

    const FindShardyFunctionsAnalysis &analysis =
        getAnalysis<FindShardyFunctionsAnalysis>();
    if (!analysis.isValid()) {
      signalPassFailure();
      return;
    }

    if (analysis.getShardyFunctions().empty()) {
      moduleOp.emitRemark()
          << "no shardy functions found; skipping physical mesh insertion";
      return;
    }

    sdy::MeshAttr commonMesh = nullptr;
    for (const FindShardyFunctionsAnalysis::FunctionInfo &info :
         analysis.getShardyFunctions()) {
      if (info.meshes.size() != 1) {
        moduleOp.emitError()
            << "expected shardy function to have exactly one mesh, found "
            << info.meshes.size() << " in function " << info.symName;
        signalPassFailure();
        return;
      }

      if (!commonMesh) {
        commonMesh = info.meshes[0];
        continue;
      }

      if (commonMesh != info.meshes[0]) {
        moduleOp.emitError()
            << "expected all shardy functions to share one mesh, found "
            << commonMesh << " and " << info.meshes[0];
        signalPassFailure();
        return;
      }
    }

    if (!commonMesh) {
      moduleOp.emitError()
          << "failed to infer a common shardy mesh for physical mesh insertion";
      signalPassFailure();
      return;
    }

    SmallVector<Attribute> axisAttrs;
    axisAttrs.reserve(commonMesh.getAxes().size());
    for (sdy::MeshAxisAttr axis : commonMesh.getAxes()) {
      int64_t extent = axis.getSize();
      if (extent <= 0 ||
          extent > static_cast<int64_t>(std::numeric_limits<unsigned>::max())) {
        moduleOp.emitError() << "unsupported shardy mesh axis size " << extent
                             << " for axis " << axis.getName();
        signalPassFailure();
        return;
      }

      Type axisType = PhysicalCommAxisType::get(moduleOp.getContext(),
                                                static_cast<unsigned>(extent));
      axisAttrs.push_back(TypeAttr::get(axisType));
    }

    std::string symbolName = "auto_pmesh";
    unsigned suffix = 0;
    while (moduleOp.lookupSymbol(symbolName)) {
      symbolName = llvm::formatv("auto_pmesh_{0}", ++suffix).str();
    }

    OpBuilder builder(moduleOp.getContext());
    builder.setInsertionPointToStart(moduleOp.getBody());
    builder.create<PhysicalMeshOp>(
        moduleOp.getLoc(), builder.getStringAttr(symbolName),
        builder.getStringAttr("mock"), builder.getArrayAttr(axisAttrs));
  }
};

} // namespace
} // namespace mlir::enzyme::distributed
