#include "src/enzyme_ad/jax/Passes/Passes.h"

#include "mlir/Bytecode/BytecodeWriter.h"
#include "stablehlo/dialect/StablehloOps.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"

#include "src/enzyme_ad/jax/Dialect/TritonExt/Dialect.h"
#include "stablehlo/dialect/StablehloOps.h"

#include "llvm/ADT/DenseMap.h"

#define DEBUG_TYPE "lower-triton"

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_LOWERTRITONPASS
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

using namespace mlir;
using namespace mlir::enzyme;
using namespace mlir::enzymexla;
using namespace mlir::enzymexla::triton_ext;

void collectTritonKernels(
    DenseMap<triton_ext::TritonCallOp,
             std::pair<ModuleOp, triton_ext::TritonModuleOp>> &tritonKernels,
    SymbolTableCollection &symbolTable, triton_ext::TritonCallOp op) {
  auto funcOp = symbolTable.lookupNearestSymbolFrom(op, op.getFnAttr());
  if (!funcOp) {
    op->emitError() << "Failed to find function '" << op.getFn() << "' in "
                    << "module";
    return;
  }

  auto wrappedMod = funcOp->getParentOfType<ModuleOp>();
  if (!wrappedMod) {
    op->emitError() << "Failed to find parent built-in module.";
    return;
  }

  auto ttModOP = wrappedMod->getParentOfType<triton_ext::TritonModuleOp>();
  if (!ttModOP) {
    op->emitError() << "No `triton_ext.module` found!";
    return;
  }

  tritonKernels[op] = {wrappedMod, ttModOP};
  return;
}

static std::optional<uint64_t> getIntFromConstant(Value v) {
  if (!v)
    return 1;
  DenseIntElementsAttr attr;
  if (matchPattern(v, m_Constant(&attr))) {
    return attr.getSplatValue<APInt>().getZExtValue();
  }
  return std::nullopt;
}

struct LowerTritonPass
    : public mlir::enzyme::impl::LowerTritonPassBase<LowerTritonPass> {
  using Base::Base;

  void runOnOperation() override {
    auto modOp = getOperation();

    DenseMap<triton_ext::TritonCallOp,
             std::pair<ModuleOp, triton_ext::TritonModuleOp>>
        tritonKernels;
    SymbolTableCollection symbolTable;
    symbolTable.getSymbolTable(modOp);
    modOp->walk([&](triton_ext::TritonCallOp op) {
      collectTritonKernels(tritonKernels, symbolTable, op);
    });

    DenseSet<triton_ext::TritonModuleOp> modulesToErase;
    OpBuilder builder(modOp.getContext());

    bool anyFailed = false;
    for (auto [ttCallOp, innerPair] : tritonKernels) {
      auto [innerMod, ttModOP] = innerPair;
      modulesToErase.insert(ttModOP);
      int32_t numWarps = 4;
      if (innerMod->hasAttrOfType<IntegerAttr>("enzymexla.num_warps")) {
        numWarps = innerMod->getAttrOfType<IntegerAttr>("enzymexla.num_warps")
                       .getInt();
      }

      int32_t numStages = backend == "rocm" ? 1 : 3;
      if (innerMod->hasAttrOfType<IntegerAttr>("enzymexla.num_stages")) {
        numStages = innerMod->getAttrOfType<IntegerAttr>("enzymexla.num_stages")
                        .getInt();
      }

      std::string bytecode;
      llvm::raw_string_ostream os(bytecode);
      if (failed(writeBytecodeToFile(innerMod, os))) {
        ttCallOp.emitError("Failed to write bytecode");
        anyFailed = true;
        continue;
      }
      os.flush();

      std::string funcName = ttCallOp.getFn().getLeafReference().str();

      auto gx = getIntFromConstant(ttCallOp.getGridx());
      auto gy = getIntFromConstant(ttCallOp.getGridy());
      auto gz = getIntFromConstant(ttCallOp.getGridz());
      if (!gx || !gy || !gz) {
        ttCallOp.emitError(
            "Dynamic grid dims not supported for Triton lowering");
        anyFailed = true;
        continue;
      }

      SmallVector<NamedAttribute> config;
      config.push_back(
          builder.getNamedAttr("name", builder.getStringAttr(funcName)));
      config.push_back(
          builder.getNamedAttr("ir", builder.getStringAttr(os.str())));
      config.push_back(builder.getNamedAttr(
          "num_stages", builder.getI32IntegerAttr(numStages)));
      config.push_back(builder.getNamedAttr(
          "num_warps", builder.getI32IntegerAttr(numWarps)));
      config.push_back(
          builder.getNamedAttr("grid_x", builder.getI32IntegerAttr(*gx)));
      config.push_back(
          builder.getNamedAttr("grid_y", builder.getI32IntegerAttr(*gy)));
      config.push_back(
          builder.getNamedAttr("grid_z", builder.getI32IntegerAttr(*gz)));
      config.push_back(
          builder.getNamedAttr("debug", builder.getBoolAttr(false)));

      builder.setInsertionPoint(ttCallOp);
      auto customCall = stablehlo::CustomCallOp::create(
          builder, ttCallOp.getLoc(), ttCallOp.getResultTypes(),
          ttCallOp.getInputs());

      customCall.setCallTargetName("__gpu$xla.gpu.triton");
      customCall.setBackendConfigAttr(builder.getDictionaryAttr(config));
      customCall.setApiVersion(
          ::mlir::stablehlo::CustomCallApiVersion::API_VERSION_TYPED_FFI);

      if (auto attr = ttCallOp.getOperandLayoutsAttr()) {
        customCall.setOperandLayoutsAttr(mlir::cast<ArrayAttr>(attr));
      }
      if (auto attr = ttCallOp.getResultLayoutsAttr()) {
        customCall.setResultLayoutsAttr(mlir::cast<ArrayAttr>(attr));
      }
      if (auto attr = ttCallOp.getOutputOperandAliasesAttr()) {
        customCall.setOutputOperandAliasesAttr(mlir::cast<ArrayAttr>(attr));
      }

      if (!ttCallOp.getXlaSideEffectFreeAttr()) {
        customCall.setHasSideEffect(true);
      }

      ttCallOp.replaceAllUsesWith(customCall.getResults());
      ttCallOp.erase();
    }

    if (anyFailed) {
      signalPassFailure();
      return;
    }

    for (auto ttModOP : modulesToErase) {
      ttModOP.erase();
    }
  }
};
