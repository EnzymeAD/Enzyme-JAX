#include "src/enzyme_ad/jax/Passes/Passes.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

#define DEBUG_TYPE "preserve-triton-warps-ctas"

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_PRESERVETRITONWARPSANDCTASPASS
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

using namespace mlir;
using namespace mlir::enzyme;

struct PreserveTritonWarpsAndCtasPass
    : public mlir::enzyme::impl::PreserveTritonWarpsAndCtasPassBase<
          PreserveTritonWarpsAndCtasPass> {
  using Base::Base;

  void runOnOperation() override {
    ModuleOp mod = getOperation();

    if (save && restore) {
      mod->emitError() << "save and restore cannot both be true";
      signalPassFailure();
      return;
    }

    if (save) {
      saveTritonWarpsAndCtas(mod);
      return;
    }

    if (restore) {
      restoreTritonWarpsAndCtas(mod);
      return;
    }

    return;
  }

private:
  void saveTritonWarpsAndCtas(mlir::ModuleOp module) {
    if (module->hasAttr("ttg.num-ctas")) {
      module->setAttr("enzymexla.ttg.num-ctas",
                      module->getAttr("ttg.num-ctas"));
    }
    if (module->hasAttr("ttg.num-warps")) {
      module->setAttr("enzymexla.ttg.num-warps",
                      module->getAttr("ttg.num-warps"));
    }
    if (module->hasAttr("ttg.threads-per-warp")) {
      module->setAttr("enzymexla.ttg.threads-per-warp",
                      module->getAttr("ttg.threads-per-warp"));
    }
  }

  void restoreTritonWarpsAndCtas(mlir::ModuleOp module) {
    if (module->hasAttr("enzymexla.ttg.num-ctas")) {
      module->setAttr("ttg.num-ctas",
                      module->getAttr("enzymexla.ttg.num-ctas"));
      module->removeAttr("enzymexla.ttg.num-ctas");
    }
    if (module->hasAttr("enzymexla.ttg.num-warps")) {
      module->setAttr("ttg.num-warps",
                      module->getAttr("enzymexla.ttg.num-warps"));
      module->removeAttr("enzymexla.ttg.num-warps");
    }
    if (module->hasAttr("enzymexla.ttg.threads-per-warp")) {
      module->setAttr("ttg.threads-per-warp",
                      module->getAttr("enzymexla.ttg.threads-per-warp"));
      module->removeAttr("enzymexla.ttg.threads-per-warp");
    }
  }
};
