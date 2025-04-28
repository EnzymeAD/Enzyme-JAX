#include "src/enzyme_ad/jax/Passes/Passes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_MARKFUNCTIONMEMORYEFFECTSPASS
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

using namespace mlir;
using namespace mlir::enzyme;

namespace {
struct MarkFunctionMemoryEffectsPass
    : public enzyme::impl::MarkFunctionMemoryEffectsPassBase<
          MarkFunctionMemoryEffectsPass> {

  void runOnOperation() override {
    ModuleOp module = getOperation();
    auto *ctx = module->getContext();
    OpBuilder builder(ctx);

    module.walk([&](FunctionOpInterface funcOp) {
      bool hasRead = false;
      bool hasWrite = false;
      bool hasAllocate = false;
      bool hasFree = false;

      funcOp.walk([&](Operation *op) {
        if (auto memOp = dyn_cast<MemoryEffectOpInterface>(op)) {
          SmallVector<MemoryEffects::EffectInstance, 4> effects;
          memOp.getEffects(effects);

          for (auto &effect : effects) {
            if (effect.getEffect() == MemoryEffects::Read::get())
              hasRead = true;
            else if (effect.getEffect() == MemoryEffects::Write::get())
              hasWrite = true;
            else if (effect.getEffect() == MemoryEffects::Allocate::get())
              hasAllocate = true;
            else if (effect.getEffect() == MemoryEffects::Free::get())
              hasFree = true;
          }
        }
      });

      if (!hasRead && !hasWrite && !hasAllocate && !hasFree)
        return; // No effects, don't attach attribute

      SmallVector<Attribute> effects;
      if (hasRead)
        effects.push_back(builder.getStringAttr("read"));
      if (hasWrite)
        effects.push_back(builder.getStringAttr("write"));
      if (hasAllocate)
        effects.push_back(builder.getStringAttr("allocate"));
      if (hasFree)
        effects.push_back(builder.getStringAttr("free"));

      funcOp->setAttr("enzymexla.memory_effects",
                      builder.getArrayAttr(effects));
    });
  }
};
} // namespace
