#include "src/enzyme_ad/jax/Passes/Passes.h"

#include "mlir/Analysis/CallGraph.h"
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

enum class VisitState { NotVisited, Visiting, Visited };

struct MarkFunctionMemoryEffectsPass
    : public enzyme::impl::MarkFunctionMemoryEffectsPassBase<
          MarkFunctionMemoryEffectsPass> {
  using Base::Base;

  DenseMap<CallGraphNode *, VisitState> visitState;
  SmallVector<CallGraphNode *> topoOrder;

  bool dfs(CallGraphNode *node) {
    auto it = visitState.find(node);
    if (it != visitState.end()) {
      if (it->second == VisitState::Visiting)
        return true; // cycle detected
      return false;
    }
    visitState[node] = VisitState::Visiting;
    for (auto &edge : *node) {
      CallGraphNode *target = edge.getTarget();
      if (dfs(target))
        return true;
    }
    visitState[node] = VisitState::Visited;
    topoOrder.push_back(node);
    return false;
  }

  void
  insertMemoryEffects(SmallVector<u_int8_t, 4> &effects,
                      SmallVector<MemoryEffects::EffectInstance> memEffects) {
    for (auto &effect : memEffects)
      insertMemoryEffects(effects, effect);
  }

  void insertMemoryEffects(SmallVector<u_int8_t, 4> &effects) {
    for (int i = 0; i < effects.size(); i++)
      effects[i] = 1;
  }

  void insertMemoryEffects(SmallVector<u_int8_t, 4> &effects,
                           MemoryEffects::EffectInstance effect) {
    if (effect.getEffect() == MemoryEffects::Read::get()) {
      effects[0] = 1;
    } else if (effect.getEffect() == MemoryEffects::Write::get()) {
      effects[1] = 1;
    } else if (effect.getEffect() == MemoryEffects::Allocate::get()) {
      effects[2] = 1;
    } else if (effect.getEffect() == MemoryEffects::Free::get()) {
      effects[3] = 1;
    } else {
      assert(false && "unknown memory effect");
    }
  }

  int64_t getNumEffects(SmallVector<u_int8_t, 4> &effects) {
    int64_t numEffects = 0;
    for (int i = 0; i < effects.size(); i++) {
      if (effects[i])
        numEffects++;
    }
    return numEffects;
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    auto *ctx = module->getContext();
    OpBuilder builder(ctx);

    DenseMap<SymbolRefAttr, SmallVector<u_int8_t, 4>> funcEffects;

    CallGraph callGraph(module);

    bool hasCycle = false;
    for (CallGraphNode *node : callGraph) {
      if (!visitState.count(node))
        hasCycle |= dfs(node);
    }

    // First pass: collect direct effects
    for (CallGraphNode *node : topoOrder) {
      if (node->isExternal())
        continue;

      Region *region = node->getCallableRegion();
      if (!region)
        return signalPassFailure();

      Operation *parentOp = region->getParentOp();
      auto funcOp = dyn_cast<FunctionOpInterface>(parentOp);
      if (!funcOp)
        return signalPassFailure();

      SmallVector<u_int8_t, 4> effects(4, 0);

      funcOp.walk([&](Operation *op) {
        if (op->hasTrait<OpTrait::HasRecursiveMemoryEffects>()) {
          auto maybeEffects = getEffectsRecursively(op);
          if (maybeEffects.has_value()) {
            insertMemoryEffects(effects, maybeEffects.value());
          } else {
            insertMemoryEffects(effects);
          }

          return WalkResult::skip();
        }

        if (auto memOp = dyn_cast<MemoryEffectOpInterface>(op)) {
          SmallVector<MemoryEffects::EffectInstance> memEffects;
          memOp.getEffects(memEffects);
          insertMemoryEffects(effects, memEffects);
        } else if (!assume_no_memory_effects) { // Operation doesn't define
                                                // memory effects
          insertMemoryEffects(effects);
        }

        return WalkResult::advance();
      });

      funcEffects[SymbolRefAttr::get(funcOp.getOperation())] =
          std::move(effects);
    }

    auto propagate = [&](FunctionOpInterface funcOp,
                         SmallVector<u_int8_t, 4> &effects) {
      funcOp.walk([&](Operation *op) {
        if (auto callOp = dyn_cast<CallOpInterface>(op)) {
          if (auto calleeAttr = callOp.getCallableForCallee()) {
            if (auto symRef = dyn_cast<SymbolRefAttr>(calleeAttr)) {

              auto funcEffectsSymRef = funcEffects.lookup(symRef);
              for (int i = 0; i < funcEffectsSymRef.size(); i++) {
                if (funcEffectsSymRef[i])
                  effects[i] = 1;
              }
            }
          }
        }
      });
    };

    if (hasCycle) {
      // Cycles: fixpoint iterate
      bool changed = true;
      int32_t iteration = 0;
      while (changed && iteration < max_iterations) {
        changed = false;
        iteration++;

        for (CallGraphNode *node : llvm::reverse(topoOrder)) {
          if (node->isExternal())
            continue;

          Region *region = node->getCallableRegion();
          if (!region)
            continue;

          Operation *parentOp = region->getParentOp();
          auto funcOp = dyn_cast<FunctionOpInterface>(parentOp);
          if (!funcOp)
            continue;

          auto &effects =
              funcEffects[SymbolRefAttr::get(ctx, funcOp.getName())];
          size_t before = getNumEffects(effects);
          propagate(funcOp, effects);
          changed = getNumEffects(effects) != before;
        }
      }

      // At this point if we haven't converged, we assume effects for all
      if (changed) {
        for (auto &[symbol, effects] : funcEffects)
          insertMemoryEffects(effects);
      }
    } else {
      // No cycles: reverse topological order and propagate
      for (CallGraphNode *node : llvm::reverse(topoOrder)) {
        if (node->isExternal())
          continue;

        Region *region = node->getCallableRegion();
        if (!region)
          continue;

        Operation *parentOp = region->getParentOp();
        auto funcOp = dyn_cast<FunctionOpInterface>(parentOp);
        if (!funcOp)
          continue;

        auto &effects = funcEffects[SymbolRefAttr::get(ctx, funcOp.getName())];
        propagate(funcOp, effects);
      }
    }

    // Finally, attach attributes
    for (auto &[symbol, effectsSet] : funcEffects) {
      auto funcOp = dyn_cast_or_null<FunctionOpInterface>(
          module.lookupSymbol(symbol.getLeafReference()));
      if (!funcOp)
        continue;

      SmallVector<Attribute> effectsAttrs;
      for (int i = 0; i < effectsSet.size(); i++) {
        if (effectsSet[i]) {
          if (i == 0) {
            effectsAttrs.push_back(builder.getStringAttr("read"));
          } else if (i == 1) {
            effectsAttrs.push_back(builder.getStringAttr("write"));
          } else if (i == 2) {
            effectsAttrs.push_back(builder.getStringAttr("allocate"));
          } else if (i == 3) {
            effectsAttrs.push_back(builder.getStringAttr("free"));
          } else {
            assert(false && "unknown memory effect");
          }
        }
      }

      funcOp->setAttr("enzymexla.memory_effects",
                      builder.getArrayAttr(effectsAttrs));
    }
  }
};
} // namespace
