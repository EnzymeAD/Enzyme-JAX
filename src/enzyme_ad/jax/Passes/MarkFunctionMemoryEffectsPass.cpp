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

  void insertMemoryEffectsAsStringRefs(
      llvm::SmallDenseSet<StringRef> &effects,
      SmallVector<MemoryEffects::EffectInstance> memEffects) {
    for (auto &effect : memEffects) {
      if (effect.getEffect() == MemoryEffects::Read::get()) {
        effects.insert("read");
      } else if (effect.getEffect() == MemoryEffects::Write::get()) {
        effects.insert("write");
      } else if (effect.getEffect() == MemoryEffects::Allocate::get()) {
        effects.insert("allocate");
      } else if (effect.getEffect() == MemoryEffects::Free::get()) {
        effects.insert("free");
      } else {
        assert(false && "unknown memory effect");
      }
    }
  }

  void
  insertMemoryEffectsAsStringRefs(llvm::SmallDenseSet<StringRef> &effects) {
    effects.insert("read");
    effects.insert("write");
    effects.insert("allocate");
    effects.insert("free");
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    auto *ctx = module->getContext();
    OpBuilder builder(ctx);

    DenseMap<SymbolRefAttr, llvm::SmallDenseSet<StringRef>> funcEffects;

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

      llvm::SmallDenseSet<StringRef> effects;

      funcOp.walk([&](Operation *op) {
        if (op->hasTrait<OpTrait::HasRecursiveMemoryEffects>()) {
          auto maybeEffects = getEffectsRecursively(op);
          if (maybeEffects.has_value()) {
            insertMemoryEffectsAsStringRefs(effects, maybeEffects.value());
          } else {
            insertMemoryEffectsAsStringRefs(effects);
          }

          return WalkResult::skip();
        }

        if (auto memOp = dyn_cast<MemoryEffectOpInterface>(op)) {
          SmallVector<MemoryEffects::EffectInstance> memEffects;
          memOp.getEffects(memEffects);
          insertMemoryEffectsAsStringRefs(effects, memEffects);
        } else if (!assume_no_memory_effects) { // Operation doesn't define
                                                // memory effects
          insertMemoryEffectsAsStringRefs(effects);
        }

        return WalkResult::advance();
      });

      funcEffects[SymbolRefAttr::get(funcOp.getOperation())] =
          std::move(effects);
    }

    auto propagate = [&](FunctionOpInterface funcOp,
                         llvm::SmallDenseSet<StringRef> &effects) {
      funcOp.walk([&](Operation *op) {
        if (auto callOp = dyn_cast<CallOpInterface>(op)) {
          if (auto calleeAttr = callOp.getCallableForCallee()) {
            if (auto symRef = dyn_cast<SymbolRefAttr>(calleeAttr)) {

              for (auto &e : funcEffects.lookup(symRef))
                effects.insert(e);
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
          size_t before = effects.size();
          propagate(funcOp, effects);
          if (effects.size() != before)
            changed = true;
        }
      }

      // At this point if we haven't converged, we assume effects for all
      if (changed) {
        for (auto &[symbol, effectsSet] : funcEffects) {
          effectsSet.insert("read");
          effectsSet.insert("write");
          effectsSet.insert("allocate");
          effectsSet.insert("free");
        }
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
      for (auto effect : effectsSet)
        effectsAttrs.push_back(builder.getStringAttr(effect));

      funcOp->setAttr("enzymexla.memory_effects",
                      builder.getArrayAttr(effectsAttrs));
    }
  }
};
} // namespace
