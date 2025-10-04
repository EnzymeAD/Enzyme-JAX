#include "src/enzyme_ad/jax/Passes/Passes.h"

#include "mlir/Analysis/CallGraph.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/BitVector.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "src/enzyme_ad/jax/Dialect/Ops.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include <queue>

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
  insertMemoryEffects(BitVector &effects,
                      SmallVector<MemoryEffects::EffectInstance> memEffects) {
    for (auto &effect : memEffects)
      insertMemoryEffects(effects, effect);
  }

  void insertMemoryEffects(BitVector &effects) {
    effects.set(0, effects.size());
  }

  void insertMemoryEffects(BitVector &effects, BitVector &argEffects) {
    for (int i = 0; i < effects.size(); i++) {
      if (argEffects[i]) {
        effects.set(i);
      }
    }
  }

  void insertMemoryEffects(BitVector &effects,
                           MemoryEffects::EffectInstance effect) {
    if (effect.getEffect() == MemoryEffects::Read::get()) {
      effects.set(0);
    } else if (effect.getEffect() == MemoryEffects::Write::get()) {
      effects.set(1);
    } else if (effect.getEffect() == MemoryEffects::Allocate::get()) {
      effects.set(2);
    } else if (effect.getEffect() == MemoryEffects::Free::get()) {
      effects.set(3);
    } else {
      assert(false && "unknown memory effect");
    }
  }

  int64_t getNumEffects(BitVector &effects) {
    int64_t numEffects = 0;
    for (int i = 0; i < effects.size(); i++) {
      if (effects[i])
        numEffects++;
    }
    return numEffects;
  }

  struct EffectInfo {
    ArrayAttr enzymexlaEffects;
    bool readOnly;
    bool writeOnly;
    bool readNone;
  };

  EffectInfo getEffectInfo(OpBuilder &builder, BitVector &effects) {
    EffectInfo info;
    info.readOnly = effects[0];
    info.writeOnly = effects[1];
    info.readNone = !effects[0] && !effects[1];
    SmallVector<Attribute> effectsAttrs;

    if (effects[0]) {
      info.writeOnly = false;
      effectsAttrs.push_back(builder.getStringAttr("read"));
    }

    if (effects[1]) {
      info.readOnly = false;
      effectsAttrs.push_back(builder.getStringAttr("write"));
    }

    if (effects[2]) {
      info.writeOnly = false;
      info.readOnly = false;
      info.readNone = false;
      effectsAttrs.push_back(builder.getStringAttr("allocate"));
    }

    if (effects[3]) {
      info.writeOnly = false;
      info.readOnly = false;
      info.readNone = false;
      effectsAttrs.push_back(builder.getStringAttr("free"));
    }

    info.enzymexlaEffects = builder.getArrayAttr(effectsAttrs);
    return info;
  }

  int32_t getArgIndex(CallOpInterface callOp, OpOperand *operand) {
    auto callOperands = callOp.getArgOperands();
    for (unsigned i = 0; i < callOperands.size(); i++) {
      if (callOperands[i] == operand->get())
        return i;
    }
    assert(false && "operand not found");
    return -1;
  }

  // TODO: at some point, we should reuse pre-existing attributes (see
  // jitcallsideeffect2.mlir)
  void handleCallOpInterface(
      CallOpInterface callOp, OpOperand *operand, BitVector &effects,
      DenseMap<SymbolRefAttr, SmallVector<BitVector>> &funcArgEffects) {
    if (auto calleeAttr = callOp.getCallableForCallee()) {
      if (auto symRef = dyn_cast<SymbolRefAttr>(calleeAttr)) {
        if (funcArgEffects.contains(symRef)) {
          auto &argEffects = funcArgEffects[symRef];
          insertMemoryEffects(effects,
                              argEffects[getArgIndex(callOp, operand)]);
          return;
        } else {
          insertMemoryEffects(effects);
          return;
        }
      }
    } else {
      insertMemoryEffects(effects);
    }
  }

  bool isPointerType(Value v) { return isPointerType(v.getType()); }

  bool isPointerType(Type t) {
    return isa<LLVM::LLVMPointerType, MemRefType, triton::PointerType>(t);
  }

  void analyzeMemoryEffects(
      Operation *op, OpOperand *operand, BitVector &effects,
      DenseMap<SymbolRefAttr, SmallVector<BitVector>> &funcArgEffects) {
    auto memEffectsOrNothing = getEffectsRecursively(op);
    if (!memEffectsOrNothing.has_value()) {
      insertMemoryEffects(effects);
      return;
    }
    auto &memEffects = memEffectsOrNothing.value();

    for (const auto &effect : memEffects) {
      if (!effect.getValue() ||
          (effect.getValue() && effect.getValue() == operand->get())) {
        if (isa<MemoryEffects::Read>(effect.getEffect())) {
          effects.set(0);
        } else if (isa<MemoryEffects::Write>(effect.getEffect())) {
          effects.set(1);
        } else if (isa<MemoryEffects::Allocate>(effect.getEffect())) {
          effects.set(2);
        } else if (isa<MemoryEffects::Free>(effect.getEffect())) {
          effects.set(3);
        } else {
          assert(false && "unknown memory effect");
        }
      }
    }
  }

  void analyzeFunctionArgumentMemoryEffects(
      FunctionOpInterface funcOp, SmallVector<BitVector> &argEffects,
      DenseMap<SymbolRefAttr, SmallVector<BitVector>> &funcArgEffects) {
    auto *ctx = funcOp->getContext();
    OpBuilder builder(ctx);

    DenseMap<Value, unsigned> valueToArgIndex;
    for (unsigned i = 0; i < funcOp.getNumArguments(); i++) {
      valueToArgIndex[funcOp.getArgument(i)] = i;
    }

    // BFS traversal starting from arguments
    std::queue<Value> worklist;
    DenseSet<Value> visited;
    for (unsigned i = 0; i < funcOp.getNumArguments(); i++) {
      Value arg = funcOp.getArgument(i);
      worklist.push(arg);
      visited.insert(arg);
    }

    // BFS through the graph
    while (!worklist.empty()) {
      Value cur = worklist.front();
      worklist.pop();

      auto argIt = valueToArgIndex.find(cur);
      if (argIt == valueToArgIndex.end())
        continue;
      unsigned argIndex = argIt->second;

      for (OpOperand &use : cur.getUses()) {
        Operation *user = use.getOwner();

        if (auto callOp = dyn_cast<CallOpInterface>(user)) {
          handleCallOpInterface(callOp, &use, argEffects[argIndex],
                                funcArgEffects);
        } else {
          analyzeMemoryEffects(user, &use, argEffects[argIndex],
                               funcArgEffects);
        }

        for (auto result : user->getResults()) {
          if (visited.insert(result).second) {
            valueToArgIndex[result] = argIndex;
            worklist.push(result);
          }
        }
      }
    }
  }

  SymbolRefAttr getFullReference(FunctionOpInterface funcOp) {
    SmallVector<StringRef> symbolPath;
    auto ctx = funcOp.getOperation()->getContext();
    auto op = funcOp.getOperation()->getParentOp();
    while (op) {
      if (auto symbolOp = dyn_cast<SymbolOpInterface>(op)) {
        symbolPath.push_back(symbolOp.getName());
      }
      op = op->getParentOp();
    }
    if (symbolPath.empty()) {
      return SymbolRefAttr::get(funcOp.getOperation());
    }
    SmallVector<FlatSymbolRefAttr> nestedRefs;
    for (int i = 1; i < symbolPath.size(); i++) {
      nestedRefs.push_back(FlatSymbolRefAttr::get(ctx, symbolPath[i]));
    }
    nestedRefs.push_back(FlatSymbolRefAttr::get(ctx, funcOp.getNameAttr()));
    return SymbolRefAttr::get(ctx, symbolPath[0], nestedRefs);
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    auto *ctx = module->getContext();
    OpBuilder builder(ctx);

    DenseMap<SymbolRefAttr, BitVector> funcEffects;
    DenseMap<SymbolRefAttr, SmallVector<BitVector>> funcArgEffects;
    DenseMap<SymbolRefAttr, FunctionOpInterface> symbolToFunc;

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

      BitVector effects(4, 0);
      SmallVector<BitVector> argEffects;
      argEffects.reserve(funcOp.getNumArguments());
      for (unsigned i = 0; i < funcOp.getNumArguments(); i++) {
        argEffects.push_back(BitVector(4, 0));
      }

      funcOp.walk([&](Operation *op) {
        if (op->hasTrait<OpTrait::HasRecursiveMemoryEffects>()) {
          return WalkResult::advance();
        }

        if (op == funcOp)
          return WalkResult::advance();

        if (auto jitcall = dyn_cast<enzymexla::JITCallOp>(op)) {
          if (jitcall.getXlaSideEffectFreeAttr()) {
            return WalkResult::advance();
          } else if (!assume_no_memory_effects) {
            insertMemoryEffects(effects);
          }
        } else if (auto kcall = dyn_cast<enzymexla::KernelCallOp>(op)) {
          if (kcall.getXlaSideEffectFreeAttr()) {
            return WalkResult::advance();
          } else {
            insertMemoryEffects(effects);
          }
        } else if (auto ccall = dyn_cast<stablehlo::CustomCallOp>(op)) {
          if (!ccall.getHasSideEffect()) {
            return WalkResult::advance();
          } else {
            insertMemoryEffects(effects);
          }
        } else if (auto memOp = dyn_cast<MemoryEffectOpInterface>(op)) {
          SmallVector<MemoryEffects::EffectInstance> memEffects;
          memOp.getEffects(memEffects);
          insertMemoryEffects(effects, memEffects);
        } else if (!assume_no_memory_effects) { // Operation doesn't define
                                                // memory effects
          insertMemoryEffects(effects);
        }

        return WalkResult::advance();
      });

      auto symRef = getFullReference(funcOp);
      funcEffects[symRef] = std::move(effects);
      funcArgEffects[symRef] = std::move(argEffects);
      symbolToFunc[symRef] = funcOp;
    }

    auto propagate = [&](FunctionOpInterface funcOp, BitVector &effects) {
      funcOp.walk([&](Operation *op) {
        if (auto callOp = dyn_cast<CallOpInterface>(op)) {
          if (auto calleeAttr = callOp.getCallableForCallee()) {
            if (auto symRef = dyn_cast<SymbolRefAttr>(calleeAttr)) {

              auto funcEffectsSymRef = funcEffects.lookup(symRef);
              for (int i = 0; i < funcEffectsSymRef.size(); i++) {
                if (funcEffectsSymRef[i])
                  effects.set(i);
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

          auto symRef = getFullReference(funcOp);
          analyzeFunctionArgumentMemoryEffects(funcOp, funcArgEffects[symRef],
                                               funcArgEffects);
          auto &effects = funcEffects[symRef];
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
      for (CallGraphNode *node : topoOrder) {
        if (node->isExternal())
          continue;

        Region *region = node->getCallableRegion();
        if (!region)
          continue;

        Operation *parentOp = region->getParentOp();
        auto funcOp = dyn_cast<FunctionOpInterface>(parentOp);
        if (!funcOp)
          continue;

        auto symRef = getFullReference(funcOp);
        analyzeFunctionArgumentMemoryEffects(funcOp, funcArgEffects[symRef],
                                             funcArgEffects);
        auto &effects = funcEffects[symRef];
        propagate(funcOp, effects);
      }
    }

    // Finally, attach attributes
    for (auto &[symbol, effectsSet] : funcEffects) {
      auto funcOp = symbolToFunc[symbol];
      auto funcEffectInfo = getEffectInfo(builder, effectsSet);
      funcOp->setAttr("enzymexla.memory_effects",
                      funcEffectInfo.enzymexlaEffects);

      auto &argEffects = funcArgEffects[symbol];
      for (unsigned i = 0; i < funcOp.getNumArguments(); i++) {
        auto argEffectInfo = getEffectInfo(builder, argEffects[i]);
        funcOp.setArgAttr(i, "enzymexla.memory_effects",
                          argEffectInfo.enzymexlaEffects);

        if (isPointerType(funcOp.getArgument(i))) {
          if (argEffectInfo.readOnly) {
            funcOp.setArgAttr(i, LLVM::LLVMDialect::getReadonlyAttrName(),
                              builder.getUnitAttr());
          }
          if (argEffectInfo.writeOnly) {
            funcOp.setArgAttr(i, LLVM::LLVMDialect::getWriteOnlyAttrName(),
                              builder.getUnitAttr());
          }
          // if (argEffectInfo.readNone) {
          //   funcOp.setArgAttr(i, LLVM::LLVMDialect::getReadnoneAttrName(),
          //                     builder.getUnitAttr());
          // }
          if (!argEffects[i][3]) {
            funcOp.setArgAttr(i, LLVM::LLVMDialect::getNoFreeAttrName(),
                              builder.getUnitAttr());
          }
        }
      }
    }
  }
};

} // namespace
