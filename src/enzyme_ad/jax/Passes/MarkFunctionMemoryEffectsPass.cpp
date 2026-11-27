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
#include "src/enzyme_ad/jax/Dialect/TritonExt/Ops.h"
#include "stablehlo/dialect/StablehloOps.h"

#ifndef ENZYME_JAX_ENABLE_TRITON
#define ENZYME_JAX_ENABLE_TRITON 1
#endif

#if ENZYME_JAX_ENABLE_TRITON
#include "triton/Dialect/Triton/IR/Dialect.h"
#endif

#include <queue>

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_MARKFUNCTIONMEMORYEFFECTSPASS
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

using namespace mlir;
using namespace mlir::enzyme;
using namespace mlir::enzymexla;

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
#if ENZYME_JAX_ENABLE_TRITON
    if (isa<triton::PointerType>(t))
      return true;
#endif
    return isa<LLVM::LLVMPointerType, MemRefType>(t);
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

  // Returns true if `use` may capture the pointer/memref value `use.get()`.
  // Otherwise returns false and appends any values that may still carry an
  // address derived from `use.get()` to `derivedValues`.
  bool analyzeOperandCapture(
      OpOperand &use, SmallVectorImpl<Value> &derivedValues,
      const DenseMap<SymbolRefAttr, BitVector> &funcArgCaptured) {
    Operation *user = use.getOwner();

    if (auto callOp = dyn_cast<CallOpInterface>(user)) {
      MutableOperandRange argOperands = callOp.getArgOperandsMutable();
      int callArgIdx = -1;
      for (unsigned k = 0, e = argOperands.size(); k < e; ++k) {
        if (&argOperands[k] == &use) {
          callArgIdx = static_cast<int>(k);
          break;
        }
      }
      if (callArgIdx < 0)
        return true;
      auto calleeAttr = callOp.getCallableForCallee();
      auto symRef = dyn_cast_if_present<SymbolRefAttr>(calleeAttr);
      if (!symRef)
        return true;
      auto it = funcArgCaptured.find(symRef);
      if (it == funcArgCaptured.end() ||
          static_cast<unsigned>(callArgIdx) >= it->second.size())
        return true;
      return it->second.test(callArgIdx);
    }

    // Memory loads read from the pointer without capturing its address.
    if (auto op = dyn_cast<LLVM::LoadOp>(user))
      return &use != &op.getAddrMutable();
    if (auto op = dyn_cast<memref::LoadOp>(user))
      return &use != &op.getMemrefMutable();
    if (auto op = dyn_cast<affine::AffineLoadOp>(user))
      return &use != &op.getMemrefMutable();
    if (auto op = dyn_cast<affine::AffineVectorLoadOp>(user))
      return &use != &op.getMemrefMutable();
    if (auto op = dyn_cast<enzymexla::CacheLoadOp>(user))
      return &use != &op.getMemrefMutable();

    // Memory stores / atomics / memory intrinsics do not capture the pointer
    // when it is only the destination/source buffer (not the stored value).
    if (auto op = dyn_cast<LLVM::StoreOp>(user))
      return &use != &op.getAddrMutable();
    if (auto op = dyn_cast<memref::StoreOp>(user))
      return &use != &op.getMemrefMutable();
    if (auto op = dyn_cast<affine::AffineStoreOp>(user))
      return &use != &op.getMemrefMutable();
    if (auto op = dyn_cast<affine::AffineVectorStoreOp>(user))
      return &use != &op.getMemrefMutable();
    if (auto op = dyn_cast<LLVM::AtomicRMWOp>(user))
      return &use != &op.getPtrMutable();
    if (auto op = dyn_cast<LLVM::AtomicCmpXchgOp>(user))
      return &use != &op.getPtrMutable();
    if (auto op = dyn_cast<memref::AtomicRMWOp>(user))
      return &use != &op.getMemrefMutable();
    if (auto op = dyn_cast<LLVM::MemcpyOp>(user))
      return &use != &op.getDstMutable() && &use != &op.getSrcMutable();
    if (auto op = dyn_cast<LLVM::MemcpyInlineOp>(user))
      return &use != &op.getDstMutable() && &use != &op.getSrcMutable();
    if (auto op = dyn_cast<LLVM::MemmoveOp>(user))
      return &use != &op.getDstMutable() && &use != &op.getSrcMutable();
    if (auto op = dyn_cast<LLVM::MemsetOp>(user))
      return &use != &op.getDstMutable();
    if (auto op = dyn_cast<memref::CopyOp>(user))
      return &use != &op.getSourceMutable() && &use != &op.getTargetMutable();

#if ENZYME_JAX_ENABLE_TRITON
    if (auto op = dyn_cast<triton::LoadOp>(user))
      return &use != &op.getPtrMutable();
    if (auto op = dyn_cast<triton::StoreOp>(user))
      return &use != &op.getPtrMutable();
    if (auto op = dyn_cast<triton::AtomicRMWOp>(user))
      return &use != &op.getPtrMutable();
    if (auto op = dyn_cast<triton::AtomicCASOp>(user))
      return &use != &op.getPtrMutable();
#endif

    // Non-capturing lifetime / metadata / dealloc ops with no pointer results.
    if (isa<LLVM::LifetimeStartOp, LLVM::LifetimeEndOp, memref::DeallocOp,
            memref::DimOp, memref::RankOp, enzymexla::NoopOp>(user))
      return false;

    // Address / view derivation ops forward the pointer to their results.
    if (auto op = dyn_cast<LLVM::GEPOp>(user)) {
      if (&use != &op.getBaseMutable())
        return true;
      derivedValues.push_back(op.getResult());
      return false;
    }
    if (isa<LLVM::BitcastOp, LLVM::AddrSpaceCastOp, LLVM::FreezeOp,
            enzymexla::Pointer2MemrefOp, enzymexla::Memref2PointerOp,
            LLVM::InsertValueOp, LLVM::ExtractValueOp>(user)) {
      llvm::append_range(derivedValues, user->getResults());
      return false;
    }
    if (isa<ViewLikeOpInterface, memref::CollapseShapeOp, memref::ExpandShapeOp,
            memref::TransposeOp, memref::AssumeAlignmentOp>(user)) {
      if (use.getOperandNumber() != 0)
        return true;
      llvm::append_range(derivedValues, user->getResults());
      return false;
    }
    if (auto op = dyn_cast<arith::SelectOp>(user)) {
      if (&use == &op.getConditionMutable())
        return true;
      derivedValues.push_back(op.getResult());
      return false;
    }
    if (auto op = dyn_cast<LLVM::SelectOp>(user)) {
      if (&use == &op.getConditionMutable())
        return true;
      derivedValues.push_back(op.getResult());
      return false;
    }

#if ENZYME_JAX_ENABLE_TRITON
    if (auto op = dyn_cast<triton::AddPtrOp>(user)) {
      if (&use != &op.getPtrMutable())
        return true;
      derivedValues.push_back(op.getResult());
      return false;
    }
    if (isa<triton::SplatOp, triton::BitcastOp, triton::BroadcastOp,
            triton::ExpandDimsOp, triton::ReshapeOp, triton::TransOp>(user)) {
      llvm::append_range(derivedValues, user->getResults());
      return false;
    }
#endif

    // Control-flow / region forwarding.
    if (auto branchOp = dyn_cast<BranchOpInterface>(user)) {
      for (unsigned succIdx = 0, e = branchOp->getNumSuccessors(); succIdx < e;
           ++succIdx) {
        SuccessorOperands succOps = branchOp.getSuccessorOperands(succIdx);
        MutableOperandRange forwarded = succOps.getMutableForwardedOperands();
        for (unsigned k = 0, fk = forwarded.size(); k < fk; ++k) {
          if (&forwarded[k] == &use) {
            unsigned argIdx = succOps.getProducedOperandCount() + k;
            derivedValues.push_back(
                branchOp->getSuccessor(succIdx)->getArgument(argIdx));
            return false;
          }
        }
      }
      return true;
    }
    if (auto forOp = dyn_cast<scf::ForOp>(user)) {
      unsigned ctrl = forOp.getNumControlOperands();
      if (use.getOperandNumber() < ctrl)
        return true;
      unsigned iterIdx = use.getOperandNumber() - ctrl;
      derivedValues.push_back(forOp.getRegionIterArgs()[iterIdx]);
      derivedValues.push_back(forOp.getResult(iterIdx));
      return false;
    }
    if (auto yieldOp = dyn_cast<scf::YieldOp>(user)) {
      unsigned idx = use.getOperandNumber();
      if (auto parentIf = dyn_cast<scf::IfOp>(yieldOp->getParentOp())) {
        derivedValues.push_back(parentIf.getResult(idx));
        return false;
      }
      if (auto parentFor = dyn_cast<scf::ForOp>(yieldOp->getParentOp())) {
        derivedValues.push_back(parentFor.getRegionIterArgs()[idx]);
        derivedValues.push_back(parentFor.getResult(idx));
        return false;
      }
      if (auto parentExec =
              dyn_cast<scf::ExecuteRegionOp>(yieldOp->getParentOp())) {
        derivedValues.push_back(parentExec.getResult(idx));
        return false;
      }
      return true;
    }
    if (auto retOp = dyn_cast<memref::AllocaScopeReturnOp>(user)) {
      if (auto scopeOp =
              dyn_cast<memref::AllocaScopeOp>(retOp->getParentOp())) {
        derivedValues.push_back(scopeOp.getResult(use.getOperandNumber()));
        return false;
      }
    }

    return true;
  }

  void analyzeFunctionArgumentCapture(
      FunctionOpInterface funcOp, BitVector &argCaptured,
      const DenseMap<SymbolRefAttr, BitVector> &funcArgCaptured) {
    for (unsigned i = 0, e = funcOp.getNumArguments(); i < e; ++i) {
      if (argCaptured.test(i))
        continue;
      Value arg = funcOp.getArgument(i);
      if (!isPointerType(arg))
        continue;

      SmallVector<Value> worklist = {arg};
      DenseSet<Value> visited = {arg};
      bool captured = false;
      while (!worklist.empty() && !captured) {
        Value cur = worklist.pop_back_val();
        for (OpOperand &use : cur.getUses()) {
          SmallVector<Value, 2> derived;
          if (analyzeOperandCapture(use, derived, funcArgCaptured)) {
            captured = true;
            break;
          }
          for (Value next : derived) {
            if (visited.insert(next).second)
              worklist.push_back(next);
          }
        }
      }
      if (captured)
        argCaptured.set(i);
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
    DenseMap<SymbolRefAttr, BitVector> funcArgCaptured;
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
        } else if (auto tcall = dyn_cast<triton_ext::TritonCallOp>(op)) {
          if (tcall.getXlaSideEffectFreeAttr()) {
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
      funcArgCaptured[symRef] = BitVector(funcOp.getNumArguments(), false);
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
          size_t argBitsBefore = 0;
          for (auto &bv : funcArgEffects[symRef])
            argBitsBefore += getNumEffects(bv);
          size_t capturedBefore = funcArgCaptured[symRef].count();
          analyzeFunctionArgumentMemoryEffects(funcOp, funcArgEffects[symRef],
                                               funcArgEffects);
          analyzeFunctionArgumentCapture(funcOp, funcArgCaptured[symRef],
                                         funcArgCaptured);
          size_t argBitsAfter = 0;
          for (auto &bv : funcArgEffects[symRef])
            argBitsAfter += getNumEffects(bv);
          auto &effects = funcEffects[symRef];
          size_t before = getNumEffects(effects);
          propagate(funcOp, effects);
          if (getNumEffects(effects) != before ||
              argBitsAfter != argBitsBefore ||
              funcArgCaptured[symRef].count() != capturedBefore)
            changed = true;
        }
      }

      // At this point if we haven't converged, we assume effects for all
      if (changed) {
        for (auto &[symbol, effects] : funcEffects)
          insertMemoryEffects(effects);
        for (auto &[symbol, argBVs] : funcArgEffects)
          for (auto &bv : argBVs)
            insertMemoryEffects(bv);
        for (auto &[symbol, captured] : funcArgCaptured)
          captured.set(0, captured.size());
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
        analyzeFunctionArgumentCapture(funcOp, funcArgCaptured[symRef],
                                       funcArgCaptured);
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
      auto &argCaptured = funcArgCaptured[symbol];
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
          if (!argCaptured.test(i)) {
            funcOp.setArgAttr(i, LLVM::LLVMDialect::getNoCaptureAttrName(),
                              builder.getUnitAttr());
          }
        }
      }
    }
  }
};

} // namespace
