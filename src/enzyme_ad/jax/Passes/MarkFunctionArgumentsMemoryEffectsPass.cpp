#include "src/enzyme_ad/jax/Passes/Passes.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include <queue>

#define DEBUG_TYPE "mark-func-args-memory-effects"

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_MARKFUNCTIONARGUMENTSMEMORYEFFECTSPASS
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

namespace {

using namespace mlir;
using namespace mlir::enzyme;

void setAllMemoryEffects(BitVector &effects) {
  for (int i = 0; i < effects.size(); i++) {
    effects.set(i);
  }
}

// Since this pass is mostly used for kernels, we don't recurse into
// callopinterfaces for now and assume those imply all memory effects. We should
// revisit this later.
void handleCallOpInterface(CallOpInterface callOp, OpOperand *operand,
                           BitVector &effects) {
  setAllMemoryEffects(effects);
}

void analyzeMemoryEffects(Operation *op, OpOperand *operand,
                          BitVector &effects) {
  auto memInterface = dyn_cast<MemoryEffectOpInterface>(op);
  if (!memInterface) {
    setAllMemoryEffects(effects);
    return;
  }

  SmallVector<MemoryEffects::EffectInstance> memEffects;
  memInterface.getEffects(memEffects);

  for (const auto &effect : memEffects) {
    if (effect.getValue() && effect.getValue() == operand->get()) {
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

LogicalResult annotateFunctionArguments(FunctionOpInterface funcOp) {
  auto *ctx = funcOp->getContext();
  OpBuilder builder(ctx);

  SmallVector<BitVector, 4> argEffects;
  DenseMap<Value, unsigned> valueToArgIndex;
  for (unsigned i = 0; i < funcOp.getNumArguments(); i++) {
    argEffects.push_back(BitVector(4, 0));
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

      analyzeMemoryEffects(user, &use, argEffects[argIndex]);

      if (auto callOp = dyn_cast<CallOpInterface>(user)) {
        handleCallOpInterface(callOp, &use, argEffects[argIndex]);
      }

      for (auto result : user->getResults()) {
        if (visited.insert(result).second) {
          valueToArgIndex[result] = argIndex;
          worklist.push(result);
        }
      }
    }
  }

  for (unsigned i = 0; i < funcOp.getNumArguments(); i++) {
    auto effects = argEffects[i];

    SmallVector<Attribute> effectsAttrs;
    bool readOnly = true;
    bool writeOnly = true;
    for (int i = 0; i < effects.size(); i++) {
      if (effects[i]) {
        if (i == 0) {
          writeOnly = false;
          effectsAttrs.push_back(builder.getStringAttr("read"));
        } else if (i == 1) {
          readOnly = false;
          effectsAttrs.push_back(builder.getStringAttr("write"));
        } else if (i == 2) {
          writeOnly = false;
          readOnly = false;
          effectsAttrs.push_back(builder.getStringAttr("allocate"));
        } else if (i == 3) {
          writeOnly = false;
          readOnly = false;
          effectsAttrs.push_back(builder.getStringAttr("free"));
        } else {
          assert(false && "unknown memory effect");
        }
      }
    }

    funcOp.setArgAttr(i, "enzymexla.memory_effects",
                      builder.getArrayAttr(effectsAttrs));

    // Set the llvm attributes
    if (readOnly) {
      funcOp.setArgAttr(i, LLVM::LLVMDialect::getReadonlyAttrName(),
                        builder.getUnitAttr());
    }
    if (writeOnly) {
      funcOp.setArgAttr(i, LLVM::LLVMDialect::getWriteOnlyAttrName(),
                        builder.getUnitAttr());
    }
  }

  return success();
}

struct MarkFunctionArgumentsMemoryEffectsPass
    : public enzyme::impl::MarkFunctionArgumentsMemoryEffectsPassBase<
          MarkFunctionArgumentsMemoryEffectsPass> {
  void runOnOperation() override {
    FunctionOpInterface funcOp = getOperation();

    if (funcOp.isExternal() || funcOp.getNumArguments() == 0)
      return;

    if (failed(annotateFunctionArguments(funcOp))) {
      funcOp->emitError() << "failed to annotate function arguments";
      signalPassFailure();
      return;
    }
  }
};

} // namespace
