#include "src/enzyme_ad/jax/Passes/Distributed/PartialOrder.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/AnalysisManager.h"

namespace mlir::enzyme::distributed {

namespace {

static bool isMainSymbol(Operation *op) {
  auto symName =
      op->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName());
  return symName && symName.getValue() == "main";
}

static bool setOrInvalidateMain(Operation *candidate, Operation *&currentMain,
                                bool &valid) {
  if (!candidate || !isMainSymbol(candidate)) {
    return false;
  }
  if (currentMain && currentMain != candidate) {
    valid = false;
    return true;
  }
  currentMain = candidate;
  return true;
}

} // namespace

void SSABlockPartialOrderAnalysis::buildPartialOrder() {
  for (Operation &op : block->getOperations()) {
    for (Value result : op.getResults()) {
      for (Operation *user : result.getUsers()) {
        if (user->getBlock() != block) {
          continue;
        }
        partial_order.addOrder(&op, user);
      }
    }
  }
  partial_order.close();
}

MainFunctionSSABlockPartialOrderAnalysis::
    MainFunctionSSABlockPartialOrderAnalysis(ModuleOp module,
                                             AnalysisManager &analysisManager) {
  // NOTE: This wrapper resolves @main directly from the current ModuleOp
  // instead of chaining through FindMainFunctionAnalysis. In the current
  // monolithic pass, step 1 rewrites func.main into distributed.main before
  // step 6 asks for this analysis, and reusing a cached main-op lookup can
  // observe stale state.
  //
  // When these steps are split into separate passes, pass boundaries will
  // provide clean analysis invalidation/recomputation. At that point, it
  // should be safe to revisit wiring this through FindMainFunctionAnalysis if
  // desired for consistency.
  Operation *mainOp = nullptr;
  for (func::FuncOp funcOp : module.getOps<func::FuncOp>()) {
    setOrInvalidateMain(funcOp.getOperation(), mainOp, valid);
    if (!valid) {
      return;
    }
  }

  for (DistributedFunctionOp distributedFuncOp :
       module.getOps<DistributedFunctionOp>()) {
    setOrInvalidateMain(distributedFuncOp.getOperation(), mainOp, valid);
    if (!valid) {
      return;
    }
  }

  if (!mainOp) {
    valid = false;
    return;
  }

  analysis = &analysisManager.getChildAnalysis<SSABlockPartialOrderAnalysis>(
      mainOp);
  if (!analysis->isValid()) {
    valid = false;
  }
}
} // namespace mlir::enzyme::distributed