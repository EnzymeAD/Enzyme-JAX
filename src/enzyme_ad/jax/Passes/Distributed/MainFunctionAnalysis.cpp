#include "src/enzyme_ad/jax/Passes/Distributed/MainFunctionAnalysis.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/SymbolTable.h"

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

FindMainFunctionAnalysis::FindMainFunctionAnalysis(ModuleOp module) {
  for (func::FuncOp funcOp : module.getOps<func::FuncOp>()) {
    setOrInvalidateMain(funcOp.getOperation(), mainFunctionOp, valid);
    if (!valid) {
      return;
    }
  }

  for (DistributedFunctionOp distributedFuncOp :
       module.getOps<DistributedFunctionOp>()) {
    setOrInvalidateMain(distributedFuncOp.getOperation(), mainFunctionOp,
                        valid);
    if (!valid) {
      return;
    }
  }

  if (!mainFunctionOp) {
    valid = false;
  }
}

func::FuncOp FindMainFunctionAnalysis::getMainFuncOp() const {
  return dyn_cast_or_null<func::FuncOp>(mainFunctionOp);
}

DistributedFunctionOp
FindMainFunctionAnalysis::getMainDistributedFunctionOp() const {
  return dyn_cast_or_null<DistributedFunctionOp>(mainFunctionOp);
}

Block *FindMainFunctionAnalysis::getMainBlock() const {
  if (auto funcOp = getMainFuncOp()) {
    Region &body = funcOp.getBody();
    if (body.hasOneBlock()) {
      return &body.front();
    }
    return nullptr;
  }
  if (auto distributedFuncOp = getMainDistributedFunctionOp()) {
    Region &body = distributedFuncOp.getBody();
    if (body.hasOneBlock()) {
      return &body.front();
    }
    return nullptr;
  }
  return nullptr;
}

} // namespace mlir::enzyme::distributed