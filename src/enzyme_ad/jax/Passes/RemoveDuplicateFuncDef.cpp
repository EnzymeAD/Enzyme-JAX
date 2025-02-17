//===- RemoveDuplicateFuncDef.cpp - Remove duplicate fund def -------------- //
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//
//
// This file implements a pass to remove duplicate function definitions.
//===---------------------------------------------------------------------===//

#include "src/enzyme_ad/jax/Dialect/Ops.h"
#include "src/enzyme_ad/jax/Passes/Passes.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_REMOVEDUPLICATEFUNCDEFPASS
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

using namespace mlir;
using namespace mlir::enzyme;

namespace {
struct RemoveDuplicateFuncDefPass
    : public enzyme::impl::RemoveDuplicateFuncDefPassBase<
          RemoveDuplicateFuncDefPass> {
  using RemoveDuplicateFuncDefPassBase::RemoveDuplicateFuncDefPassBase;

  static bool areEquivalent(LLVM::LLVMFuncOp funcOp1,
                            LLVM::LLVMFuncOp funcOp2) {
    // Same function.
    if (funcOp1 == funcOp2)
      return true;

    // Both of the functions must be declarations.
    if (funcOp1.isDeclaration() || funcOp2.isDeclaration())
      return false;

    // Check arguments.
    if (funcOp1.getNumArguments() != funcOp2.getNumArguments())
      return false;
    if (funcOp1.getArgumentTypes() != funcOp2.getArgumentTypes())
      return false;

    // Check return trypes.
    if (funcOp1.getResultTypes() != funcOp2.getResultTypes())
      return false;

    // Discardable attributes equivalence.
    if (funcOp1->getDiscardableAttrDictionary() !=
        funcOp2->getDiscardableAttrDictionary())
      return false;

    Region *body1 = funcOp1.getCallableRegion();
    Region *body2 = funcOp2.getCallableRegion();
    if (!body1 || !body2)
      return false;
    return OperationEquivalence::isRegionEquivalentTo(
        body1, body2, OperationEquivalence::IgnoreLocations);
  }

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    SymbolTable symbolTable(moduleOp);

    // TODO: Use CallableOpInterface.
    SmallVector<LLVM::LLVMFuncOp> funcOps;
    moduleOp->walk([&](LLVM::LLVMFuncOp funcOp) { funcOps.push_back(funcOp); });

    DenseMap<StringRef, StringRef> equivalenceMap;
    SmallVector<CallableOpInterface> toRemove;
    for (size_t i = 0, e = funcOps.size(); i < e; ++i) {
      // Already found equivalent.
      if (equivalenceMap.count(funcOps[i].getSymName()))
        continue;
      // Find *all* the possible equivalent functions to funcOps[i].
      for (size_t j = i + 1; j < e; ++j) {
        if (areEquivalent(funcOps[i], funcOps[j])) {
          equivalenceMap[funcOps[j].getSymName()] = funcOps[i].getSymName();
          toRemove.push_back(funcOps[j]);
          // llvm::errs() << funcOps[j].getSymName() << " is equivalent to "
          //              << funcOps[i].getSymName() << "\n";
        }
      }
    }

    SmallVector<SymbolUserOpInterface> callOps;
    moduleOp->walk(
        [&](SymbolUserOpInterface callOp) { callOps.push_back(callOp); });

    for (SymbolUserOpInterface symbolUserOp : callOps) {
      if (auto kernelOp =
              dyn_cast<enzymexla::KernelCallOp>(symbolUserOp.getOperation())) {
        auto sym = kernelOp.getFn();
        Operation *op = symbolTable.lookup(sym);
        assert(op && "Kernel function not found");
        auto funcOp = dyn_cast<FunctionOpInterface>(op);
        assert(funcOp && "Kernel function is not a function");
        if (equivalenceMap.count(funcOp.getNameAttr()))
          kernelOp.setFn(equivalenceMap[funcOp.getNameAttr()]);
      }
      if (auto callOp =
              dyn_cast<CallOpInterface>(symbolUserOp.getOperation())) {
        SymbolRefAttr sym = llvm::dyn_cast_if_present<SymbolRefAttr>(
            callOp.getCallableForCallee());
        if (!sym)
          continue;
        Operation *op = SymbolTable::lookupNearestSymbolFrom(callOp, sym);
        assert(op && "Kernel function not found");
        auto funcOp = dyn_cast<FunctionOpInterface>(op);
        assert(funcOp && "Kernel function is not a function");
        if (equivalenceMap.count(funcOp.getNameAttr()))
          callOp.setCalleeFromCallable(SymbolRefAttr::get(
              op->getContext(), equivalenceMap[funcOp.getNameAttr()]));
      }
    }

    // At this point it should be safe to remove the duplicate functions.
    for (CallableOpInterface callableOp : toRemove)
      callableOp.erase();
  }
};
} // end anonymous namespace