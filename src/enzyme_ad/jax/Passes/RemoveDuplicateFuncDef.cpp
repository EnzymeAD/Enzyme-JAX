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
#include "mlir/IR/SymbolTable.h"

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

    // Neither of the functions may be a declaration.
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
    SymbolTableCollection symbolTables;
    SymbolUserMap userMap(symbolTables, moduleOp);

    // A reference other than the callee of a call (address-of operations,
    // symbol references held in attributes, ...) may observe the identity of a
    // function, e.g. by comparing its address to the address of another
    // function. Only functions that are exclusively called can be erased.
    auto isOnlyCalled = [&](LLVM::LLVMFuncOp funcOp) {
      return llvm::all_of(userMap.getUsers(funcOp), [&](Operation *user) {
        auto callOp = dyn_cast<CallOpInterface>(user);
        auto callee = callOp ? dyn_cast_if_present<SymbolRefAttr>(
                                   callOp.getCallableForCallee())
                             : SymbolRefAttr();
        return callee &&
               symbolTables.lookupNearestSymbolFrom(user, callee) == funcOp;
      });
    };

    // TODO: Use CallableOpInterface.
    SmallVector<LLVM::LLVMFuncOp> funcOps;
    moduleOp->walk([&](LLVM::LLVMFuncOp funcOp) { funcOps.push_back(funcOp); });

    // Erasure is delayed until the end: `userMap` also records the users
    // nested in the bodies of erased functions.
    SmallVector<LLVM::LLVMFuncOp> toErase;
    for (size_t i = 0, e = funcOps.size(); i < e; ++i) {
      // Already part of an earlier group.
      if (!funcOps[i])
        continue;
      // Find *all* the functions equivalent to funcOps[i]. Only functions in
      // the same symbol table can replace each other.
      SmallVector<LLVM::LLVMFuncOp> group = {funcOps[i]};
      for (size_t j = i + 1; j < e; ++j) {
        if (funcOps[j] &&
            funcOps[i]->getParentOp() == funcOps[j]->getParentOp() &&
            areEquivalent(funcOps[i], funcOps[j])) {
          group.push_back(funcOps[j]);
          funcOps[j] = nullptr;
        }
      }
      // Calls do not observe the identity of their callee, so they can be
      // redirected to a function with non-call uses. Prefer keeping such a
      // function to minimize the number of remaining functions.
      auto *keptIt = llvm::find_if_not(group, isOnlyCalled);
      LLVM::LLVMFuncOp kept = keptIt != group.end() ? *keptIt : group.front();
      for (LLVM::LLVMFuncOp funcOp : group) {
        if (funcOp == kept || !isOnlyCalled(funcOp))
          continue;
        userMap.replaceAllUsesWith(funcOp, kept.getSymNameAttr());
        toErase.push_back(funcOp);
      }
    }

    for (LLVM::LLVMFuncOp funcOp : toErase)
      funcOp.erase();
  }
};
} // end anonymous namespace
