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
#include "mlir/IR/AttrTypeSubElements.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/DenseSet.h"

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

    // TODO: Use CallableOpInterface.
    SmallVector<LLVM::LLVMFuncOp> funcOps;
    moduleOp->walk([&](LLVM::LLVMFuncOp funcOp) { funcOps.push_back(funcOp); });

    // Collect the call sites in a single walk over the module, along with the
    // symbols referenced other than as the callee of a call (address-of
    // operations, symbol references held in attributes, ...). Such a reference
    // may observe the identity of a function, e.g. by comparing its address to
    // the address of another function, so a function referenced this way is
    // never erased.
    SymbolTableCollection symbolTables;
    SmallVector<std::pair<CallOpInterface, Operation *>> calls;
    DenseSet<Operation *> hasNonCallUses;
    moduleOp->walk([&](Operation *op) {
      DictionaryAttr attrs = op->getAttrDictionary();
      if (attrs.empty())
        return;
      auto callOp = dyn_cast<CallOpInterface>(op);
      SymbolRefAttr callee = callOp ? dyn_cast_if_present<SymbolRefAttr>(
                                          callOp.getCallableForCallee())
                                    : SymbolRefAttr();
      AttrTypeWalker walker;
      walker.addWalk([&](SymbolRefAttr ref) {
        if (Operation *symbol = symbolTables.lookupNearestSymbolFrom(op, ref)) {
          if (ref == callee) {
            calls.emplace_back(callOp, symbol);
            // Any other occurrence of the callee is not a call.
            callee = {};
          } else {
            hasNonCallUses.insert(symbol);
          }
        }
        // Never recurse into a reference: its nested parts are not references
        // resolvable from `op` on their own.
        return WalkResult::skip();
      });
      (void)walker.walk<WalkOrder::PreOrder>(attrs);
    });

    // Maps each erased duplicate function to the function replacing it.
    DenseMap<Operation *, LLVM::LLVMFuncOp> replacements;
    SmallVector<bool> grouped(funcOps.size(), false);
    for (size_t i = 0, e = funcOps.size(); i < e; ++i) {
      if (grouped[i])
        continue;
      // Find *all* the functions equivalent to funcOps[i]. Only functions in
      // the same symbol table can replace each other.
      SmallVector<LLVM::LLVMFuncOp> group = {funcOps[i]};
      for (size_t j = i + 1; j < e; ++j) {
        if (grouped[j] ||
            funcOps[i]->getParentOp() != funcOps[j]->getParentOp())
          continue;
        if (areEquivalent(funcOps[i], funcOps[j])) {
          group.push_back(funcOps[j]);
          grouped[j] = true;
        }
      }
      // Calls do not observe the identity of their callee, so they can be
      // redirected to a function with non-call uses. Prefer such a function to
      // keep the number of remaining functions minimal.
      auto *keptIt = llvm::find_if(group, [&](LLVM::LLVMFuncOp funcOp) {
        return hasNonCallUses.contains(funcOp);
      });
      LLVM::LLVMFuncOp kept = keptIt != group.end() ? *keptIt : group.front();
      for (LLVM::LLVMFuncOp funcOp : group)
        if (funcOp != kept && !hasNonCallUses.contains(funcOp))
          replacements[funcOp] = kept;
    }
    if (replacements.empty())
      return;

    // Erased duplicates only have call uses, which are redirected to the kept
    // function.
    for (auto [callOp, callee] : calls) {
      auto it = replacements.find(callee);
      if (it == replacements.end())
        continue;
      // The duplicate and its replacement live in the same symbol table, so
      // only the leaf reference changes.
      auto ref = cast<SymbolRefAttr>(callOp.getCallableForCallee());
      auto leaf = FlatSymbolRefAttr::get(it->second.getSymNameAttr());
      SmallVector<FlatSymbolRefAttr> nested(ref.getNestedReferences());
      if (nested.empty()) {
        callOp.setCalleeFromCallable(leaf);
        continue;
      }
      nested.back() = leaf;
      callOp.setCalleeFromCallable(
          SymbolRefAttr::get(ref.getRootReference(), nested));
    }

    for (Operation *funcOp : llvm::make_first_range(replacements))
      funcOp->erase();
  }
};
} // end anonymous namespace
