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

    // Maps each duplicate function to the name of the function replacing it.
    DenseMap<Operation *, StringAttr> replacements;
    for (size_t i = 0, e = funcOps.size(); i < e; ++i) {
      // Already found equivalent.
      if (replacements.count(funcOps[i]))
        continue;
      // Find *all* the possible equivalent functions to funcOps[i]. Only
      // functions in the same symbol table can replace each other.
      for (size_t j = i + 1; j < e; ++j) {
        if (replacements.count(funcOps[j]) ||
            funcOps[i]->getParentOp() != funcOps[j]->getParentOp())
          continue;
        if (areEquivalent(funcOps[i], funcOps[j]))
          replacements[funcOps[j]] = funcOps[i].getSymNameAttr();
      }
    }
    if (replacements.empty())
      return;

    // Redirect every symbol reference to a duplicate, not only call sites:
    // kernel launches, address-of operations and other symbol users would
    // otherwise be left dangling once the duplicates are erased. This is done
    // in a single walk rather than with SymbolTable::replaceAllSymbolUses per
    // duplicate, which would walk the whole module for each of them.
    SymbolTableCollection symbolTables;
    moduleOp->walk([&](Operation *op) {
      if (op->getAttrDictionary().empty())
        return;
      // References resolve relative to `op`, so the replacer (which caches
      // its results) cannot be shared across operations.
      AttrTypeReplacer replacer;
      // Never recurse into a reference: its nested parts are not references
      // resolvable from `op` on their own.
      replacer.addReplacement([&](SymbolRefAttr ref)
                                  -> std::pair<Attribute, WalkResult> {
        Operation *symbol = symbolTables.lookupNearestSymbolFrom(op, ref);
        auto it = replacements.find(symbol);
        if (it == replacements.end())
          return {ref, WalkResult::skip()};
        auto leaf = FlatSymbolRefAttr::get(it->second);
        if (ref.getNestedReferences().empty())
          return {leaf, WalkResult::skip()};
        SmallVector<FlatSymbolRefAttr> nested(ref.getNestedReferences());
        nested.back() = leaf;
        return {SymbolRefAttr::get(ref.getRootReference(), nested),
                WalkResult::skip()};
      });
      replacer.replaceElementsIn(op);
    });

    // At this point it should be safe to remove the duplicate functions.
    for (Operation *funcOp : llvm::make_first_range(replacements))
      funcOp->erase();
  }
};
} // end anonymous namespace
